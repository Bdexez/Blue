"""Mesures pondérées par les cibles observées, et diagnostics des prévisions."""

from contextlib import nullcontext
from collections import Counter
import math

import torch

from .binomial import wilson_interval
from .observed_objective import LOSS_WEIGHTS
from .routing_metrics import summarize_expert_usage
from .probability_metrics import ProbabilityMetrics
from .series_probability import log_probabilities
from .streaming_rmse import StreamingRMSE


class ForecastMetrics:
    """Accumulateurs des métriques de prévision : erreurs, quantiles, intervalles bruts et calibrés."""

    def __init__(self, config):
        self.config = config
        self.forecast_count = 0
        self.complete_forecasts = self.raw_joint_covered = self.calibrated_joint_covered = 0
        self.absolute = self.covered = self.width = 0.0
        self.rmse = StreamingRMSE()
        self.horizon_error = torch.zeros(self.config.forecast_horizon, dtype=torch.float64)
        self.horizon_count = torch.zeros_like(self.horizon_error)
        self.feature_count = torch.zeros(self.config.series_features, dtype=torch.float64)
        self.feature_error = torch.zeros_like(self.feature_count)
        self.cell_count = torch.zeros(self.config.forecast_horizon, self.config.series_features, dtype=torch.float64)
        self.cell_error = torch.zeros_like(self.cell_count)
        self.feature_width = torch.zeros_like(self.feature_count)
        self.calibrated_feature_width = torch.zeros_like(self.feature_count)
        self.feature_covered = torch.zeros_like(self.feature_count)
        self.calibrated_feature_covered = torch.zeros_like(self.feature_count)
        self.feature_below = torch.zeros_like(self.feature_count)
        self.feature_above = torch.zeros_like(self.feature_count)
        self.calibrated_feature_below = torch.zeros_like(self.feature_count)
        self.calibrated_feature_above = torch.zeros_like(self.feature_count)
        self.cell_covered = torch.zeros_like(self.cell_count)
        self.calibrated_cell_covered = torch.zeros_like(self.cell_count)
        self.quantile_hits = torch.zeros(len(self.config.quantiles), dtype=torch.float64)
        self.quantile_pinball = torch.zeros_like(self.quantile_hits)
        self.feature_quantile_hits = torch.zeros(self.config.series_features, len(self.config.quantiles), dtype=torch.float64)
        self.feature_quantile_pinball = torch.zeros_like(self.feature_quantile_hits)
        self.interval_covered = torch.zeros_like(self.horizon_error)
        self.interval_width = torch.zeros_like(self.horizon_error)
        self.calibrated_covered = torch.zeros_like(self.horizon_error)
        self.calibrated_width = torch.zeros_like(self.horizon_error)
        self.calibrated_score = 0.0
        self.raw_score = 0.0

    def by_horizon(self, values):
        return [total.item() / count.item() if count else None
                for total, count in zip(values, self.horizon_count)]

    def by_feature(self, values):
        return [total.item() / count.item() if count else None
                for total, count in zip(values, self.feature_count)]

    def by_cell(self, values):
        return [[total.item() / count.item() if count else None for total, count in zip(row, counts)]
                for row, counts in zip(values, self.cell_count)]

    def by_feature_quantile(self, values):
        return [[total.item() / count.item() if count else None for total in row]
                for row, count in zip(values, self.feature_count)]

    def update(self, target, forecast, calibration=None):
        """Ajouter un lot de prévisions ; renvoie le nombre de cibles observées."""
        observed = torch.isfinite(target)
        count = observed.sum().item()
        self.forecast_count += count
        target = target.double()
        predictions = forecast.double()
        median = predictions[..., self.config.quantiles.index(0.5)]
        error = torch.where(observed, median - target, 0)
        self.absolute += error.abs().sum().item()
        self.rmse.update(error)
        self.horizon_error += error.abs().sum((0, 2)).cpu()
        self.horizon_count += observed.sum((0, 2)).cpu()
        self.feature_count += observed.sum((0, 1)).cpu()
        self.feature_error += error.abs().sum((0, 1)).cpu()
        self.cell_count += observed.sum(0).cpu()
        self.cell_error += error.abs().sum(0).cpu()
        levels = predictions.new_tensor(self.config.quantiles)
        residual = torch.where(observed.unsqueeze(-1), target.unsqueeze(-1) - predictions, 0)
        hits = (target.unsqueeze(-1) <= predictions) & observed.unsqueeze(-1)
        pinball = torch.maximum(levels * residual, (levels - 1) * residual)
        self.quantile_hits += hits.sum((0, 1, 2)).cpu()
        self.quantile_pinball += pinball.sum((0, 1, 2)).cpu()
        self.feature_quantile_hits += hits.sum((0, 1)).cpu()
        self.feature_quantile_pinball += pinball.sum((0, 1)).cpu()
        if len(self.config.quantiles) > 1:
            lower, upper = predictions[..., 0], predictions[..., -1]
            complete = observed.flatten(1).all(-1)
            self.complete_forecasts += complete.sum().item()
            inside = (target >= lower) & (target <= upper)
            self.raw_joint_covered += (inside.flatten(1).all(-1) & complete).sum().item()
            self.covered += ((target >= lower) & (target <= upper) & observed).sum().item()
            self.width += torch.where(observed, upper - lower, 0).double().sum().item()
            self.interval_covered += ((target >= lower) & (target <= upper) & observed).sum((0, 2)).cpu()
            self.feature_covered += (inside & observed).sum((0, 1)).cpu()
            self.cell_covered += (inside & observed).sum(0).cpu()
            self.feature_below += ((target < lower) & observed).sum((0, 1)).cpu()
            self.feature_above += ((target > upper) & observed).sum((0, 1)).cpu()
            self.interval_width += torch.where(observed, upper - lower, 0).double().sum((0, 2)).cpu()
            self.feature_width += torch.where(observed, upper - lower, 0).sum((0, 1)).cpu()
            alpha = 1 - (self.config.quantiles[-1] - self.config.quantiles[0])
            score = upper - lower + 2 / alpha * (
                (lower - target).clamp_min(0) + (target - upper).clamp_min(0))
            self.raw_score += torch.where(observed, score, 0).double().sum().item()
            if calibration is not None:
                interval = calibration.interval(forecast)
                low, high = interval.double().unbind(-1)
                inside = (target >= low) & (target <= high)
                self.calibrated_joint_covered += (inside.flatten(1).all(-1) & complete).sum().item()
                self.calibrated_covered += ((target >= low) & (target <= high) & observed).sum((0, 2)).cpu()
                self.calibrated_feature_covered += (inside & observed).sum((0, 1)).cpu()
                self.calibrated_cell_covered += (inside & observed).sum(0).cpu()
                self.calibrated_feature_below += ((target < low) & observed).sum((0, 1)).cpu()
                self.calibrated_feature_above += ((target > high) & observed).sum((0, 1)).cpu()
                self.calibrated_width += torch.where(observed, high - low, 0).double().sum((0, 2)).cpu()
                self.calibrated_feature_width += torch.where(observed, high - low, 0).sum((0, 1)).cpu()
                if calibration.scope == "marginal":
                    alpha = 1 - calibration.coverage
                    if calibration.lower_tail_fraction in (None, 0.5):
                        score = high - low + 2 / alpha * (
                            (low - target).clamp_min(0) + (target - high).clamp_min(0))
                    else:
                        fraction = calibration.lower_tail_fraction
                        score = (high - low + (low - target).clamp_min(0) / (alpha * fraction)
                                 + (target - high).clamp_min(0) / (alpha * (1 - fraction)))
                    self.calibrated_score += torch.where(observed, score, 0).double().sum().item()
        return count

    def report(self, result, calibration=None):
        """Écrire les métriques agrégées dans ``result``."""
        result["forecast_observations"] = self.forecast_count
        result["forecast_observations_by_horizon"] = self.horizon_count.long().tolist()
        result["forecast_observations_by_feature"] = self.feature_count.long().tolist()
        result["forecast_observations_by_horizon_feature"] = self.cell_count.long().tolist()
        result["forecast_mae"] = self.absolute / self.forecast_count if self.forecast_count else None
        result["forecast_rmse"] = self.rmse.result(self.forecast_count)
        result["forecast_mae_by_horizon"] = self.by_horizon(self.horizon_error)
        result["forecast_mae_by_feature"] = self.by_feature(self.feature_error)
        result["forecast_mae_by_horizon_feature"] = self.by_cell(self.cell_error)
        result["forecast_quantile_levels"] = list(self.config.quantiles)
        result["forecast_quantile_coverage"] = [value.item() / self.forecast_count if self.forecast_count else None
                                                for value in self.quantile_hits]
        result["forecast_pinball_by_quantile"] = [value.item() / self.forecast_count if self.forecast_count else None
                                                  for value in self.quantile_pinball]
        result["forecast_quantile_coverage_by_feature"] = self.by_feature_quantile(self.feature_quantile_hits)
        result["forecast_pinball_by_feature_quantile"] = self.by_feature_quantile(self.feature_quantile_pinball)
        if len(self.config.quantiles) > 1:
            result["forecast_complete_examples"] = self.complete_forecasts
            result["forecast_joint_interval_coverage"] = (
                self.raw_joint_covered / self.complete_forecasts if self.complete_forecasts else None)
            result["forecast_interval_coverage"] = self.covered / self.forecast_count if self.forecast_count else None
            result["forecast_interval_width"] = self.width / self.forecast_count if self.forecast_count else None
            result["forecast_nominal_coverage"] = self.config.quantiles[-1] - self.config.quantiles[0]
            result["forecast_interval_score"] = self.raw_score / self.forecast_count if self.forecast_count else None
            result["forecast_interval_coverage_by_horizon"] = self.by_horizon(self.interval_covered)
            result["forecast_interval_coverage_by_feature"] = self.by_feature(self.feature_covered)
            result["forecast_interval_coverage_by_horizon_feature"] = self.by_cell(self.cell_covered)
            result["forecast_interval_below_by_feature"] = self.by_feature(self.feature_below)
            result["forecast_interval_above_by_feature"] = self.by_feature(self.feature_above)
            result["forecast_interval_width_by_horizon"] = self.by_horizon(self.interval_width)
            result["forecast_interval_width_by_feature"] = self.by_feature(self.feature_width)
            if calibration is not None:
                result["calibrated_interval_coverage"] = self.calibrated_covered.sum().item() / self.forecast_count if self.forecast_count else None
                result["calibrated_interval_width"] = self.calibrated_width.sum().item() / self.forecast_count if self.forecast_count else None
                result["calibrated_interval_score"] = (
                    self.calibrated_score / self.forecast_count if self.forecast_count and calibration.scope == "marginal" else None)
                result["calibrated_nominal_coverage"] = calibration.coverage
                result["calibrated_coverage_scope"] = calibration.scope
                result["calibrated_score_scale"] = calibration.score_scale
                if calibration.lower_tail_fraction is not None:
                    result["calibrated_lower_tail_fraction"] = calibration.lower_tail_fraction
                if calibration.score_scale == "width":
                    result["calibrated_scale_floor"] = calibration.scale_floor
                result["calibrated_joint_interval_coverage"] = (
                    self.calibrated_joint_covered / self.complete_forecasts if self.complete_forecasts else None)
                result["calibrated_interval_coverage_by_horizon"] = self.by_horizon(self.calibrated_covered)
                result["calibrated_interval_coverage_by_feature"] = self.by_feature(self.calibrated_feature_covered)
                result["calibrated_interval_coverage_by_horizon_feature"] = self.by_cell(self.calibrated_cell_covered)
                result["calibrated_interval_below_by_feature"] = self.by_feature(self.calibrated_feature_below)
                result["calibrated_interval_above_by_feature"] = self.by_feature(self.calibrated_feature_above)
                result["calibrated_interval_width_by_horizon"] = self.by_horizon(self.calibrated_width)
                result["calibrated_interval_width_by_feature"] = self.by_feature(self.calibrated_feature_width)


def evaluate(model, batches, amp="none", *, calibration=None, series_classifier=None, series_probability=None, series_set=None, iid_intervals=False, series_gate=None, series_gate_guard=None):
    if amp not in ("none", "bf16", "fp16"):
        raise ValueError("amp doit être none, bf16 ou fp16")
    if calibration is not None:
        calibration.validate(model.config, amp=amp)
    if series_classifier is not None:
        series_classifier.validate(model.config, amp=amp)
    if series_probability is not None:
        if series_classifier is None:
            raise ValueError("La calibration de probabilités nécessite une tête temporelle")
        series_probability.validate(series_classifier, amp=amp)
    if series_set is not None:
        if series_classifier is None:
            raise ValueError("Les ensembles conformes nécessitent une tête temporelle")
        series_set.validate(series_classifier, series_probability, amp=amp)
    if series_gate is not None:
        if series_classifier is None:
            raise ValueError("La porte nécessite une tête temporelle")
        series_gate.validate(series_classifier, amp=amp)
    if series_gate_guard is not None:
        if series_gate is None or series_classifier is None:
            raise ValueError("La protection nécessite une porte et une tête temporelles")
        series_gate_guard.validate(series_gate, series_classifier, amp=amp)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    totals, counts = {}, {}
    auxiliary_totals, auxiliary_counts = {}, {}
    expert_totals, expert_tokens = None, 0
    correct = class_count = 0
    series_class_correct = series_class_count = 0
    series_class_confusion = Counter() if series_classifier is not None else None
    probability_raw = ProbabilityMetrics() if series_probability is not None else None
    probability_calibrated = ProbabilityMetrics() if series_probability is not None else None
    gate_examples = gate_selected = gate_labeled = gate_correct = 0
    gate_selected_labeled = gate_selected_correct = 0
    gate_unguarded_correct = gate_prevented = gate_harms = gate_benefits = 0
    set_examples = set_size = set_singletons = set_labeled = set_covered = 0
    set_labeled_singletons = set_correct_singletons = 0
    set_class_counts = torch.zeros(model.config.num_classes, dtype=torch.long)
    set_class_covered = torch.zeros_like(set_class_counts)
    forecasts = ForecastMetrics(model.config)
    multi_token_count = 0
    autocast = (lambda: nullcontext()) if amp == "none" else (
        lambda: torch.autocast(device.type, dtype=torch.bfloat16 if amp == "bf16" else torch.float16)
    )

    def accumulate(name, value, weight):
        totals[name] = totals.get(name, 0.0) + float(value) * weight
        counts[name] = counts.get(name, 0) + weight

    try:
        with torch.inference_mode():
            for batch in batches:
                batch = {key: value.to(device) for key, value in batch.items()}
                if calibration is not None:
                    calibration.validate(model.config, batch)
                n = next(iter(batch.values())).size(0)
                with autocast():
                    output = model(**batch)
                    series_scores = series_classifier.scores(model, batch) if series_classifier is not None else None
                gate_predictions = gate_choice = unguarded_predictions = None
                if series_gate is not None:
                    unguarded_predictions, gate_choice, advantage = series_gate.choose(
                        batch, output.classification_logits, series_scores)
                    gate_predictions = unguarded_predictions
                    if series_gate_guard is not None:
                        guarded_choice = series_gate_guard.select(advantage)
                        gate_prevented += (gate_choice & ~guarded_choice).sum().item()
                        gate_choice = guarded_choice
                        gate_predictions = torch.where(gate_choice, series_scores.argmax(-1),
                                                       output.classification_logits.argmax(-1))
                    gate_examples += gate_choice.numel()
                    gate_selected += gate_choice.sum().item()
                set_mask = None
                if series_set is not None:
                    probs = (series_probability.probabilities(series_scores) if series_probability
                             else log_probabilities(series_scores).exp())
                    set_mask = series_set.predict(probs, series_scores.argmax(-1))
                    sizes = set_mask.sum(-1)
                    set_examples += sizes.numel()
                    set_size += sizes.sum().item()
                    set_singletons += (sizes == 1).sum().item()
                usage = getattr(output, "expert_usage", None)
                routed = getattr(output, "routed_tokens", None)
                if usage is not None and routed is not None:
                    count = routed.item()
                    weighted = usage.double().cpu() * count
                    expert_totals = weighted if expert_totals is None else expert_totals + weighted
                    expert_tokens += count
                weights = {}
                if "labels" in batch:
                    labels = batch["labels"]
                    valid = batch.get("attention_mask", torch.ones_like(labels, dtype=torch.bool)).bool()
                    weights["text"] = ((labels[:, 1:] != -100) & valid[:, 1:] & valid[:, :-1]).sum().item()
                    for distance in range(2, min(model.config.prediction_steps + 1, labels.size(1))):
                        multi_token_count += ((labels[:, distance:] != -100) & valid[:, distance:]
                                              & valid[:, :-distance]).sum().item()
                if "class_labels" in batch:
                    labels = batch["class_labels"]
                    observed = labels != -100
                    weights["classification"] = observed.sum().item()
                    correct += ((output.classification_logits.argmax(-1) == labels) & observed).sum().item()
                    class_count += weights["classification"]
                    if gate_predictions is not None:
                        gate_labeled += observed.sum().item()
                        gate_correct += ((gate_predictions == labels) & observed).sum().item()
                        gate_selected_labeled += (gate_choice & observed).sum().item()
                        gate_selected_correct += ((gate_predictions == labels) & gate_choice & observed).sum().item()
                        if series_gate_guard is not None:
                            gate_unguarded_correct += ((unguarded_predictions == labels) & observed).sum().item()
                            base_correct = output.classification_logits.argmax(-1) == labels
                            head_correct = series_scores.argmax(-1) == labels
                            gate_harms += (gate_choice & base_correct & ~head_correct & observed).sum().item()
                            gate_benefits += (gate_choice & ~base_correct & head_correct & observed).sum().item()
                    if set_mask is not None and observed.any():
                        selected = set_mask[observed]
                        observed_labels = labels[observed]
                        hits = selected.gather(1, observed_labels.unsqueeze(1)).squeeze(1)
                        singletons = selected.sum(-1) == 1
                        set_labeled += observed_labels.numel()
                        set_covered += hits.sum().item()
                        set_labeled_singletons += singletons.sum().item()
                        set_correct_singletons += (hits & singletons).sum().item()
                        set_class_counts += torch.bincount(observed_labels.cpu(), minlength=model.config.num_classes)
                        set_class_covered += torch.bincount(observed_labels[hits].cpu(), minlength=model.config.num_classes)
                    if series_scores is not None:
                        predictions = series_scores.argmax(-1)
                        series_class_correct += ((predictions == labels) & observed).sum().item()
                        series_class_count += observed.sum().item()
                        pairs = labels[observed] * model.config.num_classes + predictions[observed]
                        distinct, frequencies = pairs.unique(return_counts=True)
                        for pair, frequency in zip(distinct.cpu().tolist(), frequencies.cpu().tolist()):
                            series_class_confusion[divmod(pair, model.config.num_classes)] += frequency
                        if series_probability is not None and observed.any():
                            probability_raw.update(log_probabilities(series_scores[observed]), labels[observed])
                            probability_calibrated.update(series_probability.log_probabilities(series_scores[observed]),
                                                          labels[observed])
                if "forecast_targets" in batch:
                    weights["forecast"] = forecasts.update(batch["forecast_targets"], output.forecast, calibration)
                if output.loss is not None:
                    accumulate("loss", output.loss.item(), n)
                for name, value in output.losses.items():
                    if name == "multi_token" and getattr(output, "auxiliary_text_losses", None) is not None:
                        for distance, (head_loss, count) in output.auxiliary_text_losses.items():
                            count = count.item()
                            auxiliary_totals[distance] = auxiliary_totals.get(distance, 0.0) + head_loss.item() * count
                            auxiliary_counts[distance] = auxiliary_counts.get(distance, 0) + count
                        continue
                    accumulate(name, value.item(), weights.get(name, n))
        result = {name: total / counts[name] if counts[name] else None for name, total in totals.items()}
        if expert_tokens:
            result.update(summarize_expert_usage(expert_totals, expert_tokens))
        if auxiliary_counts:
            per_horizon = [auxiliary_totals[distance] / count for distance, count in auxiliary_counts.items() if count]
            result["multi_token"] = sum(per_horizon) / len(per_horizon) if per_horizon else None
        if "text" in counts:
            result["text_tokens"] = counts["text"]
            if model.config.prediction_steps > 1:
                result["multi_token_targets"] = multi_token_count
                result.setdefault("multi_token", None)
            result["byte_perplexity"] = math.exp(min(result["text"], 20)) if counts["text"] else None
        if "classification" in counts:
            result["classified_examples"] = class_count
            result["accuracy"] = correct / class_count if class_count else None
        if series_classifier is not None:
            result["series_classifier_examples"] = series_class_count
            result["series_classifier_accuracy"] = series_class_correct / series_class_count if series_class_count else None
            result["series_classifier_confusion"] = [dict(actual=actual, predicted=predicted, count=count)
                                                      for (actual, predicted), count in sorted(series_class_confusion.items())]
        if series_probability is not None:
            result["series_classifier_probability_metrics"] = dict(
                uncalibrated=probability_raw.result(), calibrated=probability_calibrated.result())
            result["series_classifier_inverse_temperature"] = series_probability.inverse_temperature
        if series_gate is not None:
            result.update(series_gate_examples=gate_examples,
                          series_gate_labeled_examples=gate_labeled,
                          series_gate_accuracy=gate_correct / gate_labeled if gate_labeled else None,
                          series_gate_gain_vs_base=(gate_correct - correct) / gate_labeled if gate_labeled else None,
                          series_gate_head_selection_rate=gate_selected / gate_examples if gate_examples else None,
                          series_gate_head_selected_labeled=gate_selected_labeled,
                          series_gate_head_selected_accuracy=(gate_selected_correct / gate_selected_labeled
                                                              if gate_selected_labeled else None))
        if series_gate_guard is not None:
            result.update(series_gate_guard_max_harm=series_gate_guard.max_harm,
                          series_gate_guard_threshold=series_gate_guard.threshold,
                          series_gate_guard_unguarded_accuracy=(gate_unguarded_correct / gate_labeled
                                                                 if gate_labeled else None),
                          series_gate_guard_prevented_selection_rate=(gate_prevented / gate_examples
                                                                      if gate_examples else None),
                          series_gate_guard_harm_rate=gate_harms / gate_labeled if gate_labeled else None,
                          series_gate_guard_benefit_rate=gate_benefits / gate_labeled if gate_labeled else None)
        if series_set is not None:
            result.update(series_prediction_set_method=series_set.method,
                          series_prediction_set_scope=series_set.scope,
                          series_prediction_set_nominal_coverage=series_set.coverage,
                          series_prediction_set_examples=set_examples,
                          series_prediction_set_labeled_examples=set_labeled,
                          series_prediction_set_coverage=set_covered / set_labeled if set_labeled else None,
                          series_prediction_set_mean_size=set_size / set_examples if set_examples else None,
                          series_prediction_set_singleton_rate=set_singletons / set_examples if set_examples else None,
                          series_prediction_set_singleton_accuracy=(set_correct_singletons / set_labeled_singletons
                                                                    if set_labeled_singletons else None),
                          series_prediction_set_counts_by_class=set_class_counts.tolist(),
                          series_prediction_set_coverage_by_class=[hits.item() / count.item() if count else None
                              for hits, count in zip(set_class_covered, set_class_counts)])
        if iid_intervals:
            intervals = dict(method="wilson", confidence=.95,
                             accuracy=wilson_interval(correct, class_count))
            if series_classifier is not None:
                intervals["series_classifier_accuracy"] = wilson_interval(series_class_correct, series_class_count)
            if series_gate is not None:
                intervals["series_gate_accuracy"] = wilson_interval(gate_correct, gate_labeled)
                intervals["series_gate_head_selected_accuracy"] = wilson_interval(
                    gate_selected_correct, gate_selected_labeled)
            if series_gate_guard is not None:
                intervals["series_gate_guard_harm_rate"] = wilson_interval(gate_harms, gate_labeled)
                intervals["series_gate_guard_benefit_rate"] = wilson_interval(gate_benefits, gate_labeled)
            if series_set is not None:
                intervals["series_prediction_set_coverage"] = wilson_interval(set_covered, set_labeled)
                intervals["series_prediction_set_coverage_by_class"] = [
                    wilson_interval(hits.item(), count.item())
                    for hits, count in zip(set_class_covered, set_class_counts)]
                intervals["series_prediction_set_singleton_accuracy"] = wilson_interval(
                    set_correct_singletons, set_labeled_singletons)
            result["iid_classification_confidence_intervals"] = intervals
        if "loss" in result:
            # Chaque tâche est d'abord agrégée selon son nombre de cibles ; les
            # termes auxiliaires conservent leur agrégation par lot existante.
            supervised = ("text", "classification", "forecast", "multi_token")
            result["observed_loss"] = (
                sum(getattr(model.config, weight) * result[name] for name, weight in LOSS_WEIGHTS.items()
                    if result.get(name) is not None)
                if any(result.get(name) is not None for name in supervised) else None)
        if "forecast" in counts:
            forecasts.report(result, calibration)
        return result
    finally:
        model.train(was_training)
