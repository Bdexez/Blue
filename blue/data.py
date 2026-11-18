from array import array
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer


def evaluation_data_signature(config):
    """Champs qui définissent les entrées et les cibles des comparaisons synthétiques."""
    names = ("vocab_size", "max_seq_len", "image_size", "image_channels", "series_features",
             "max_series_len", "num_classes", "forecast_horizon", "quantiles")
    return {name: getattr(config, name) for name in names}


def numeric_tensor(values):
    def convert(value):
        if value is None:
            return float("nan")
        if isinstance(value, list):
            return [convert(element) for element in value]
        return value
    return torch.tensor(convert(values), dtype=torch.float32)


def file_identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class JointDataset(Dataset):
    """JSONL local : texte, image, historique et cibles, sans téléchargement implicite."""

    def __init__(self, path, config, *, allow_mixed=False, include_alignment_ids=False):
        self.path = Path(path).resolve()
        self.include_alignment_ids = include_alignment_ids
        self.config = config
        initial_identity = file_identity(self.path.stat())
        self.offsets, self.line_numbers = array("Q"), array("Q")
        self.signatures = array("B") if allow_mixed else None
        signature = None
        digest = hashlib.sha256()
        image_lines = {}
        with self.path.open("rb") as stream:
            if file_identity(os.fstat(stream.fileno())) != initial_identity:
                raise ValueError(f"Le jeu de données a changé pendant son indexation : {self.path}")
            line_number = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                line_number += 1
                digest.update(line)
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError(f"{self.path}:{line_number}: JSON invalide") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{self.path}:{line_number}: un objet JSON est attendu")
                current = tuple(key in record or (key == "image" and "image_path" in record)
                                for key in ("text", "image", "series", "audio", "graph_nodes", "points"))
                if not any(current):
                    raise ValueError(f"{self.path}:{line_number}: fournir text, image/image_path, series, audio, graph_nodes ou points")
                if not allow_mixed and signature is not None and current != signature:
                    raise ValueError(f"{self.path}:{line_number}: les modalités doivent être identiques dans le fichier")
                signature = current
                if self.signatures is not None:
                    self.signatures.append(sum(int(present) << index for index, present in enumerate(current)))
                if "image_path" in record and "image" not in record:
                    image_path = record["image_path"]
                    if not isinstance(image_path, str) or not image_path:
                        raise ValueError(f"{self.path}:{line_number}: image_path doit être une chaîne non vide")
                    image_lines.setdefault(image_path, line_number)
                self.offsets.append(offset)
                self.line_numbers.append(line_number)
            if (file_identity(os.fstat(stream.fileno())) != initial_identity
                    or file_identity(self.path.stat()) != initial_identity):
                raise ValueError(f"Le jeu de données a changé pendant son indexation : {self.path}")
        if not self.offsets:
            raise ValueError("Le fichier JSONL ne contient aucun exemple")
        self.fingerprint = digest.hexdigest()
        self._image_identities = {}
        image_digests = {}
        for name, line_number in image_lines.items():
            image_path = self.path.parent / name
            try:
                image_digest = hashlib.sha256()
                with image_path.open("rb") as image_stream:
                    before = file_identity(os.fstat(image_stream.fileno()))
                    for chunk in iter(lambda: image_stream.read(1024 * 1024), b""):
                        image_digest.update(chunk)
                    after = file_identity(os.fstat(image_stream.fileno()))
                if before != after or after != file_identity(image_path.stat()):
                    raise ValueError(f"L'image a changé pendant son indexation : {image_path}")
            except (OSError, ValueError) as exc:
                raise ValueError(f"{self.path}:{line_number}: {exc}") from exc
            self._image_identities[name] = after
            image_digests[name] = image_digest.hexdigest()
        manifest = json.dumps(image_digests, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        self.image_fingerprint = hashlib.sha256(manifest.encode("ascii")).hexdigest() if image_digests else None
        self._identity = file_identity(self.path.stat())
        if self._identity != initial_identity:
            raise ValueError(f"Le jeu de données a changé pendant son indexation : {self.path}")

    def __len__(self):
        return len(self.offsets)

    def references_image(self, path):
        """Reconnaître aussi les liens physiques vers une image du jeu."""
        if path is None:
            return False
        try:
            identity = file_identity(Path(path).stat())[:2]
        except FileNotFoundError:
            return False
        return any(identity == known[:2] for known in self._image_identities.values())

    def read_record(self, index):
        """Lire l'objet JSON indexé sans décoder ses images ni convertir ses tableaux."""
        with self.path.open("rb") as stream:
            if file_identity(os.fstat(stream.fileno())) != self._identity:
                raise ValueError(f"Le jeu de données a changé après son indexation : {self.path}")
            stream.seek(self.offsets[index])
            line = stream.readline()
            if (file_identity(os.fstat(stream.fileno())) != self._identity
                    or file_identity(self.path.stat()) != self._identity):
                raise ValueError(f"Le jeu de données a changé pendant sa lecture : {self.path}")
        return json.loads(line)

    def __getitem__(self, index):
        record = self.read_record(index)
        try:
            item = self._convert(record)
            if self.include_alignment_ids:
                source_index = index if index >= 0 else len(self) + index
                item['alignment_ids'] = torch.tensor(source_index, dtype=torch.long)
            return item
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            raise ValueError(f"{self.path}:{self.line_numbers[index]}: {exc}") from exc

    def _convert(self, record):
        item = {}
        if "text" in record:
            if not isinstance(record["text"], str):
                raise ValueError("text doit être une chaîne UTF-8")
            ids = ByteTokenizer().encode(record["text"])
            if len(ids) > self.config.max_seq_len:
                ids = ids[:self.config.max_seq_len]
                ids[-1] = ByteTokenizer.eos_id
            item["input_ids"] = torch.tensor(ids, dtype=torch.long)
        if "image" in record:
            item["images"] = torch.tensor(record["image"], dtype=torch.float32)
        elif "image_path" in record:
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError("Pour lire PNG/JPEG : pip install '.[vision]'") from exc
            path = self.path.parent / record["image_path"]
            if self.config.image_channels not in (1, 3):
                raise ValueError("image_path prend en charge image_channels=1 ou 3 ; utiliser image sinon")
            if file_identity(path.stat()) != self._image_identities[record["image_path"]]:
                raise ValueError(f"L'image a changé après son indexation : {path}")
            with Image.open(path) as source:
                mode = "RGB" if self.config.image_channels == 3 else "L"
                image = source.convert(mode).resize((self.config.image_size,) * 2)
                pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            if file_identity(path.stat()) != self._image_identities[record["image_path"]]:
                raise ValueError(f"L'image a changé pendant sa lecture : {path}")
            item["images"] = pixels.view(self.config.image_size, self.config.image_size, -1).permute(2, 0, 1).float() / 255
        if "series" in record:
            item["series"] = numeric_tensor(record["series"])
        if "audio" in record:
            item["audio"] = numeric_tensor(record["audio"])
        if "graph_nodes" in record:
            item["graph_nodes"] = numeric_tensor(record["graph_nodes"])
            if "graph_edges" in record:
                item["graph_edges"] = numeric_tensor(record["graph_edges"])
        if "points" in record:
            item["points"] = numeric_tensor(record["points"])
            if "point_features" in record:
                item["point_features"] = numeric_tensor(record["point_features"])
        if "class_label" in record:
            label = record["class_label"]
            if type(label) is not int or not (label == -100 or 0 <= label < self.config.num_classes):
                raise ValueError("class_label doit être une classe entière valide ou -100")
            item["class_labels"] = torch.tensor(record["class_label"], dtype=torch.long)
        if "future" in record:
            item["forecast_targets"] = numeric_tensor(record["future"])
        return item


def collate_records(items):
    if not items:
        raise ValueError("Le batch ne peut pas être vide")
    result = {}
    for name in ("input_ids", "images", "series", "audio", "graph_nodes", "points"):
        present = [name in item for item in items]
        if any(present) and not all(present):
            raise ValueError(f"{name} doit être fourni pour tout le batch")
    identities = ["alignment_ids" in item for item in items]
    if any(identities) and not all(identities):
        raise ValueError("alignment_ids doit être fourni pour tout le batch")
    if all(identities):
        result["alignment_ids"] = torch.stack([item["alignment_ids"] for item in items])
    if "images" in items[0]:
        result["images"] = torch.stack([item["images"] for item in items])
    for name, missing in (("class_labels", -100), ("forecast_targets", float("nan"))):
        example = next((item[name] for item in items if name in item), None)
        if example is not None:
            result[name] = torch.stack([item[name] if name in item else torch.full_like(example, missing)
                                        for item in items])
    if "input_ids" in items[0]:
        length = max(item["input_ids"].size(0) for item in items)
        ids = torch.full((len(items), length), ByteTokenizer.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, item in enumerate(items):
            n = item["input_ids"].size(0)
            ids[i, :n] = item["input_ids"]
            mask[i, :n] = True
        result.update(input_ids=ids, labels=ids.clone(), attention_mask=mask)
    if "series" in items[0]:
        length = max(item["series"].size(0) for item in items)
        features = items[0]["series"].size(-1)
        values = torch.full((len(items), length, features), float("nan"))
        mask = torch.zeros(len(items), length, dtype=torch.bool)
        for i, item in enumerate(items):
            n = item["series"].size(0)
            values[i, :n] = item["series"]
            mask[i, :n] = True
        result.update(series=values, series_mask=mask)
    if "audio" in items[0]:
        length = max(item["audio"].size(-1) for item in items)
        channels = items[0]["audio"].size(0)
        values = torch.zeros(len(items), channels, length, dtype=torch.float32)
        mask = torch.zeros(len(items), length, dtype=torch.bool)
        for i, item in enumerate(items):
            n = item["audio"].size(-1)
            values[i, :, :n] = item["audio"]
            mask[i, :n] = True
        result.update(audio=values, audio_mask=mask)
    if "graph_nodes" in items[0]:
        max_nodes = max(item["graph_nodes"].size(0) for item in items)
        node_feat = items[0]["graph_nodes"].size(-1)
        nodes = torch.zeros(len(items), max_nodes, node_feat, dtype=torch.float32)
        g_mask = torch.zeros(len(items), max_nodes, dtype=torch.bool)
        for i, item in enumerate(items):
            n = item["graph_nodes"].size(0)
            nodes[i, :n] = item["graph_nodes"]
            g_mask[i, :n] = True
        result.update(graph_nodes=nodes, graph_mask=g_mask)
        edge_present = ["graph_edges" in item for item in items]
        if any(edge_present):
            if not all(edge_present):
                raise ValueError("graph_edges doit être fourni pour tout le batch")
            edge_feat = items[0]["graph_edges"].size(-1)
            edges = torch.zeros(len(items), max_nodes, max_nodes, edge_feat, dtype=torch.float32)
            for i, item in enumerate(items):
                n = item["graph_nodes"].size(0)
                edges[i, :n, :n] = item["graph_edges"]
            result["graph_edges"] = edges
    if "points" in items[0]:
        max_pts = max(item["points"].size(0) for item in items)
        pts = torch.zeros(len(items), max_pts, 3, dtype=torch.float32)
        p_mask = torch.zeros(len(items), max_pts, dtype=torch.bool)
        for i, item in enumerate(items):
            n = item["points"].size(0)
            pts[i, :n] = item["points"]
            p_mask[i, :n] = True
        result.update(points=pts, point_mask=p_mask)
        feat_present = ["point_features" in item for item in items]
        if any(feat_present):
            if not all(feat_present):
                raise ValueError("point_features doit être fourni pour tout le batch")
            f_dim = items[0]["point_features"].size(-1)
            feats = torch.zeros(len(items), max_pts, f_dim, dtype=torch.float32)
            for i, item in enumerate(items):
                n = item["points"].size(0)
                feats[i, :n] = item["point_features"]
            result["point_features"] = feats
    return result


def synthetic_batch(config, *, batch_size=4, seed=0, device="cpu"):
    """Les modalités décrivent le même régime latent ; futur jamais donné en entrée."""
    gen = torch.Generator().manual_seed(seed)
    classes = torch.randint(config.num_classes, (batch_size,), generator=gen)
    length = min(config.max_series_len, 32)
    times = torch.arange(length + config.forecast_horizon).float()[None, :, None]
    phase = torch.rand(batch_size, 1, config.series_features, generator=gen) * 3
    frequency = 0.08 + classes[:, None, None].float() * 0.025
    slope = (classes[:, None, None].float() - config.num_classes / 2) * 0.015
    sequence = torch.sin(times * frequency + phase) + times * slope
    sequence = sequence + torch.randn(sequence.shape, generator=gen) * 0.015
    axis = torch.linspace(-1, 1, config.image_size)
    pattern = torch.sin((classes[:, None, None].float() + 1) * axis[None, None, :] * 3)
    image = pattern.expand(-1, config.image_size, -1)[:, None].expand(-1, config.image_channels, -1, -1)
    image = (image + 1) / 2 + torch.randn(image.shape, generator=gen) * 0.01
    audio = None
    if getattr(config, "audio_enabled", False):
        audio_len = min(config.max_audio_len, max(config.audio_patch_size * 4, 64))
        a_times = torch.linspace(0.0, 1.0, audio_len)[None, None, :]
        base_hz = 2.0 + classes[:, None, None].float() * 1.5
        harmonics = (torch.sin(2.0 * torch.pi * base_hz * a_times)
                     + 0.4 * torch.sin(4.0 * torch.pi * base_hz * a_times))
        audio = harmonics.expand(-1, config.audio_channels, -1)
        audio = audio + torch.randn(audio.shape, generator=gen) * 0.02
    graph_nodes = graph_edges = None
    if getattr(config, "graph_enabled", False):
        n_nodes = min(config.max_graph_nodes, 8)
        graph_nodes = torch.randn(batch_size, n_nodes, config.graph_node_features, generator=gen) * 0.2
        graph_nodes = graph_nodes + classes[:, None, None].float() * 0.25
        graph_edges = torch.zeros(batch_size, n_nodes, n_nodes, config.graph_edge_features)
        for b in range(batch_size):
            stride = (classes[b].item() % max(n_nodes - 1, 1)) + 1
            for node in range(n_nodes):
                neighbor = (node + stride) % n_nodes
                if neighbor != node:
                    val = 1.0 + 0.1 * classes[b].item()
                    graph_edges[b, node, neighbor, 0] = val
                    graph_edges[b, neighbor, node, 0] = val
    points = point_features = None
    if getattr(config, "pointcloud_enabled", False):
        n_pts = min(config.max_pointcloud_points, 12)
        angles = torch.linspace(0.0, 2.0 * torch.pi, n_pts + 1)[:-1]
        radii = 0.5 + 0.2 * (classes.float() + 1.0)
        xs = radii[:, None] * torch.cos(angles[None, :])
        ys = radii[:, None] * torch.sin(angles[None, :])
        zs = 0.3 * torch.sin((classes[:, None].float() + 1.0) * angles[None, :])
        points = torch.stack((xs, ys, zs), dim=-1) + torch.randn(batch_size, n_pts, 3, generator=gen) * 0.02
        if config.pointcloud_features > 0:
            point_features = torch.randn(batch_size, n_pts, config.pointcloud_features, generator=gen) * 0.1
            point_features = point_features + classes[:, None, None].float() * 0.2
    tokenizer = ByteTokenizer()
    items = []
    for i in range(batch_size):
        text = f"Signal de classe {classes[i].item()}."
        tokens = tokenizer.encode(text)[:config.max_seq_len]
        entry = dict(input_ids=torch.tensor(tokens), images=image[i], series=sequence[i, :length],
                     class_labels=classes[i], forecast_targets=sequence[i, length:])
        if audio is not None:
            entry["audio"] = audio[i]
        if graph_nodes is not None:
            entry["graph_nodes"] = graph_nodes[i]
            entry["graph_edges"] = graph_edges[i]
        if points is not None:
            entry["points"] = points[i]
            if point_features is not None:
                entry["point_features"] = point_features[i]
        items.append(entry)
    return {key: value.to(device) for key, value in collate_records(items).items()}
