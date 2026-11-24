# Audit du code : corrections et performances

Cet audit a relu les modules du modèle, de l'entraînement, du décodage, de l'optimiseur et de la recherche de vecteurs. Chaque hypothèse de défaut a été confirmée par une reproduction avant correction. Chaque optimisation a été comparée à des sorties de référence de la version d'origine : logits, pertes, gradients de tous les paramètres et générations gloutonnes, stochastiques, spéculatives, Jacobi et en faisceau, sur 14 combinaisons de fonctionnalités. Les écarts restent nuls, ou inférieurs à 2e-5 en relatif lorsque l'ordre des opérations flottantes change.

## Défauts corrigés

| Défaut | Conséquence | Correction |
| --- | --- | --- |
| L'entropie du routeur et les requêtes de la première couche étaient stockées sur les modules (`last_router_entropy`, `last_q_first`) | Le graphe autograd restait vivant après la passe, et `copy.deepcopy(model)` échouait après un pas d'entraînement | Valeurs renvoyées explicitement par `DecoderBlockOutput` ; aucun tenseur du graphe n'est conservé |
| Beam search en lot | Pour B > 1, `beam_width` était ignoré en silence au profit de l'échantillonnage | Recherche par ligne, résultats complétés par `eos_id` |
| Beam plus large que le vocabulaire, ou `eos_id=None` | `IndexError` | Premier pas borné à `vocab_size`, suivi de fin de faisceau corrigé |
| Beam search avec `forecast_adapter_context > 0` et une série | `AttributeError` : champs inexistants dans le réordonnancement du cache | Parcours générique des champs du cache |
| Série 1D de longueur `series_features` dans un JSONL | Diffusée en pas identiques, sans erreur | Formes vérifiées à la lecture, avec fichier et ligne |
| `audit-splits` et signatures de calibration | Ignoraient audio, graphe, nuage 3D, tabulaire et vidéo : faux doublons, jeux tabulaires refusés, calibration acceptée pour une autre combinaison de modalités | Liste unique des modalités d'entrée |
| `generate` en ligne de commande | Échantillonnage non reproductible | Option `--seed` |
| Génération en lot avec padding à gauche et `attention_sinks` ou `dynamic_attn_scale` | Une ligne paddée produisait d'autres logits et tokens que la même ligne seule (positions absolues décalées par le padding) | Positions comptées depuis le premier token valide de chaque ligne ; NSA reste limité aux positions brutes du cache |
| `observed_loss` de validation (`--monitor observed_loss`) | Ignorait les termes ajoutés ensuite (entropie et orthogonalité du routeur, distillation MTP, z-loss, lissage des prévisions) : sélection du meilleur checkpoint sur un autre objectif | Tous les termes pondérés inclus |
| Guidage (`context_guidance_scale`) ou Mirostat combinés à beam, Jacobi ou spéculatif | Option ignorée en silence | `ValueError` explicite |
| `attention_mask=None` passé à `generate` | `TypeError` | Les entrées `None` sont ignorées |
| Échelles INT4 en FP16 | Échelle nulle (0/0) pour des groupes de poids très petits ou nuls | Bornées aux valeurs FP16 représentables |
| `--loss-normalization observed` | `mtp_decay` ignoré, pondération par incertitude absente (les `task_log_vars` ne recevaient aucun gradient) et perte de flow matching supprimée | Même pondération que la passe avant ; identité vérifiée sur un micro-lot |
| Muon appliqué à tout tenseur 2D entraînable | Gains de LayerNorm, biais, embeddings de modalité, fréquences PLR et biais résiduels orthogonalisés | Muon limité aux poids linéaires (et facteurs LoRA), le reste suit AdamW |
| Reprise au milieu d'une époque | Tous les lots déjà vus étaient relus, parsés et assemblés | Les indices sont sautés dans l'échantillonneur de lots, sans lecture |
| `search_embeddings` | Les vecteurs candidats étaient relus et parsés à chaque lot de requêtes | Lots gardés en mémoire dans une limite de 512 Mio |
| `predict` | Une passe avant par exemple | Option `--batch-size` (1 par défaut, sorties inchangées) |

## Performances

Mesures de `tools/benchmark_hotpaths.py`, médianes après échauffement : preset `small`, lot de 8, 96 tokens, 64 tokens générés, 6 threads CPU, RTX 3060 Ti, PyTorch 2.15. Chaque valeur est la meilleure de deux exécutions alternées des deux versions dans la même session. Les données brutes sont dans [audit-hotpaths-before.json](artifacts/audit-hotpaths-before.json) et [audit-hotpaths-after.json](artifacts/audit-hotpaths-after.json).

| Mesure (GPU) | Avant | Après | Gain |
| --- | ---: | ---: | ---: |
| Pas d'entraînement complet par la CLI (`train --preset small`, 60 pas) | 72.6 ms | 59.3 ms | ×1.23 |
| Pas d'entraînement, `small` | 63.0 ms | 54.4 ms | ×1.16 |
| Pas d'entraînement, `small` + SSM/TTT/mémoire | 1051.5 ms | 537.5 ms | ×1.96 |
| Pas d'entraînement, `small` + spectral/ondelettes | 74.1 ms | 54.8 ms | ×1.35 |
| Mémoire de pointe à l'entraînement, `small` | 785 Mio | 738 Mio | −6 % |
| Décodage, `small` | 10.77 ms/token | 7.70 ms/token | ×1.40 |
| Décodage, `small` + SSM/TTT/mémoire | 18.61 ms/token | 14.44 ms/token | ×1.29 |
| TTT-Linear, avant + arrière | 99.6 ms | 47.4 ms | ×2.10 |
| Mémoire épisodique, avant + arrière | 103.6 ms | 69.5 ms | ×1.49 |

| Mesure (CPU) | Avant | Après | Gain |
| --- | ---: | ---: | ---: |
| Pas d'entraînement, `small` + SSM/TTT/mémoire | 977.7 ms | 717.9 ms | ×1.36 |
| TTT-Linear, avant + arrière | 52.9 ms | 31.3 ms | ×1.69 |
| Mémoire épisodique, avant + arrière | 77.4 ms | 59.8 ms | ×1.29 |
| Décodage, `small` | 9.38 ms/token | 8.79 ms/token | ×1.07 |
| Recherche exacte, 3 000 vecteurs contre eux-mêmes | 23.1 s | 0.69 s | ×33 |
| `predict`, 512 lignes texte + série (`--batch-size 32`) | 8.1 s | 1.5 s | ×5.4 |

Origine des gains :

- **Boucles récurrentes.** Indexer `x[:, t]` à chaque pas rendait la rétropropagation quadratique : chaque `SelectBackward` matérialisait un tenseur complet. Les pas sont maintenant extraits une seule fois avec `unbind`. TTT intègre son biais à l'état empilé (un `bmm` et un `baddbmm` par pas), calcule son gradient interne avec une `layer_norm` fusionnée et normalise ses sorties en une seule fois après la boucle.
- **Synchronisations GPU.** Un seul indicateur « toutes les positions sont valides » est calculé par passe et conservé dans `ModelCache`, au lieu de `key_valid.all()` et `nonzero()` dans chaque couche. Le MoE répartit les tokens avec un tri stable et un `scatter_add`, lus en une seule synchronisation, au lieu d'un `torch.where` synchronisant par expert. Les blocs d'encodeur reçoivent la même indication, les composantes de la perte journalisée sont copiées vers l'hôte en un seul transfert et les vérifications de cibles en une seule lecture.
- **Tables RoPE.** Cosinus et sinus sont mis en cache par module. Ils sont reconstruits si `inv_freq` est réassigné (chargement de checkpoint) ou si le device ou le dtype change, et ne sont jamais sérialisés.
- **Mélangeurs temporels.** Les boucles Python par exemple (spectral, ondelettes, historique de prévision) traitent ensemble les lignes de même longueur valide (`blue/ragged.py`), ce qui garde chaque calcul inchangé.
- **RMSNorm.** Le noyau fusionné `F.rms_norm` applique aussi le gain : la passe avant est identique au bit près, sans le retour au dtype réduit suivi d'une nouvelle promotion en FP32 sous autocast (moins d'activations sauvegardées).

Sur le preset `small`, `--amp bf16` reste plus lent que FP32 sur GPU (76.9 contre 68.9 ms par pas avec la CLI) : à cette taille, le temps est dominé par les lancements de noyaux et les conversions vers FP32 des routeurs, normalisations et récurrences. Avec un modèle plus large (`d_model=768`, `hidden_dim=2048`, 454 M paramètres, 256 tokens), bf16 est 1.9 fois plus rapide (161 contre 298 ms par pas).

## Vérifications

- `tests/test_audit_regressions.py` et `tests/test_ragged.py` couvrent chaque défaut ci-dessus : copie après entraînement, absence de graphe résiduel (y compris après recalcul des checkpoints), beam en lot, refus des options incompatibles, échelles INT4, identité de l'objectif observé, gradients des poids d'incertitude, répartition Muon, tables RoPE, indicateur dense, cache de recherche et détection d'un fichier modifié.
- 510 tests passent sous Python 3.14 (CUDA) et 507 sous Python 3.11 (CPU, les 2 tests CUDA étant ignorés), ainsi que le lint `ruff`. Un test d'échantillonnage dépendant de l'état aléatoire global échouait sous 3.11 ; il est corrigé et `blue verify` restaure désormais cet état. Lors de quelques exécutions parallèles deux fois plus lentes que d'habitude (machine chargée par d'autres processus), des tests multi-processus et de benchmark ont échoué ; quatre exécutions complètes consécutives à charge normale passent. Le test de faisceau `test_wave18_features`, dont le modèle dépendait des tests exécutés avant lui, reçoit désormais une graine fixe. La CI exécute désormais les tests en parallèle (`pytest -n auto`) : environ 2 minutes sur 4 cœurs contre plus de 25 minutes en série ; cette durée n'a pas encore été observée sur GitHub Actions.
- Balayage des modes de génération (glouton, échantillonnage, faisceau, spéculatif glouton et stochastique, Jacobi, guidage) sur 23 fonctionnalités, avec et sans contexte : aucun échec. Paramètres d'échantillonnage extrêmes (400 combinaisons) et séries extrêmes (entièrement manquantes, constantes, ±1e30, 1e-30) : sorties et gradients finis.
- Génération incrémentale avec cache contre calcul complet, sur 45 combinaisons aléatoires de fonctionnalités : écart maximal de 6e-7.
- A/B Muon (3 seeds, 150 pas, données synthétiques) : perte de validation inchangée à la précision du bruit (0.8291 contre 0.8295, 0.8382 contre 0.8374, 0.7700 contre 0.7707). La correction repose sur la définition de Muon, pas sur un gain mesuré sur ces données faciles.

## Maintenance

- `Blue._encode_inputs` est l'unique chemin d'encodage des modalités non textuelles, partagé par `forward` et `encode_modalities` (sorties identiques au bit près).
- `Blue._parallel_decode` porte la boucle commune des décodeurs spéculatifs (glouton et stochastique) et de Jacobi ; chacun ne fournit plus que sa proposition et sa règle d'acceptation (−83 lignes, sorties et flux aléatoire identiques).
- `Blue._reorder_cache` parcourt génériquement les champs du cache.
- `modalities.INPUT_MODALITIES` est la liste unique des entrées utilisée par la sélection des modalités, les signatures de calibration, l'audit des jeux et les journaux d'entraînement.
- La restauration des options d'entraînement à la reprise passe par un seul assistant, avec les mêmes contrôles et messages.

## Compatibilité

Les poids des checkpoints restent compatibles. Seul l'état de l'optimiseur d'un entraînement Muon antérieur ne correspond plus à la nouvelle répartition des groupes : la reprise s'arrête alors avec un message explicite, et `--init-from` repart des mêmes poids avec un optimiseur neuf. `SparseMoE.forward` renvoie un cinquième élément (entropie du routeur) et `DecoderBlock.forward` un `NamedTuple` ; `last_null_fraction` reste lisible comme avant.

## Reproduire

```bash
.venv/bin/python tools/benchmark_hotpaths.py --device cpu --device cuda --output runs/hotpaths.json
.venv/bin/python -m pytest -n 8 tests          # ou : python -m unittest discover -s tests
ruff check blue tools tests
```
