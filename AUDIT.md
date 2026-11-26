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
| Pas d'entraînement, `small` | 63.7 ms | 52.4 ms | ×1.22 |
| Pas d'entraînement, `small` + SSM/TTT/mémoire | 1060.4 ms | 537.8 ms | ×1.97 |
| Pas d'entraînement, `small` + spectral/ondelettes | 77.5 ms | 53.2 ms | ×1.46 |
| Décodage, `small` | 10.85 ms/token | 8.20 ms/token | ×1.32 |
| TTT-Linear, avant + arrière | 99.9 ms | 47.3 ms | ×2.11 |
| Mémoire épisodique, avant + arrière | 105.4 ms | 71.1 ms | ×1.48 |

| Mesure (CPU) | Avant | Après | Gain |
| --- | ---: | ---: | ---: |
| Pas d'entraînement, `small` + SSM/TTT/mémoire | 1010.4 ms | 730.0 ms | ×1.38 |
| TTT-Linear, avant + arrière | 53.3 ms | 31.3 ms | ×1.70 |
| Décodage, `small` | 9.49 ms/token | 8.92 ms/token | ×1.06 |
| Recherche exacte, 3 000 vecteurs contre eux-mêmes | 23.1 s | 0.69 s | ×33 |
| `predict`, 512 lignes texte + série (`--batch-size 32`) | 8.1 s | 1.5 s | ×5.4 |

Origine des gains :

- **Boucles récurrentes.** Indexer `x[:, t]` à chaque pas rendait la rétropropagation quadratique : chaque `SelectBackward` matérialisait un tenseur complet. Les pas sont maintenant extraits une seule fois avec `unbind`. TTT intègre son biais à l'état empilé (un `bmm` et un `baddbmm` par pas), calcule son gradient interne avec une `layer_norm` fusionnée et normalise ses sorties en une seule fois après la boucle.
- **Synchronisations GPU.** Un seul indicateur « toutes les positions sont valides » est calculé par passe et conservé dans `ModelCache`, au lieu de `key_valid.all()` et `nonzero()` dans chaque couche. Le MoE répartit les tokens avec un tri stable et un seul `bincount`, au lieu d'un `torch.where` synchronisant par expert.
- **Tables RoPE.** Cosinus et sinus sont mis en cache par module. Ils sont reconstruits si `inv_freq` est réassigné (chargement de checkpoint) ou si le device ou le dtype change, et ne sont jamais sérialisés.
- **Mélangeurs temporels.** Les boucles Python par exemple (spectral, ondelettes, historique de prévision) traitent ensemble les lignes de même longueur valide (`blue/ragged.py`), ce qui garde chaque calcul inchangé.
- **RMSNorm.** Le noyau fusionné `F.rms_norm` donne une passe avant identique au bit près.

Sur le preset `small`, `--amp bf16` reste plus lent que FP32 sur GPU (76.9 contre 68.9 ms par pas avec la CLI) : à cette taille, le temps est dominé par les lancements de noyaux et les conversions vers FP32 des routeurs, normalisations et récurrences. Avec un modèle plus large (`d_model=768`, `hidden_dim=2048`, 454 M paramètres, 256 tokens), bf16 est 1.8 fois plus rapide (168 contre 304 ms par pas).

## Vérifications

- `tests/test_audit_regressions.py` et `tests/test_ragged.py` couvrent chaque défaut ci-dessus : copie après entraînement, absence de graphe résiduel (y compris après recalcul des checkpoints), beam en lot, refus des options incompatibles, échelles INT4, identité de l'objectif observé, gradients des poids d'incertitude, répartition Muon, tables RoPE, indicateur dense, cache de recherche et détection d'un fichier modifié.
- 498 tests passent, ainsi que le lint `ruff` désormais exécuté en CI.
- Génération incrémentale avec cache contre calcul complet, sur 45 combinaisons aléatoires de fonctionnalités : écart maximal de 6e-7.
- A/B Muon (3 seeds, 150 pas, données synthétiques) : perte de validation inchangée à la précision du bruit (0.8291 contre 0.8295, 0.8382 contre 0.8374, 0.7700 contre 0.7707). La correction repose sur la définition de Muon, pas sur un gain mesuré sur ces données faciles.

## Compatibilité

Les poids des checkpoints restent compatibles. Seul l'état de l'optimiseur d'un entraînement Muon antérieur ne correspond plus à la nouvelle répartition des groupes : la reprise s'arrête alors avec un message explicite, et `--init-from` repart des mêmes poids avec un optimiseur neuf. `SparseMoE.forward` renvoie un cinquième élément (entropie du routeur) et `DecoderBlock.forward` un `NamedTuple` ; `last_null_fraction` reste lisible comme avant.

## Reproduire

```bash
.venv/bin/python tools/benchmark_hotpaths.py --device cpu --device cuda --output runs/hotpaths.json
.venv/bin/python -m pytest -n 8 tests          # ou : python -m unittest discover -s tests
ruff check blue tools tests
```
