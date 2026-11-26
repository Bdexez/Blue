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
| `BlueConfig` modifiée après création (`config.x = ...`) | Jamais revalidée : `n_heads=5`, `top_k=99`, quantiles sans médiane ou booléen remplacé par 1 étaient acceptés puis échouaient plus loin | `Blue` revalide la configuration à la construction |
| `blue export --output` identique au `--checkpoint` | Accepté sans transformation : le checkpoint d'entraînement était remplacé par un export d'inférence (optimiseur, états aléatoires et poids d'origine perdus avec `--use-ema`) | Refusé dans tous les cas |
| Erreurs d'utilisation en ligne de commande | Trace Python complète pour une option ou un fichier invalide | Message `blue : erreur : …` et code 2 (`BLUE_TRACEBACK=1` garde la trace) |
| Checkpoints | Renommage atomique sans `fsync` : après une coupure de courant, `latest.pt` pouvait être vide ou tronqué | Fichier puis répertoire synchronisés avant publication |
| Génération en lot avec padding à gauche et `attention_sinks` ou `dynamic_attn_scale` | Une ligne paddée produisait d'autres logits et tokens que la même ligne seule (positions absolues décalées par le padding) | Positions comptées depuis le premier token valide de chaque ligne, blocs NSA réalignés de même |
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
| Pas d'entraînement `small` par défaut (`--threads` non précisé, Ryzen 6 cœurs) | 436 ms (4 threads imposés) | 303 ms (choix de PyTorch : 6) | ×1.44 |
| Pas d'entraînement GPU `small` sur JSONL réel (images en ligne), `--workers` non précisé | 79.3 ms (0 processus) | 59.8 ms (2 processus par défaut sur GPU) | ×1.33 |
| Même entraînement sur 3 époques : processus de lecture persistants | 71.9 ms en moyenne (pics de 836 ms à chaque époque) | 61.9 ms (pics de 117 ms) | ×1.16 |
| 60 pas GPU avec 6 validations sur un JSONL séparé : chargeur de validation persistant | 12.4 s | 9.5 s | ×1.31 |
| `train --preset small --device cuda --steps 100` (défauts : validation et sauvegarde tous les 25 pas), durée totale | 13.3 s | 11.2 s | ×1.19 |

Origine des gains :

- **Boucles récurrentes.** Indexer `x[:, t]` à chaque pas rendait la rétropropagation quadratique : chaque `SelectBackward` matérialisait un tenseur complet. Les pas sont maintenant extraits une seule fois avec `unbind`. TTT intègre son biais à l'état empilé (un `bmm` et un `baddbmm` par pas), calcule son gradient interne avec une `layer_norm` fusionnée et normalise ses sorties en une seule fois après la boucle.
- **Synchronisations GPU.** Un seul indicateur « toutes les positions sont valides » est calculé par passe et conservé dans `ModelCache`, au lieu de `key_valid.all()` et `nonzero()` dans chaque couche. Le MoE répartit les tokens avec un tri stable et un `scatter_add`, lus en une seule synchronisation, au lieu d'un `torch.where` synchronisant par expert. Les blocs d'encodeur reçoivent la même indication, les composantes de la perte journalisée sont copiées vers l'hôte en un seul transfert et les vérifications de cibles en une seule lecture.
- **Tables RoPE.** Cosinus et sinus sont mis en cache par module. Ils sont reconstruits si `inv_freq` est réassigné (chargement de checkpoint) ou si le device ou le dtype change, et ne sont jamais sérialisés.
- **Mélangeurs temporels.** Les boucles Python par exemple (spectral, ondelettes, historique de prévision) traitent ensemble les lignes de même longueur valide (`blue/ragged.py`), ce qui garde chaque calcul inchangé.
- **Convolutions d'images et de vidéos.** Format `channels_last` pour l'encodeur de vision (avant + arrière : 52 → 39 ms sur CPU, 5.7 → 3.9 ms sur GPU ; pas d'entraînement `small` sur CPU : environ 322 → 307 ms) et `channels_last_3d` pour la vidéo sur CPU (50.9 → 40.1 ms). Sorties modifiées d'au plus 1.2e-6 en relatif.
- **RMSNorm.** Le noyau fusionné `F.rms_norm` applique aussi le gain : la passe avant est identique au bit près, sans le retour au dtype réduit suivi d'une nouvelle promotion en FP32 sous autocast (moins d'activations sauvegardées).

Sur le preset `small`, `--amp bf16` reste plus lent que FP32 sur GPU (76.9 contre 68.9 ms par pas avec la CLI) : à cette taille, le temps est dominé par les lancements de noyaux et les conversions vers FP32 des routeurs, normalisations et récurrences. Avec un modèle plus large (`d_model=768`, `hidden_dim=2048`, 454 M paramètres, 256 tokens), bf16 est 1.9 fois plus rapide (161 contre 298 ms par pas).

Le preset `hyper` (1.85 milliard de paramètres, bf16, RTX 3060 Ti) fonctionne en inférence (3.8 Gio de pointe) mais décode à environ 320 ms par token : 16 couches × 2 itérations latentes, et la projection de Sinkhorn des hyper-connexions (20 itérations sur des matrices 4×4, trois fois par couche) occupe à elle seule 30 % du temps, en lancements de petits noyaux. Seule une compilation (`torch.compile`) de cette fonction les fusionnerait ; elle n'a pas été ajoutée, faute de gain bout à bout démontré sur l'entraînement (voir ci-dessous) et pour ne pas imposer de compilation à l'exécution.

L'option globale `--matmul-precision high` active TF32 pour les produits matriciels FP32 (désactivé par défaut, résultats inchangés). Sur un pas d'entraînement isolé (lot de 8, 256 tokens), elle accélère le preset `small` de ×1.09 (77.2 contre 71.1 ms) et un modèle de 454 M paramètres de ×1.27 (294.8 contre 232.8 ms) ; la boucle CLI du preset `small` sur les textes courts synthétiques, limitée par les lancements de noyaux, n'en bénéficie pas (62.2 contre 62.1 ms).

Les experts routés du MoE représentent environ 39 % d'un pas d'entraînement GPU du preset `small` (50.9 ms, contre 31.2 ms si l'on ne garde que l'expert partagé). Un calcul groupé de tous les experts (deux `bmm` sur des groupes complétés jusqu'à l'expert le plus chargé) s'est révélé plus lent (55.5 ms sur GPU, 536 ms contre 320 ms sur CPU) : l'empilement des poids et le remplissage coûtent davantage que les lancements économisés. Calculer au contraire chaque expert sur tous les jetons, sans aiguillage ni synchronisation, est aussi plus lent (63.5 contre 50.6 ms sur GPU, 558 contre 318 ms sur CPU). L'aiguillage actuel reste le meilleur compromis mesuré ; aucun de ces deux essais n'a été conservé.

Inférence quantifiée sur CPU (preset `small`, 6 threads, décodage glouton) : FP32 9.07 ms par token pour 212 Mio, INT8 7.71 ms pour 63 Mio, INT4 13.09 ms pour 41 Mio. L'INT8 est à la fois plus compact et plus rapide (moins de mémoire lue) ; l'INT4 gagne encore en taille mais paie le dépaquetage de ses demi-octets à chaque appel (une variante vectorisée du dépaquetage, exacte, s'est montrée plus lente et n'a pas été conservée).

Une sauvegarde de checkpoint `small` (542 Mo avec l'optimiseur) coûte environ 450 ms, dont la moitié pour `fsync`. Une écriture en arrière-plan (instantané CPU puis sérialisation dans un thread) a été essayée : la boucle d'entraînement, limitée par Python, ralentissait davantage (100 pas : 12.3 contre 11.2 s) à cause de la copie de l'instantané et de la concurrence pour le GIL. Elle n'a pas été conservée ; la sauvegarde double au même pas est en revanche évitée (lien physique entre `best.pt` et `latest.pt`).

Un essai de `torch.compile` sur les blocs du décodeur (preset `small`) accélérait un pas isolé de 51.5 à 41.5 ms, mais n'apportait rien dans la boucle d'entraînement réelle (61.7 contre 59.5 ms) : les formes changent d'un lot à l'autre (abandon de modalités, validation) et les recompilations annulent le gain. L'option n'a donc pas été conservée.

## Vérifications

- `tests/test_audit_regressions.py` et `tests/test_ragged.py` couvrent chaque défaut ci-dessus : copie après entraînement, absence de graphe résiduel (y compris après recalcul des checkpoints), beam en lot, refus des options incompatibles, échelles INT4, identité de l'objectif observé, gradients des poids d'incertitude, répartition Muon, tables RoPE, indicateur dense, cache de recherche et détection d'un fichier modifié.
- 517 tests passent sous Python 3.14 (CUDA) et 515 sous Python 3.11 (CPU, les 2 tests CUDA étant ignorés), ainsi que le lint `ruff`. Un test d'échantillonnage dépendant de l'état aléatoire global échouait sous 3.11 ; il est corrigé et `blue verify` restaure désormais cet état. Des tests multi-processus et de benchmark échouaient lors d'exécutions parallèles deux fois plus lentes que d'habitude : la cause était la pression mémoire (swap) créée par 8 Go de fichiers d'essai de l'audit dans `/tmp`, monté en RAM. Ces fichiers supprimés, cinq exécutions complètes consécutives passent. Le test de faisceau `test_wave18_features`, dont le modèle dépendait des tests exécutés avant lui, reçoit désormais une graine fixe. La CI exécute désormais les tests en parallèle (`pytest -n auto`) : environ 2 minutes sur 4 cœurs contre plus de 25 minutes en série ; cette durée n'a pas encore été observée sur GitHub Actions.
- Balayage des modes de génération (glouton, échantillonnage, faisceau, spéculatif glouton et stochastique, Jacobi, guidage) sur 23 fonctionnalités, avec et sans contexte : aucun échec. Paramètres d'échantillonnage extrêmes (400 combinaisons) et séries extrêmes (entièrement manquantes, constantes, ±1e30, 1e-30) : sorties et gradients finis.
- Les quatre outils de `tools/` sans test (`make_dataset`, `profile_runtime`, `compare_modality_training`, `compare_top1_routing`) et `benchmark_hotpaths` s'exécutent désormais dans la suite ; l'empreinte du cache mesurée par `profile_runtime` reste celle publiée dans `IMPROVEMENTS.md` (56 832 octets, positions 31 / 160 / 31).
- Couverture des tests (`pytest --cov=blue`) : 93 % des lignes du paquet ; l'alignement distribué, exécuté dans des sous-processus DDP, n'est pas compté par cette mesure.
- Aller-retour sauvegarde/chargement de checkpoint sur 12 combinaisons de fonctionnalités : sorties et générations identiques au bit près. Invariance à la composition du lot et au padding à droite sur l'ensemble des fonctionnalités : écart ≤ 3e-7. Gradient checkpointing : pertes et gradients identiques au bit près sur 10 configurations.
- Génération incrémentale avec cache contre calcul complet, sur 45 combinaisons aléatoires de fonctionnalités : écart maximal de 6e-7.
- Entraînement GPU réel (`train --preset small --device cuda`, 60 pas) en FP32, bf16 et FP16 : perte 8.15 → 0.937 / 0.934 / 0.942, perte de validation 0.949 / 0.947 / 0.955. En FP16, `GradScaler` saute deux pas au démarrage (débordements détectés puis échelle réduite), puis l'apprentissage se poursuit : le chemin FP16 sur GPU, non exécuté lors des rapports précédents, est ainsi vérifié.
- Commandes d'ajustement et d'inférence sur GPU : `calibrate`, `evaluate`, `refine-forecast`, `refine-series-classifier`, `calibrate-series-probability`, `calibrate-series-set`, `predict`, `embed` et `verify` s'exécutent avec `--device cuda` ; tête temporelle et ensembles conformes identiques à ceux calculés sur CPU (36/36 exemples). `verify --device cuda` échouait jusqu'ici sur le contrôle du gradient checkpointing (tolérance absolue trop stricte pour les additions atomiques GPU) ; il passe désormais.
- A/B Muon (3 seeds, 150 pas, données synthétiques) : perte de validation inchangée à la précision du bruit (0.8291 contre 0.8295, 0.8382 contre 0.8374, 0.7700 contre 0.7707). La correction repose sur la définition de Muon, pas sur un gain mesuré sur ces données faciles.

## Maintenance

- Option globale `--deterministic` : deux entraînements GPU identiques divergeaient (392 tenseurs, jusqu'à 6e-6, à cause des additions atomiques) ; avec l'option, deux exécutions et une reprise sur GPU sont identiques au bit près (preset `small` : environ 63 ms par pas au lieu de 60).
- `train --resume auto` reprend depuis `OUTPUT/latest.pt` s'il existe et démarre sinon un nouvel entraînement : une tâche interrompue se relance avec la même commande (reprise identique au bit près, testée).
- `metrics.jsonl` indique la durée de chaque pas (`step_seconds`) ; l'EMA des poids se met à jour en une opération multi-tenseurs (2.3 → 1.6 ms sur GPU pour `small`, résultat identique).

- `Blue._encode_inputs` est l'unique chemin d'encodage des modalités non textuelles, partagé par `forward` et `encode_modalities` (sorties identiques au bit près).
- `Blue._parallel_decode` porte la boucle commune des décodeurs spéculatifs (glouton et stochastique) et de Jacobi ; chacun ne fournit plus que sa proposition et sa règle d'acceptation (−83 lignes, sorties et flux aléatoire identiques).
- `Blue._reorder_cache` parcourt génériquement les champs du cache.
- `modalities.INPUT_MODALITIES` est la liste unique des entrées utilisée par la sélection des modalités, les signatures de calibration, l'audit des jeux et les journaux d'entraînement.
- La restauration des options d'entraînement à la reprise passe par un seul assistant, avec les mêmes contrôles et messages.

## Compatibilité

Les poids des checkpoints restent compatibles. Seul l'état de l'optimiseur d'un entraînement Muon antérieur ne correspond plus à la nouvelle répartition des groupes : la reprise s'arrête alors avec un message explicite, et `--init-from` repart des mêmes poids avec un optimiseur neuf. `SparseMoE.forward` renvoie un cinquième élément (entropie du routeur) et `DecoderBlock.forward` un `NamedTuple` ; `last_null_fraction` reste lisible comme avant.

## Pistes restantes

Mesures faites sur le preset `small` (RTX 3060 Ti, Ryzen 6 cœurs) ; chaque piste est classée par gain attendu et risque.

- **Décodage GPU par CUDA graphs.** Le décodage (7.7 ms par token) est limité par le lancement de centaines de petits noyaux ; seule une capture en CUDA graph le réduirait nettement. Elle impose un cache KV de taille fixe, des positions et un indicateur de validité sous forme de tenseurs, et la suppression des synchronisations de l'aiguillage MoE : refonte importante du cache, non entreprise.
- **Projection de Sinkhorn des hyper-connexions** (preset `hyper`, 30 % du décodage) : une version fusionnée exige `torch.compile` ; la variante exponentielle, plus rapide, change les résultats pour des logits de grande amplitude.
- **MoE** (39 % d'un pas d'entraînement GPU) : calcul groupé et calcul dense tous deux plus lents que l'aiguillage actuel ; un noyau d'experts groupés dédié (grouped GEMM) serait la prochaine étape.
- **Déquantification INT4 à chaque appel** : seule une multiplication matricielle sur poids 4 bits (noyau dédié) rendrait l'INT4 plus rapide que le FP32 sur CPU.
- Gains mesurés inférieurs à 3 %, non retenus : têtes de classification et de prévision pendant le décodage (2 %), génération des lots synthétiques (au plus 3 %), mémoire épinglée pour les transferts (0 %).

## Reproduire

```bash
.venv/bin/python tools/golden_outputs.py record --output runs/golden-before.pt   # avant une modification
.venv/bin/python tools/golden_outputs.py record --output runs/golden-after.pt    # après
.venv/bin/python tools/golden_outputs.py compare runs/golden-before.pt runs/golden-after.pt --tolerance 1e-5
.venv/bin/python tools/benchmark_hotpaths.py --device cpu --device cuda --output runs/hotpaths.json
.venv/bin/python -m pytest -n 8 tests          # ou : python -m unittest discover -s tests
ruff check blue tools tests
```
