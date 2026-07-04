# Entraînement Distribué de BERT — Data Parallelism Multi-GPU avec PyTorch DDP

Fine-tuning de BERT (`bert-base-uncased`) pour la classification de sentiments sur le dataset **IMDb**, avec comparaison mesurée entre un entraînement sur **1 GPU** et un entraînement distribué sur **2 GPU** via **PyTorch Lightning + DistributedDataParallel (DDP)**.

Ce projet a pour but de démontrer, avec des mesures réelles et non des hypothèses, comment le parallélisme de données accélère l'entraînement d'un modèle Deep Learning, quelles sont ses limites mémoire, et jusqu'où on peut réellement espérer un speedup.

---

## Objectifs

- Distinguer l'entraînement sur un seul GPU et l'entraînement distribué multi-GPU.
- Utiliser PyTorch Lightning et `DistributedDataParallel` (DDP) avec les paramètres `devices`, `strategy` et `precision`.
- Mesurer et analyser un speedup réel sur une charge de travail suffisamment grande pour être significative.
- Comprendre le rôle de la synchronisation des gradients (AllReduce / NCCL) avant chaque mise à jour des poids.
- Appliquer la loi d'Amdahl pour expliquer les performances mesurées d'un entraînement multi-GPU.

---

## Environnement requis

- Au moins **2 GPU** (idéalement 2× NVIDIA T4, 16 Go de VRAM chacun).
- **Kaggle Notebook** (recommandé) : Settings → Accelerator → GPU T4 x2, avec Internet activé (téléchargement du dataset IMDb et des poids BERT).
- **Google Colab** : la version gratuite ne fournit qu'un seul GPU par session ; l'expérience multi-GPU nécessite Colab Pro+ ou un environnement équivalent.

### Dépendances

```bash
pip install "transformers>=4.41.0" "datasets>=2.20.0" "pytorch-lightning==2.2.4" "torchmetrics==1.4.0"
```

---

## Structure du projet

| Fichier | Rôle |
|---|---|
| `single-gpu-experiment.ipynb` | Expérience A — baseline sur 1 GPU, + démonstration du mur mémoire (OOM) |
| `multi-gpu-experiment.ipynb` | Expérience B — entraînement distribué DDP sur 2 GPU, même dataset et même timing par epoch que l'expérience A |
| `train_ddp.py` | Script généré automatiquement par le notebook multi-GPU, lancé via `torchrun` |
| `results_single_gpu.json` | Résultats de l'expérience A (produits par le notebook single-GPU) |
| `results_multi_gpu.json` | Résultats de l'expérience B (produits par `train_ddp.py`) |

**Remarque importante :** `DistributedDataParallel` lance un **processus système indépendant par GPU**. Il est donc impossible de lancer DDP directement dans une cellule d'un notebook Jupyter déjà en cours d'exécution — le contexte CUDA y est déjà initialisé, ce qui provoque des erreurs de fork/spawn. La bonne pratique, utilisée ici, consiste à écrire l'entraînement dans un script (`train_ddp.py`) puis à le lancer avec `torchrun`, comme on le ferait en production.

---

## Données et modèle

- **Dataset** : IMDb (25 000 critiques d'entraînement, 25 000 de test), classification binaire (positif / négatif).
- **Découpage** : `TRAIN_SIZE = 20 000`, `VAL_SIZE = 5 000`, avec un seed fixe (`torch.Generator().manual_seed(42)`) pour garantir un split identique et reproductible entre les deux expériences.
- **Tokenisation** : `BertTokenizer`, `max_length=128`.
- **Modèle** : `BertForSequenceClassification` (`bert-base-uncased`, ~110M paramètres), entraîné via un `LightningModule` (`BERTSentimentClassifier`) réutilisé à l'identique dans les deux expériences — seule la stratégie de parallélisation change.

---

## Expérience A — Baseline sur 1 GPU

```python
trainer_single = pl.Trainer(
    accelerator="gpu", devices=1, strategy="auto",
    enable_progress_bar=False, callbacks=[EpochTimer()],
)
trainer_single.fit(model_single, train_loader, val_loader)
```

- Batch size = 32, précision `16-mixed`, 2 epochs.
- Aucune synchronisation nécessaire : un seul processus, un seul GPU.
- Sert de référence pour mesurer le speedup du multi-GPU.

### Le mur mémoire (OOM)

Avec `batch_size = 1024` sur un seul GPU T4, on obtient de façon fiable une erreur `CUDA out of memory` (à `batch_size = 256`, les tenseurs tiennent tout juste dans les 16 Go, sans garantie). En dehors des poids du modèle, la VRAM est occupée par :

1. les **activations** du forward pass, nécessaires au backward ;
2. les **gradients**, de même taille que les paramètres ;
3. les **états de l'optimiseur AdamW** (moments *m* et *v* → 2× la taille des paramètres).

C'est cette limite qui motive la répartition du batch sur plusieurs GPU.

---

## Expérience B — Entraînement distribué DDP (2 GPU)

```bash
torchrun --standalone --nproc_per_node=2 train_ddp.py \
    --batch-size-per-gpu 128 --epochs 2 \
    --train-size 20000 --val-size 5000 --num-gpus 2
```

- `torchrun` lance 2 processus système indépendants (1 par GPU), chacun recevant les variables d'environnement `RANK`, `LOCAL_RANK` et `WORLD_SIZE`.
- Un `DistributedSampler` donne à chaque GPU une portion disjointe des données.
- Les gradients sont synchronisés via **AllReduce (NCCL)** avant chaque mise à jour des poids — cette synchronisation est **automatique et obligatoire** sous DDP, à chaque pas.
- Seul le processus `rank == 0` écrit les logs et le fichier de résultats, pour éviter des écritures concurrentes dans `results_multi_gpu.json`.
- La synchronisation des **métriques de logging** (`sync_dist=True` dans `self.log(...)`) est en revanche optionnelle : elle ne sert qu'à l'affichage, pas à l'apprentissage.

### Rendre le temps d'entraînement visible

Un `EpochTimer` (callback Lightning) imprime le temps de chaque epoch dès qu'elle se termine, filtré sur `trainer.is_global_zero` pour n'afficher qu'une seule ligne par epoch malgré les deux processus parallèles. Le dataset a volontairement été porté à 20 000 exemples (au lieu de 5 000 initialement) pour que le calcul GPU domine les coûts fixes (téléchargement, tokenisation, démarrage de `torchrun`) et que le speedup mesuré soit fiable.

---

## Résultats mesurés

| Métrique | 1 GPU | 2 GPU (DDP) |
|---|---|---|
| Temps d'entraînement (2 epochs) | 340.5 s | 244.3 s |
| Accuracy de validation | 87.2% | 87.1% |
| **Speedup mesuré** | — | **1.39×** |
| **Temps gagné** | — | **28.2%** |

L'accuracy quasi identique entre les deux configurations confirme que l'AllReduce préserve bien la qualité de l'apprentissage : la parallélisation ne dégrade pas le modèle.

---

## Analyse — Loi d'Amdahl

La loi d'Amdahl prédit le speedup théorique maximal d'un programme partiellement parallélisé :

```
S(N) = 1 / ( s + (1 − s) / N )
```

où *s* est la fraction séquentielle du pipeline (chargement des données, tokenisation, initialisation du modèle, sauvegarde du checkpoint) et *N* le nombre de GPU.

En résolvant `1.39 = 1 / (s + (1−s)/2)` à partir du speedup réellement mesuré :

- **s ≈ 0.44** — bien au-delà des 10% souvent supposés par défaut ;
- **S_max ≈ 2.3×** — speedup maximal théorique, même avec un nombre infini de GPU (`S_max = 1/s`).

**Conclusion** : le pipeline réel a une part séquentielle beaucoup plus grande que prévu. Dans cette configuration, ajouter des GPU au-delà de 2-3 n'apporterait presque rien sans d'abord réduire cette part séquentielle (tokenisation, chargement des données, initialisation du modèle).

---

## Limites

- **Tokenisation dupliquée** : chaque processus DDP re-tokenise le dataset indépendamment, un coût fixe qui ne diminue pas avec le nombre de GPU.
- **Validité limitée à cette configuration** : le speedup de 1.39× et le *s* mesuré à 0.44 décrivent ce setup précis (2× T4, BERT-base, un seul nœud) — ils ne se généralisent pas automatiquement à un cluster multi-nœuds ni à un modèle plus grand.
- **Le speedup a un plafond** : avec *s* ≈ 0.44, ajouter des GPU au-delà de 2-3 n'apporte presque rien sans réduire d'abord la partie séquentielle du pipeline.

---

## Pour aller plus loin

- Passer `--batch-size-per-gpu` à 256 et observer si l'OOM réapparaît même en DDP (pistes : gradient accumulation, ZeRO).
- Comparer le speedup à 2 GPU à un entraînement où seul le batch size est augmenté sur 1 GPU sans parallélisme.
- Étudier l'effet de la précision (32 bits vs `16-mixed` vs `bf16`) sur le temps d'entraînement et la mémoire utilisée.
- Comparer, à charge égale par GPU (batch=128 sur 1 GPU vs 128×2 en DDP), le speedup « pur » du parallélisme sans l'effet confondu de l'augmentation du batch global.
- Relancer l'expérience B avec `--batch-size-per-gpu 512` (batch global de 1024, identique à celui qui provoquait l'OOM sur 1 GPU) pour confirmer qu'il s'exécute sans erreur une fois réparti sur 2 GPU.

---

## À retenir

**Mesurer plutôt que supposer.** La loi d'Amdahl ne vaut que ce que valent vos propres mesures.
