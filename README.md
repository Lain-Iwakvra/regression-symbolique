# Régression Symbolique

Un réseau de neurones Transformer qui retrouve la **formule mathématique** d'une fonction à partir d'un nuage de points.

## Principe

Le modèle encode un nuage `(x, y)` et décode l'expression sous forme d'arbre syntaxique contraint par une grammaire mathématique. À l'inférence, il génère plusieurs formules, affine les constantes par moindres carrés et sélectionne la meilleure.

## Fonctionnalités

- **Interface graphique** (`interface.py`) - Saisir `f(x)`, générer un nuage de points ou tracer à la souris pour retrouver la formule
- **Tracer à la souris** - Placer manuellement des points sur le graphique et laisser l'IA deviner la formule
- **Import CSV** - Charger des points depuis un fichier CSV
- **Ligne de commande** - `--predict`, `--predict-file`, `--verifier`
- **Polyvalent** - Fonctionne avec différents nombres de points (4-256) et domaines
- **Affinage automatique** - Beam search, dédoublonnage canonique/sémantique, optimisation des constantes (Levenberg–Marquardt)
- **Simplification symbolique** - Simplification via SymPy pour des formules plus lisibles
- **Benchmark figé** - Évaluation reproductible sur un jeu de fonctions identiques

## Installation

### Avec uv (recommandé)

```bash
git clone https://github.com/Lain-Iwakvra/regression-symbolique.git
cd regression-symbolique

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python python3.12
source .venv/bin/activate

# GPU (CUDA 12.1)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# ou CPU
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

uv pip install numpy scipy sympy matplotlib
```

### Avec pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # GPU
# ou: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # CPU
pip install numpy scipy sympy matplotlib
```

## Utilisation

### Interface graphique

```bash
python interface.py
```

- **Mode fonction** : Entrer `f(x)` et cliquer sur "Prédire la fonction"
- **Mode tracé manuel** : Sélectionner "Tracer des points", cliquer sur le graphique pour placer des points, puis prédire
- **Mode CSV** : Charger un fichier CSV avec des points `(x,y)`

### Ligne de commande

```bash
# Depuis des points inline
python regression_symbolique.py --predict "0,0;1,1;2,4;3,9"

# Depuis un fichier CSV/TXT (séparateurs ; , tab ou espace)
python regression_symbolique.py --predict-file points.csv

# Vérifier une formule contre un nuage généré
python interface.py --verifier "2.5*x - 1" --points 64
```

### Entraîner un modèle

```bash
./entrainer_model_uv.sh
```

Le script est configurable via des variables d'environnement. Par défaut, le checkpoint est nommé automatiquement selon la taille du modèle (ex: `modele_d256_e4d4_ff1024.pt`).
- `--resume` recharge le meilleur état connu et refuse la reprise si la config du checkpoint ne correspond pas aux arguments.
- Au début d'un réentraînement, le premier checkpoint n'est pas sauvegardé automatiquement (pour éviter d'écraser un meilleur état).

### Comparer des modèles

```bash
python comparer_modeles.py modele_regression.pt autre_modele.pt --samples 200 --seed 1234 --candidates 48 --refine 8
```

## Modèles fournis

| Fichier | d_model | Couches | ff | Têtes | Params | Remarques |
|---|---:|---:|---:|---:|---:|---|
| `modele_regression_256.pt` | 256 | 4/4 | 1024 | 8 | ~7.4M | Léger, bon compromis vitesse/précision |
| `modele_regression_320.pt` | 320 | 4/4 | 1280 | 10 | ~11.6M | Plus grand, légèrement meilleur sur benchmarks |

Capacités communes : 4-256 points d'entrée, longueur max formule 64 tokens.

## Fichiers

| Fichier | Description |
|---|---|
| `regression_symbolique.py` | Cœur du modèle, entraînement, inférence et prédiction |
| `interface.py` | Interface graphique Tkinter + Matplotlib |
| `comparer_modeles.py` | Comparaison de modèles sur des fonctions identiques |
| `entrainer_model_uv.sh` | Script d'entraînement configurable (GPU/CPU) |
| `modele_regression_256.pt` | Modèle pré-entraîné (léger) |
| `modele_regression_320.pt` | Modèle pré-entraîné (plus grand) |

## Optimisations

Le code intègre plusieurs optimisations pour de meilleures performances :
- Masquage grammatical vectorisé/caché
- Décodage optimisé avec caches de masques
- `torch.compile`, AMP, `cudnn.benchmark`
- Génération parallèle des batches (sur CPU)
- Affinage efficace des constantes (Levenberg–Marquardt)

## Licence

MIT
