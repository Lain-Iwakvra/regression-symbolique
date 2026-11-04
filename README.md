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
- **Affinage automatique** - Beam search, dédoublonnage canonique, optimisation des constantes

## Installation

```bash
git clone https://github.com/Lain-Iwakvra/regression-symbolique.git
cd regression-symbolique
```

### Avec uv (recommandé)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python python3.12
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # GPU
# ou: uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # CPU
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
# Depuis des points
python regression_symbolique.py --predict "0,0;1,1;2,4;3,9"

# Depuis un fichier CSV/TXT
python regression_symbolique.py --predict-file points.csv

# Vérifier une formule
python interface.py --verifier "2.5*x - 1" --points 64
```

### Entraîner un modèle

```bash
./entrainer_model_uv.sh
```

Le script est configurable via des variables d'environnement. Voir le script pour plus de détails.

### Comparer des modèles

```bash
python comparer_modeles.py modele_regression.pt autre_modele.pt --samples 200 --seed 1234 --candidates 48 --refine 8
```

## Modèle

| Paramètre | Valeur |
|---|---|
| Architecture | Transformer encodeur-décodeur |
| `d_model` | 256 |
| Couches encodeur / décodeur | 4 / 4 |
| Têtes d'attention | 8 |
| Dimension FFN | 1024 |
| Paramètres | ~7.4M |
| Points d'entrée | 4-256 |
| Longueur max formule | 64 tokens |

## Fichiers

| Fichier | Description |
|---|---|
| `regression_symbolique.py` | Cœur du modèle, entraînement et inférence |
| `interface.py` | Interface graphique Tkinter + Matplotlib |
| `entrainer_model_uv.sh` | Script d'entraînement configurable (GPU/CPU) |
| `comparer_modeles.py` | Comparaison de modèles sur des fonctions identiques |
| `modele_regression.pt` | Modèle pré-entraîné |

## Licence

MIT
