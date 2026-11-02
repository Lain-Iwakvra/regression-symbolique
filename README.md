# Régression Symbolique

Un réseau de neurones Transformer qui retrouve la **formule mathématique** d'une fonction à partir d'un nuage de points.

## Principe

Le modèle encode un nuage `(x, y)` et décode l'expression sous forme d'arbre syntaxique contraint par une grammaire mathématique. À l'inférence, il génère plusieurs formules, affine les constantes par moindres carrés et sélectionne la meilleure.

## Fonctionnalités

- **Interface graphique** (`interface.py`) - Saisir `f(x)`, générer un nuage de points et voir la formule retrouvée
- **Tracer à la souris** - Placer manuellement des points sur le graphique et laisser l'IA deviner la formule
- **Ligne de commande** - `--predict`, `--predict-file`, `--verifier`
- **Polyvalent** - Fonctionne avec différents nombres de points et domaines
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
python3 -m venv .venv
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
- **Mode tracé manuel** : Cocher "tracer à la souris", cliquer sur le graphique pour placer des points, puis "Prédire la fonction"

### Ligne de commande

```bash
# Depuis des points
python regression_symbolique.py --predict "0,0;1,1;2,4;3,9"

# Depuis un fichier
python regression_symbolique.py --predict-file points.csv

# Test rapide sans interface
python interface.py --verifier "2.5*x - 1" --points 64
```

### Entraîner un modèle

```bash
./entrainer_model_uv.sh
```

Le script s'adapte automatiquement à votre matériel (GPU avec détection VRAM, ou CPU).

## Fichiers

| Fichier | Description |
|---|---|
| `regression_symbolique.py` | Cœur du modèle, entraînement et inférence |
| `interface.py` | Interface graphique Tkinter + Matplotlib |
| `entrainer_model_uv.sh` | Script d'entraînement adaptatif (GPU/CPU) |
| `modele_regression.pt` | Modèle pré-entraîné |

## Licence

MIT
