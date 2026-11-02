# Régression Symbolique

Un réseau de neurones Transformer qui **retrouve la formule mathématique** d'une fonction à partir d'un simple nuage de points. Plutôt que d'interpoler, il devine l'expression analytique sous-jacente.

## Principe

1. **Génération** - Des fonctions aléatoires sont construites à partir d'une grammaire (`+`, `-`, `*`, `/`, `sin`, `cos`, `exp`, `log`, `sqrt`, `tanh`, `x`, constantes)
2. **Apprentissage** - Un Transformer encode le nuage de points `(x,y)` et décode la formule sous forme de jetons avec masquage grammatical
3. **Inférence** - Génère plusieurs candidats, affine les constantes par moindres carrés, sélectionne le meilleur par RMSE relatif + pénalité de complexité

## Fonctionnalités

- **Deviner depuis une fonction** - Saisis `f(x)` et génère un nuage de points aléatoire
- **Tracer à la souris** - Place tes propres points directement sur le graphique pour que l'IA retrouve la formule
- **Interface graphique** - Visualisation avec Matplotlib, affichage de la formule en LaTeX
- **Polyvalent** - Accepte des nuages de tailles et de domaines variables
- **Affinage intelligent** - Beam search, dédoublonnage canonique, optimisation des constantes

## Installation

### Prérequis

- Python 3.11, 3.12 ou 3.13 (PyTorch ne supporte pas encore Python 3.14)
- CUDA (recommandé) ou CPU

### Installation rapide

```bash
# Cloner le repo
git clone https://github.com/Lain-Iwakvra/regression-symbolique.git
cd regression-symbolique

# Avec uv (recommandé)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python python3.12
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install numpy scipy sympy matplotlib

# Ou avec pip classique
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy sympy matplotlib
```

## Utilisation

### Interface graphique

```bash
python interface.py
```

Depuis l'interface :
1. **Mode fonction** : Entre `f(x)` et lance la prédiction
2. **Mode tracé manuel** : Coche "tracer à la souris", clique sur le graphique pour placer tes points, puis lance la prédiction

### Ligne de commande

```bash
# Deviner depuis des points
python regression_symbolique.py --predict "0,0;1,1;2,4;3,9"

# Depuis un fichier CSV
python regression_symbolique.py --predict-file points.csv

# Vérifier sans interface
python interface.py --verifier "2.5*x - 1"
```

### Entraîner son propre modèle

```bash
# Avec le script automatique (s'adapte au matériel)
./entrainer_model_uv.sh
```

## Architecture

- **Encodeur** : Transformer sur le nuage de points trié avec embeddings sinusoidaux
- **Décodeur** : Transformer autoregressif avec masquage grammatical
- **Inférence** : Sampling + beam search, dédoublonnage canonique, affinage LM
- **Polyvalent** : Gère n'importe quel nombre de points (4-256) dans n'importe quel domaine

## Licence

MIT
