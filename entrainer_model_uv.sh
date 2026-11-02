#!/bin/bash
# Script d'entraînement automatisé avec uv - Régression symbolique (from scratch)
# Config recommandée pour RTX 3060 : d_model 256, 4+4 couches

set -e

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}   ENTRAÎNEMENT AVEC UV - FROM SCRATCH${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

# Détection du répertoire
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}[INFO]${NC} Répertoire de travail : $SCRIPT_DIR"
echo ""

# Vérification des fichiers requis
echo -e "${BLUE}[VÉRIFICATION]${NC} Fichiers requis..."
for f in regression_symbolique.py benchmark_valide.json; do
    if [ ! -f "$f" ]; then
        echo -e "${RED}[ERREUR]${NC} Fichier manquant : $f"
        exit 1
    fi
    echo -e "${GREEN}[OK]${NC} $f"
done
echo ""

# Vérification/installation de uv
echo -e "${BLUE}[UV]${NC} Vérification de uv..."
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}[INFO]${NC} uv non trouvé, installation en cours..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    export PATH="$HOME/.local/bin:$PATH"
    echo -e "${GREEN}[OK]${NC} uv installé"
else
    echo -e "${GREEN}[OK]${NC} $(uv --version)"
fi
echo ""

# Forcer l'utilisation de Python 3.12 (PyTorch ne supporte pas 3.14)
echo -e "${BLUE}[PYTHON]${NC} Recherche de Python 3.12..."
PYTHON_PATH=""

# Vérifier d'abord python3.12
if command -v python3.12 &> /dev/null; then
    ver=$(python3.12 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$ver" == "3.12" ]]; then
        PYTHON_PATH=$(command -v python3.12)
        echo -e "${GREEN}[OK]${NC} Trouvé: $PYTHON_PATH (Python $ver)"
    fi
fi

# Si pas trouvé, chercher dans /usr/bin
if [ -z "$PYTHON_PATH" ] && [ -f "/usr/bin/python3.12" ]; then
    ver=$(/usr/bin/python3.12 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$ver" == "3.12" ]]; then
        PYTHON_PATH="/usr/bin/python3.12"
        echo -e "${GREEN}[OK]${NC} Trouvé: $PYTHON_PATH (Python $ver)"
    fi
fi

# Si toujours pas trouvé, chercher dans /usr/local/bin
if [ -z "$PYTHON_PATH" ] && [ -f "/usr/local/bin/python3.12" ]; then
    PYTHON_PATH="/usr/local/bin/python3.12"
    echo -e "${GREEN}[OK]${NC} Trouvé: $PYTHON_PATH"
fi

if [ -z "$PYTHON_PATH" ]; then
    echo -e "${RED}[ERREUR]${NC} Python 3.12 introuvable."
    echo -e "${YELLOW}[INFO]${NC} Vérifiez avec: which python3.12 ou find /usr -name 'python3.12' 2>/dev/null"
    echo -e "${YELLOW}[INFO]${NC} Ou installez-le à nouveau: yay -S python312"
    exit 1
fi
echo ""

# Supprimer l'ancien venv s'il utilise une mauvaise version
if [ -d ".venv" ]; then
    venv_ver=$(source .venv/bin/activate && python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "unknown")
    echo -e "${BLUE}[INFO]${NC} Venv actuel utilise Python $venv_ver"
    if [[ "$venv_ver" == "3.14" ]]; then
        echo -e "${YELLOW}[INFO]${NC} Suppression du venv Python 3.14..."
        rm -rf .venv
        echo -e "${GREEN}[OK]${NC} Venv supprimé"
    fi
fi

# Créer/environnement avec uv en forçant Python 3.12
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}[INFO]${NC} Création de l'environnement virtuel avec Python 3.12..."
    uv venv --python "$PYTHON_PATH"
    echo -e "${GREEN}[OK]${NC} Environnement virtuel créé"
else
    echo -e "${GREEN}[OK]${NC} Environnement virtuel existant"
fi

echo -e "${BLUE}[ACTIVATION]${NC} Activation du venv..."
source .venv/bin/activate
echo -e "${GREEN}[OK]${NC} Venv activé"
echo ""

echo -e "${BLUE}[INFO]${NC} Python : $(python --version)"
echo -e "${BLUE}[INFO]${NC} Chemin : $(which python)"
echo ""

# Installation de PyTorch avec CUDA 12.1
echo -e "${BLUE}[TORCH]${NC} Installation de PyTorch pour CUDA 12.1 avec uv..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo -e "${GREEN}[OK]${NC} PyTorch installé"
echo ""

# Installation des dépendances
echo -e "${BLUE}[DÉPENDANCES]${NC} Installation de numpy, scipy, sympy, matplotlib..."
uv pip install numpy scipy sympy matplotlib
echo -e "${GREEN}[OK]${NC} Dépendances installées"
echo ""

# Vérification GPU
echo -e "${BLUE}[VÉRIFICATION GPU]${NC}"
python -c "
import torch
print('Python version   :', torch.__file__.split('/')[0] if hasattr(torch, '__file__') else 'N/A')
print('CUDA disponible :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU détecté      :', torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print('Mémoire totale   : {:.1f} GB'.format(props.total_memory / 1024**3))
else:
    print('ATTENTION : Aucun GPU CUDA détecté')
"
echo ""

# Lancement entraînement
echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}[DÉMARRAGE]${NC} Entraînement from scratch 256/4/4"
echo -e "${BLUE}==================================================${NC}"
echo ""
echo -e "${YELLOW}[CONFIG]${NC} d_model=256 | 4+4 couches | n-heads=8 | ff=1024 | batch_size=48 | steps=40000"
echo -e "${YELLOW}[INFO]${NC}  Le meilleur checkpoint sera sauvegardé automatiquement"
echo ""

python regression_symbolique.py \
  --steps 40000 \
  --d-model 256 \
  --n-heads 8 \
  --enc-layers 4 \
  --dec-layers 4 \
  --ff 1024 \
  --dropout 0.1 \
  --batch-size 48 \
  --max-depth 3 \
  --n-points 64 \
  --canonique \
  --loss-const huber \
  --lr 2e-4 \
  --warmup 3000 \
  --eval-every 2000 \
  --eval-samples 100 \
  --examples 10 \
  --ckpt modele_gros_scratch.pt

echo ""
echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}[TERMINÉ]${NC} Entraînement terminé"
echo -e "${BLUE}==================================================${NC}"
echo ""

if [ -f "modele_gros_scratch.pt" ]; then
    echo -e "${BLUE}[TEST]${NC} Pour comparer avec l'actuel :"
    echo -e "${YELLOW}python regression_symbolique.py --eval-only --ckpt modele_gros_scratch.pt --canonique --benchmark benchmark_valide.json --benchmark-repetitions 3 --candidates 48${NC}"
    echo ""
    echo -e "${BLUE}[COPIE]${NC} Si meilleur :"
    echo -e "${YELLOW}mv modele_gros_scratch.pt modele_regression.pt${NC}"
fi

echo -e "${GREEN}[FIN]${NC}"
