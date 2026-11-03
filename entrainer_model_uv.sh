#!/bin/bash
# Script d'entraînement automatisé avec uv - Régression symbolique (from scratch)
# Progressif : légèrement plus gros que le modèle actuel

set -e

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}   ENTRAÎNEMENT PROGRESSIF - d_model 288${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

# Détection du répertoire
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${GREEN}[INFO]${NC} Répertoire de travail : $SCRIPT_DIR"
echo ""

# Vérification des fichiers requis
for f in regression_symbolique.py; do
    if [ -f "$f" ]; then
        echo -e "${GREEN}[OK]${NC} $f"
    else
        echo -e "${RED}[ERREUR]${NC} $f manquant"
        exit 1
    fi
done
echo ""

# Vérification/installation de uv
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}[INFO]${NC} Installation de uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    export PATH="$HOME/.local/bin:$PATH"
fi
echo -e "${GREEN}[OK]${NC} $(uv --version 2>/dev/null || echo 'uv installé')"
echo ""

# Détection Python compatible
PYTHON_PATH=""
for py in python3.12 python3.11 python3.13 /usr/bin/python3.12 /usr/local/bin/python3.12; do
    if [ -f "$py" ] || command -v $py &> /dev/null; then
        p=$(command -v $py 2>/dev/null || echo $py)
        if [ -x "$p" ]; then
            ver=$($p -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
            if [[ "$ver" == "3.11" || "$ver" == "3.12" || "$ver" == "3.13" ]]; then
                PYTHON_PATH=$p
                echo -e "${GREEN}[OK]${NC} Python $ver : $p"
                break
            fi
        fi
    fi
done

if [ -z "$PYTHON_PATH" ]; then
    echo -e "${RED}[ERREUR]${NC} Python 3.11/3.12/3.13 requis. Installez-le."
    exit 1
fi
echo ""

# Nettoyer venv si mauvaise version
if [ -d ".venv" ]; then
    venv_ver=$(source .venv/bin/activate 2>/dev/null && python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "unknown")
    if [[ "$venv_ver" == "3.14" ]]; then
        rm -rf .venv
    fi
fi

[ ! -d ".venv" ] && uv venv --python "$PYTHON_PATH"
source .venv/bin/activate
echo -e "${GREEN}[OK]${NC} $(python --version)"
echo ""

# Installation deps
echo -e "${BLUE}[INSTALLATION]${NC} Dépendances..."
FORCE_CPU=${FORCE_CPU:-0}
if [ $FORCE_CPU -eq 1 ]; then
    echo -e "${YELLOW}[INFO]${NC} FORCE_CPU=1 -> PyTorch CPU"
    uv pip uninstall torch torchvision torchaudio -y >/dev/null 2>&1 || true
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
        echo -e "${YELLOW}[INFO]${NC} PyTorch CUDA détecté"
    else
        echo -e "${YELLOW}[INFO]${NC} Tentative d'installation PyTorch CUDA (cu121)..."
        uv pip uninstall torch torchvision torchaudio -y >/dev/null 2>&1 || true
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
fi
uv pip install numpy scipy sympy matplotlib -q 2>/dev/null || uv pip install numpy scipy sympy matplotlib
echo -e "${GREEN}[OK]${NC} Dépendances installées"
echo ""

echo -e "${BLUE}[CONFIG]${NC} d_model=288 | enc=4, dec=4 | heads=9 | ff=1152 | batch=64 | steps=35000 | lr=3e-4 | warmup=1500 | max_depth=4 | canonique + huber"
echo ""

# Lancement
echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}[DÉMARRAGE]${NC} Entraînement progressif"
echo -e "${BLUE}==================================================${NC}"
echo ""

python regression_symbolique.py \
  --steps 35000 \
  --d-model 288 \
  --n-heads 9 \
  --enc-layers 4 \
  --dec-layers 4 \
  --ff 1152 \
  --dropout 0.08 \
  --batch-size 64 \
  --max-depth 4 \
  --n-points 64 \
  --canonique \
  --loss-const huber \
  --w-const 1.2 \
  --lr 3e-4 \
  --warmup 1500 \
  --eval-every 1000 \
  --eval-samples 100 \
  --examples 10 \
  --num-workers 8 \
  --ckpt modele_regression_m288.pt

echo ""
echo -e "${GREEN}[TERMINÉ]${NC} Comparer avec modele_regression.pt (étape 28000, best 0.69)"
echo -e "${YELLOW}Si meilleur : mv modele_regression_m288.pt modele_regression.pt${NC}"
