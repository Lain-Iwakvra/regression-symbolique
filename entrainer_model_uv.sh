#!/bin/bash
# Script d'entraînement automatisé avec uv - Régression symbolique (from scratch)
# S'adapte automatiquement à la configuration matérielle

set -e

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}   ENTRAÎNEMENT ADAPTATIF - FROM SCRATCH${NC}"
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

# Détection GPU NVIDIA (RTX/GTX)
HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    HAS_NVIDIA=1
    echo -e "${GREEN}[OK]${NC} nvidia-smi détecté"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1
fi
if [ $HAS_NVIDIA -eq 0 ] && [ -e /dev/nvidia0 ] || [ -e /dev/nvidiactl ] || [ -e /dev/nvidia-uvm ]; then
    HAS_NVIDIA=1
    echo -e "${GREEN}[OK]${NC} Périphériques NVIDIA détectés"
fi
if [ $HAS_NVIDIA -eq 0 ] && command -v lspci >/dev/null 2>&1 && lspci | grep -qi "nvidia"; then
    HAS_NVIDIA=1
    echo -e "${GREEN}[OK]${NC} Carte NVIDIA détectée via lspci"
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
FORCE_CUDA=${FORCE_CUDA:-0}
FORCE_CPU=${FORCE_CPU:-0}
if [ $FORCE_CPU -eq 1 ]; then
    echo -e "${YELLOW}[INFO]${NC} FORCE_CPU=1 -> PyTorch CPU"
    uv pip uninstall torch torchvision torchaudio -y >/dev/null 2>&1 || true
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
elif [ $HAS_NVIDIA -eq 1 ] || [ $FORCE_CUDA -eq 1 ]; then
    echo -e "${YELLOW}[INFO]${NC} Tentative d'installation PyTorch CUDA (cu121)..."
    uv pip uninstall torch torchvision torchaudio -y >/dev/null 2>&1 || true
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
        echo -e "${YELLOW}[INFO]${NC} PyTorch CUDA déjà présent"
    else
        echo -e "${YELLOW}[INFO]${NC} Installation de PyTorch CPU"
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
fi
uv pip install numpy scipy sympy matplotlib -q 2>/dev/null || uv pip install numpy scipy sympy matplotlib
echo -e "${GREEN}[OK]${NC} Dépendances installées"
echo ""

# Détection matérielle
echo -e "${BLUE}[DÉTECTION MATÉRIELLE]${NC}"
python << 'PYEOF'
import torch, os
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu} ({vram:.1f} GB VRAM)")
    if vram >= 14:
        cfg = {"d_model": 384, "enc_layers": 6, "dec_layers": 6, "nhead": 12, "ff": 1536, "batch": 96, "steps": 50000}
    elif vram >= 11.0:
        cfg = {"d_model": 320, "enc_layers": 5, "dec_layers": 5, "nhead": 10, "ff": 1280, "batch": 96, "steps": 48000}
    elif vram >= 10.0:
        cfg = {"d_model": 320, "enc_layers": 5, "dec_layers": 5, "nhead": 10, "ff": 1280, "batch": 80, "steps": 46000}
    elif vram >= 9.0:
        cfg = {"d_model": 288, "enc_layers": 5, "dec_layers": 5, "nhead": 9, "ff": 1152, "batch": 80, "steps": 45000}
    elif vram >= 7.5:
        cfg = {"d_model": 256, "enc_layers": 4, "dec_layers": 5, "nhead": 8, "ff": 1024, "batch": 72, "steps": 42000}
    elif vram >= 6.0:
        cfg = {"d_model": 224, "enc_layers": 4, "dec_layers": 4, "nhead": 7, "ff": 896, "batch": 48, "steps": 40000}
    else:
        cfg = {"d_model": 192, "enc_layers": 4, "dec_layers": 4, "nhead": 6, "ff": 768, "batch": 32, "steps": 38000}
    print(f"Config GPU: d_model={cfg['d_model']} | {cfg['enc_layers']}+{cfg['dec_layers']} | batch={cfg['batch']} | steps={cfg['steps']} | ff={cfg['ff']}")
    with open('/tmp/train_config.sh','w') as f:
        for k,v in cfg.items(): f.write(f"{k.upper()}={v}\n")
else:
    print("CUDA non détecté - passage en mode CPU haute qualité")
    with open('/tmp/train_config.sh','w') as f:
        f.write("D_MODEL=320\nENC_LAYERS=5\nDEC_LAYERS=5\nNHEAD=10\nFF=1280\nBATCH=64\nSTEPS=45000\n")
PYEOF
source /tmp/train_config.sh
echo ""

# Lancement
echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}[DÉMARRAGE]${NC} Entraînement adaptatif from scratch"
echo -e "${BLUE}==================================================${NC}"
echo ""

python regression_symbolique.py \
  --steps ${STEPS} \
  --d-model ${D_MODEL} \
  --n-heads ${NHEAD} \
  --enc-layers ${ENC_LAYERS} \
  --dec-layers ${DEC_LAYERS} \
  --ff ${FF} \
  --dropout 0.08 \
  --batch-size ${BATCH} \
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
  --ckpt modele_regression_nouveau.pt

echo ""
echo -e "${GREEN}[TERMINÉ]${NC} Si meilleur que l'actuel, renommez :"
echo -e "${YELLOW}mv modele_regression_nouveau.pt modele_regression.pt${NC}"
rm -f /tmp/train_config.sh
