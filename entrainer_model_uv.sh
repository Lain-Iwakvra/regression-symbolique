#!/bin/bash
# Script d'entraînement automatisé avec uv - Régression symbolique (from scratch)
# Configurable via variables d'environnement

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}   ENTRAÎNEMENT CONFIGURABLE - REGRESSION SYMBOLIQUE${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

# Répertoire
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration par défaut (modèle actuel)
D_MODEL=${D_MODEL:-256}
N_HEADS=${N_HEADS:-8}
ENC_LAYERS=${ENC_LAYERS:-4}
DEC_LAYERS=${DEC_LAYERS:-4}
FF=${FF:-1024}
DROPOUT=${DROPOUT:-0.05}
BATCH_SIZE=${BATCH_SIZE:-32}
STEPS=${STEPS:-100000}
LR=${LR:-3e-4}
WARMUP=${WARMUP:-1000}
MAX_DEPTH=${MAX_DEPTH:-3}
N_POINTS=${N_POINTS:-64}
W_CONST=${W_CONST:-1.0}
LOSS_CONST=${LOSS_CONST:-huber}
CANONIQUE=${CANONIQUE:-1}
EVAL_EVERY=${EVAL_EVERY:-1000}
EVAL_SAMPLES=${EVAL_SAMPLES:-100}
NUM_WORKERS=${NUM_WORKERS:-0}
CKPT_NAME=${CKPT_NAME:-"modele_regression.pt"}
RS_NO_COMPILE=${RS_NO_COMPILE:-1}
export RS_NO_COMPILE

# Générer nom auto si non spécifié
if [ -z "$CKPT_NAME" ]; then
    CKPT_NAME="modele_regression_d${D_MODEL}_e${ENC_LAYERS}d${DEC_LAYERS}_ff${FF}.pt"
fi

# Affichage config
echo -e "${CYAN}[CONFIGURATION]${NC}"
echo -e "  Architecture : d_model=${D_MODEL} | heads=${N_HEADS} | enc=${ENC_LAYERS} dec=${DEC_LAYERS} | ff=${FF}"
echo -e "  Entraînement  : batch=${BATCH_SIZE} | steps=${STEPS} | lr=${LR} | warmup=${WARMUP} | dropout=${DROPOUT}"
echo -e "  Données       : max_depth=${MAX_DEPTH} | n_points=${N_POINTS} | canonique=$([ "$CANONIQUE" = "1" ] && echo oui || echo non)"
echo -e "  Perte         : loss_const=${LOSS_CONST} | w_const=${W_CONST}"
echo -e "  Évaluation    : eval_every=${EVAL_EVERY} | eval_samples=${EVAL_SAMPLES}"
echo -e "  Workers       : ${NUM_WORKERS}"
echo -e "  Checkpoint    : ${CKPT_NAME}"
echo ""

# Vérification
for f in regression_symbolique.py; do
    if [ ! -f "$f" ]; then echo -e "${RED}[ERREUR]${NC} $f manquant"; exit 1; fi
done
echo -e "${GREEN}[OK]${NC} Fichiers requis présents"

# uv
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}[INFO]${NC} Installation uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    export PATH="$HOME/.local/bin:$PATH"
fi

# Python
PYTHON_PATH=""
for py in python3.12 python3.11 python3.13; do
    if command -v $py &> /dev/null; then
        p=$(command -v $py)
        ver=$($p -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
        if [[ "$ver" == "3.11" || "$ver" == "3.12" || "$ver" == "3.13" ]]; then
            PYTHON_PATH=$p; break
        fi
    fi
done
[ -z "$PYTHON_PATH" ] && echo -e "${RED}[ERREUR]${NC} Python 3.11-3.13 requis" && exit 1

# venv
[ -d ".venv" ] && venv_ver=$(source .venv/bin/activate 2>/dev/null && python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo x)
[[ "$venv_ver" == "3.14" ]] && rm -rf .venv
[ ! -d ".venv" ] && uv venv --python "$PYTHON_PATH"
source .venv/bin/activate
echo -e "${GREEN}[OK]${NC} $(python --version)"

# Déps
echo -e "${BLUE}[INSTALL]${NC} Dépendances..."
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
    echo -e "${YELLOW}[INFO]${NC} PyTorch CUDA détecté"
else
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 >/dev/null 2>&1 || uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1
fi
uv pip install numpy scipy sympy matplotlib >/dev/null 2>&1
echo -e "${GREEN}[OK]${NC} Prêt"
echo ""

# Reprise auto
RESUME_ARGS=()
if [ -f "$CKPT_NAME" ]; then
    echo -e "${YELLOW}[INFO]${NC} Reprise depuis $CKPT_NAME (--resume)"
    RESUME_ARGS=(--resume)
fi

CANON_ARGS=()
[ "$CANONIQUE" = "1" ] && CANON_ARGS=(--canonique)

echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}[LANCEMENT]${NC}"
echo -e "${BLUE}==================================================${NC}"
echo ""

python regression_symbolique.py \
  --steps $STEPS \
  --d-model $D_MODEL \
  --n-heads $N_HEADS \
  --enc-layers $ENC_LAYERS \
  --dec-layers $DEC_LAYERS \
  --ff $FF \
  --dropout $DROPOUT \
  --batch-size $BATCH_SIZE \
  --max-depth $MAX_DEPTH \
  --n-points $N_POINTS \
  "${CANON_ARGS[@]}" \
  --loss-const $LOSS_CONST \
  --w-const $W_CONST \
  --lr $LR \
  --warmup $WARMUP \
  --eval-every $EVAL_EVERY \
  --eval-samples $EVAL_SAMPLES \
  --examples 10 \
  --num-workers $NUM_WORKERS \
  --ckpt "$CKPT_NAME" \
  "${RESUME_ARGS[@]}"

echo ""
echo -e "${GREEN}[TERMINÉ]${NC} Checkpoint: $CKPT_NAME"
