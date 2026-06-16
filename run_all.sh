#!/usr/bin/env bash
# run_all.sh — Cài requirements + tạo splits 1 lần, sau đó train cả 3 ablation config.
#
# Cách dùng:
#   bash run_all.sh              # train đủ 3 lượt
#   bash run_all.sh --dry-run    # smoke-test (5 iters mỗi lượt)

set -e

DATA_ROOT="/kaggle/input/datasets/dyiyacao/openearthmap"
WORK_BASE="/kaggle/working/unetformer-openearthmap"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN="${1:-}"

echo "========================================================"
echo " UNetFormer — Full Ablation (3 experiments)"
if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo " [DRY-RUN] 5 iterations mỗi lượt"
fi
echo "========================================================"

# ── Bước 1: Cài requirements (1 lần) ────────────────────────────────────────
echo ""
echo "[1/5] Cài đặt requirements ..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"
echo "      Done."

# ── Bước 2: Tạo splits (1 lần) ───────────────────────────────────────────────
echo ""
echo "[2/5] Tạo split files ..."
python "$SCRIPT_DIR/Tools/create_splits.py" \
    --data-root "$DATA_ROOT" \
    --output-dir "$WORK_BASE"
echo "      Done."

cd "$SCRIPT_DIR"

# ── Bước 3: Train luot1 — 500 ảnh ───────────────────────────────────────────
echo ""
echo "[3/5] Training luot1 — 500 ảnh ..."
if [[ "$DRY_RUN" == "--dry-run" ]]; then
    torchrun --nproc_per_node=2 src/train.py --config configs/luot1_500.yaml --dry-run
else
    torchrun --nproc_per_node=2 src/train.py --config configs/luot1_500.yaml
fi
echo "      Luot 1 hoàn thành."

# ── Bước 4: Train luot2 — 1000 ảnh ──────────────────────────────────────────
echo ""
echo "[4/5] Training luot2 — 1000 ảnh ..."
if [[ "$DRY_RUN" == "--dry-run" ]]; then
    torchrun --nproc_per_node=2 src/train.py --config configs/luot2_1000.yaml --dry-run
else
    torchrun --nproc_per_node=2 src/train.py --config configs/luot2_1000.yaml
fi
echo "      Luot 2 hoàn thành."

# ── Bước 5: Train luot3 — 1500 ảnh ──────────────────────────────────────────
echo ""
echo "[5/5] Training luot3 — 1500 ảnh ..."
if [[ "$DRY_RUN" == "--dry-run" ]]; then
    torchrun --nproc_per_node=2 src/train.py --config configs/luot3_1500.yaml --dry-run
else
    torchrun --nproc_per_node=2 src/train.py --config configs/luot3_1500.yaml
fi
echo "      Luot 3 hoàn thành."

echo ""
echo "========================================================"
echo " Tất cả experiments hoàn thành."
echo " Kết quả tại: $WORK_BASE/work_dirs/"
echo "   time_500.txt   — thời gian luot1"
echo "   time_1000.txt  — thời gian luot2"
echo "   time_1500.txt  — thời gian luot3"
echo "========================================================"
