#!/usr/bin/env bash
# run_experiment.sh — cài requirements, tạo splits, chạy 1 trong 3 ablation configs
#
# Cách dùng:
#   bash run_experiment.sh 1        # train với 500 ảnh  (luot1_500.yaml)
#   bash run_experiment.sh 2        # train với 1000 ảnh (luot2_1000.yaml)
#   bash run_experiment.sh 3        # train với 1500 ảnh (luot3_1500.yaml)
#   bash run_experiment.sh 1 --dry-run   # smoke-test (5 iters)

set -e  # dừng ngay nếu có lỗi

# ── Cấu hình đường dẫn ───────────────────────────────────────────────────────
DATA_ROOT="/kaggle/input/datasets/dyiyacao/openearthmap"
WORK_BASE="/kaggle/working/unetformer-openearthmap"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Đọc tham số ──────────────────────────────────────────────────────────────
EXPERIMENT="${1:-}"
DRY_RUN="${2:-}"

if [[ -z "$EXPERIMENT" ]]; then
    echo "Cách dùng: bash run_experiment.sh <1|2|3> [--dry-run]"
    echo "  1 → train với 500 ảnh  (luot1_500.yaml)"
    echo "  2 → train với 1000 ảnh (luot2_1000.yaml)"
    echo "  3 → train với 1500 ảnh (luot3_1500.yaml)"
    exit 1
fi

case "$EXPERIMENT" in
    1) CONFIG="configs/luot1_500.yaml";  LABEL="luot1 — 500 ảnh"  ;;
    2) CONFIG="configs/luot2_1000.yaml"; LABEL="luot2 — 1000 ảnh" ;;
    3) CONFIG="configs/luot3_1500.yaml"; LABEL="luot3 — 1500 ảnh" ;;
    *)
        echo "Lỗi: experiment phải là 1, 2 hoặc 3 (nhận được: '$EXPERIMENT')"
        exit 1
        ;;
esac

echo "========================================================"
echo " UNetFormer Ablation — $LABEL"
if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo " [DRY-RUN] Chỉ chạy 5 iterations để kiểm tra"
fi
echo "========================================================"

# ── Bước 1: Cài requirements ─────────────────────────────────────────────────
echo ""
echo "[1/3] Cài đặt requirements ..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"
echo "      Done."

# ── Bước 2: Tạo split files ───────────────────────────────────────────────────
echo ""
echo "[2/3] Tạo split files ..."
python "$SCRIPT_DIR/Tools/create_splits.py" \
    --data-root "$DATA_ROOT" \
    --output-dir "$WORK_BASE"
echo "      Done."

# ── Bước 3: Chạy training ────────────────────────────────────────────────────
echo ""
echo "[3/3] Bắt đầu training: $CONFIG"
echo "      Kết quả lưu tại: $WORK_BASE/work_dirs/"
echo ""

cd "$SCRIPT_DIR"

if [[ "$DRY_RUN" == "--dry-run" ]]; then
    torchrun --nproc_per_node=2 src/train.py --config "$CONFIG" --dry-run
else
    torchrun --nproc_per_node=2 src/train.py --config "$CONFIG"
fi

echo ""
echo "========================================================"
echo " Hoàn thành: $LABEL"
echo " Xem kết quả tại: $WORK_BASE/work_dirs/"
echo "========================================================"
