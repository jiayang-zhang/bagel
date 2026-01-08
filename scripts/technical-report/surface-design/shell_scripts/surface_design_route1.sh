export PYTHONWARNINGS="ignore"

cd /rds/general/user/jz2319/home/bagel/scripts/technical-report/surface-design || exit 1

SCRIPT="surface_design_route1.py"
CONFIG="configs/surface_design_route1_grid_search.yaml"
BASE_OUTDIR="data/surface-structure-optim-route1-gridsearch"
mkdir -p "$BASE_OUTDIR"

python "$SCRIPT" \
    --config "$CONFIG" \
    --outdir "$BASE_OUTDIR" \
    --tme_distogram True \
    --tme_weight 1 \
    --ptm_weight 1 \
    --overall_plddt_weight 1
