export PYTHONWARNINGS="ignore"

cd /rds/general/user/jz2319/home/bagel/scripts/technical-report/surface-design || exit 1

SCRIPT="lysozyme_activesite_route1.py"
CONFIG="configs/lysozyme_route1.yaml"
BASE_OUTDIR="data/lysozyme-route1"
mkdir -p "$BASE_OUTDIR"

python "$SCRIPT" \
    --config "$CONFIG" \
    --outdir "$BASE_OUTDIR" \
    --tme_distogram True \
    --tme_weight 0.8 \
    --ptm_weight 5 \
    --overall_plddt_weight 1 \
    --hydrophilic_surface_posweight 2 \
    --hydrophobic_core_negweight -2 \
    --SASA -0.2 \
    --secondarystructure_weight 0.2 
