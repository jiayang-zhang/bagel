export PYTHONWARNINGS="ignore"

cd /rds/general/user/jz2319/home/bagel/scripts/technical-report/surface-design || exit 1

SCRIPT="surface_design_route1.py"
CONFIG="configs/surface_design_route1_grid_search.yaml"
BASE_OUTDIR="data/surface-structure-optim-route1-gridsearch"
mkdir -p "$BASE_OUTDIR"


vals=(1 5 10 15 20)
bools=(True False)

for i in "${vals[@]}"; do
  for j in "${vals[@]}"; do
    for k in "${vals[@]}"; do
        
        OUTDIR="${BASE_OUTDIR}/tme_${k}_ptm_${j}_plddt_${i}"
        mkdir -p "$OUTDIR"

        python "$SCRIPT" \
            --config "$CONFIG" \
            --outdir "$OUTDIR" \
            --tme_distogram True \
            --tme_weight "$k" \
            --ptm_weight "$j" \
            --overall_plddt_weight "$i"
            
    done
  done
done
