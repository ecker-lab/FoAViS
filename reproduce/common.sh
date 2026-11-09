# Shared by table2.sh and table3.sh: evaluate one table cell and append it to $OUT/cells.csv.
#
# Environment (all optional):
#   CKPT_DIR   checkpoints, named <dataset>_<arm>_s<seed>.pth (default ./checkpoints); missing seed-1 ones are downloaded
#   DATA_DIR   dataset root (default ./data)
#   OUT        results dir (default ./reproduce_out): cells.csv, logs/, scanpaths/
#   BS         evaluation batch size (default 8, as for the reported numbers)
#   SEEDS      model seeds to evaluate (default 1, the released checkpoints; the paper reports 1 2 3)
#
# Every cell is evaluated once; a cell already in cells.csv is skipped, so an interrupted run resumes.
set -u
CKPT_DIR=${CKPT_DIR:-./checkpoints}
DATA_DIR=${DATA_DIR:-./data}
OUT=${OUT:-./reproduce_out}
BS=${BS:-8}
SEEDS=${SEEDS:-1}
mkdir -p "$OUT/logs" "$OUT/scanpaths"
[ -f "$OUT/cells.csv" ] || echo "dataset,arm,seed,policy,T,metric,pxacc" > "$OUT/cells.csv"

# Training config of a model; ImageNet models are the end of phase 2.
config () { # $1=dataset $2=arm
  if [ "$1" = imagenet ]; then echo "configs/paper/imagenet_$2_phase2.yaml"; else echo "configs/paper/$1_$2.yaml"; fi
}

ckpt () { echo "$CKPT_DIR/$1_$2_s$3.pth"; } # $1=dataset $2=arm $3=seed

# Downloads a missing released checkpoint (seed 1) from GRO.data, see foavis/io/download.py.
need_ckpt () { # $1=dataset $2=arm $3=seed
  [ -f "$(ckpt "$1" "$2" "$3")" ] && return
  echo "fetch $1_$2_s$3: not in $CKPT_DIR, downloading the released checkpoint"
  python -m foavis.io.download --root "$CKPT_DIR" --models "$1_$2_s$3" > /dev/null 2>&1 \
    || { echo "FAIL  no checkpoint $(ckpt "$1" "$2" "$3") (only seed 1 is released)"; return 1; }
}

# cell <dataset> <arm> <seed> <policy> <T> [main.py args]
#   evaluates the model <dataset>_<arm>_s<seed> with fixation policy <policy> (a name in cells.csv) for T glimpses
cell () {
  local DS=$1 ARM=$2 S=$3 POL=$4 T=$5; shift 5
  local KEY="$DS,$ARM,$S,$POL,$T" LOG="$OUT/logs/${DS}_${ARM}_s${S}_${POL}_T${T}.txt"
  grep -q "^$KEY," "$OUT/cells.csv" && { echo "skip  $KEY (done)"; return; }
  need_ckpt "$DS" "$ARM" "$S" || return 1
  python main.py --config_path "$(config "$DS" "$ARM")" --data_dir "$DATA_DIR" --just_evaluate true \
    --pretrained_model_path "$(ckpt "$DS" "$ARM" "$S")" \
    --num_glimpses "$T" --batch_size "$BS" "$@" > "$LOG" 2>&1
  local RC=$?
  # The last evaluation pass is the reported one.
  local V PX=
  if [ "$DS" = cmnist ] || [ "$DS" = imagenet ]; then
    V=$(grep -oE "'val_accuracy': [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$")
  else
    V=$(grep -oE "'val_miou': [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$")
    PX=$(grep -oE "'val_pixel_accuracy': [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$")
  fi
  if [ "$RC" -ne 0 ] || [ -z "$V" ]; then echo "FAIL  $KEY (rc=$RC, see $LOG)"; return 1; fi
  echo "$KEY,$V,$PX" >> "$OUT/cells.csv"; echo "done  $KEY: $V $PX"
}

# transplant <reader dataset> <reader arm> <seed> <policy dataset>
#   records the scanpaths of the learned policy of <policy dataset> (same seed) on the reader's validation
#   images, then replays them into the reader. Cell policy name: <policy dataset>_policy.
transplant () {
  local DS=$1 ARM=$2 S=$3 PDS=$4 POL="$4_policy"
  grep -q "^$DS,$ARM,$S,$POL,4," "$OUT/cells.csv" && { echo "skip  $DS,$ARM,$S,$POL,4 (done)"; return; }
  local SPD="$OUT/scanpaths/${PDS}_learned_s${S}_on_${DS}" GLOG="$OUT/logs/gen_${PDS}_learned_s${S}_on_${DS}.txt"
  if [ ! -f "$SPD/scanpaths.parquet" ]; then
    need_ckpt "$PDS" learned "$S" || return 1
    # The reader dataset's learned config supplies images and resolution; the policy weights come from
    # <policy dataset>. All learned models share backbone, memory and policy head, so only the task head
    # (not needed to record fixations) fails to load.
    python main.py --config_path "$(config "$DS" learned)" --data_dir "$DATA_DIR" --just_evaluate true \
      --pretrained_model_path "$(ckpt "$PDS" learned "$S")" \
      --policy learned --num_glimpses 4 --batch_size "$BS" --save_scanpaths "$SPD" \
      > "$GLOG" 2>&1 || { echo "FAIL  scanpaths of $PDS policy on $DS (see $GLOG)"; return 1; }
    if grep -E "^Warning: .*parameters (missing|shape mismatch)" "$GLOG" | grep -qE "'(foveation|backbone|memory|policy_head)\."; then
      echo "FAIL  $PDS policy did not load into the $DS model (see $GLOG)"; rm -rf "$SPD"; return 1
    fi
  fi
  cell "$DS" "$ARM" "$S" "$POL" 4 --policy random --replay_scanpaths "$SPD/scanpaths.parquet"
}
