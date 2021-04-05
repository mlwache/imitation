#!/usr/bin/env bash
set -e

# This script trains experts for experiments/imit_benchmark.sh.
# When training is finished, it reports the mean episode reward of each
# expert.

ENVS+="pendulum cartpole mountain_car "
# ENVS+="reacher half_cheetah hopper ant humanoid swimmer walker "
# ENVS+="half_cheetah ant humanoid"
# ENVS+="two_d_maze custom_ant disabled_ant "

SEEDS="0 1 2"

EXPERT_MODELS_DIR="data/expert_models"
TIMESTAMP=$(date --iso-8601=seconds)
OUTPUT_DIR="output/train_bc/${TIMESTAMP}"
RESULTS_FILE="results.txt"
extra_configs=""

TEMP=$(getopt -o frT -l fast,regenerate,tmux,run_name:,mvp_fast,mvp,mvp_seals -- "$@")
if [[ $? != 0 ]]; then exit 1; fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    -f | --fast)
      # Fast mode (debug)
      ENVS="cartpole pendulum"
      SEEDS="0"
      extra_configs+="fast "
      EXPERT_MODELS_DIR="tests/data/expert_models"
      shift
      ;;
    --mvp)  # Starting to look what I actually want is a Sacred named_config XD
      # I could actually spend some time doing this today (or designing it if I want)
      # Will depend on my other priorities though
      ENVS="cartpole mountain_car half_cheetah "
      shift
      ;;
    --mvp_seals)  # Starting to look what I actually want is a Sacred named_config XD
      # I could actually spend some time doing this today (or designing it if I want)
      # Will depend on my other priorities though
      ENVS="seals_cartpole seals_mountain_car half_cheetah "
      shift
      ;;
    --mvp_fast)
      ENVS="cartpole mountain_car half_cheetah "
      SEEDS="0"
      extra_configs+="fast "
      shift
      ;;
    --run_name)
      extra_options+="--name $2 "
      shift 2
      ;;
    -T | --tmux)
      extra_parallel_options+="--tmux "
      shift
    ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unrecognized flag $1" >&2
      exit 1
      ;;
  esac
done

echo "Writing logs in ${OUTPUT_DIR}"

parallel -j 25% --header : --results ${OUTPUT_DIR}/parallel/ --colsep , --progress \
  ${extra_parallel_options} \
  python -m imitation.scripts.train_bc \
  --capture=sys \
  ${extra_options} \
  with \
  ${extra_configs} \
  {env_cfg_name} \
  expert_data_src=${EXPERT_MODELS_DIR}/{env_cfg_name}_0/rollouts/final.pkl \
  expert_data_src_format="path" \
  seed={seed} \
  log_root=${OUTPUT_DIR} \
  ::: env_cfg_name ${ENVS} \
  ::: seed ${SEEDS}

pushd $OUTPUT_DIR

popd
