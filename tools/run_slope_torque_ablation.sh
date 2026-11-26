#!/usr/bin/env bash
# Slope ablation for the torque input. Four estimator inputs, three seeds each,
# same frozen slope teacher, same DAgger schedule.
#
#   enc           joint encoders only (the JOSE baseline)
#   enc+act       encoders + last action (control arm, the command without the load)
#   enc+tau       encoders + motor-current proxy, default sensor noise
#   enc+tau-clean encoders + noiseless torque (upper bound on what torque can add)
#
# Usage:
#   TEACHER=logs/rsl_rl/isaac_g1_ppo_walk_slope_jose_v0/<run>/model_4999.pt \
#     bash tools/run_slope_torque_ablation.sh            # full run
#   SMOKE=1 TEACHER=... bash tools/run_slope_torque_ablation.sh   # pipeline check, minutes
#   TASK=Isaac-G1-PPO-Walk-SteepSlope-Estimator-JOSE-v0 PREFIX=steep SEEDS=42 TEACHER=... bash ...  # steep terrain
#
# Compare grid_tilt_error_deg_mean, grid_target_rmse[6:9] and the command
# tracking error in logs/jose_g1/estimators/slope_*/training.json.
set -euo pipefail
: "${TEACHER:?set TEACHER to the slope teacher checkpoint}"
TASK=${TASK:-Isaac-G1-PPO-Walk-Slope-Estimator-JOSE-v0}
PREFIX=${PREFIX:-slope}
SEEDS=${SEEDS:-"42 43 44"}
EXTRA=()
SUFFIX=""
if [[ "${SMOKE:-0}" == "1" ]]; then
  SEEDS=42
  SUFFIX=_smoke
  EXTRA=(--num-envs 16 --collect-steps 50 --epochs 1 --dagger-rounds 1 --dagger-epochs 1
         --eval-episodes 2 --max-episode-steps 50)
fi
declare -A ARMS=(
  [enc]=""
  [enc_act]="--action-input"
  [enc_tau]="--torque-input --torque-noise default"
  [enc_tau_clean]="--torque-input --torque-noise none"
)
for seed in $SEEDS; do
  for arm in enc enc_act enc_tau enc_tau_clean; do
    run="${PREFIX}_${arm}_seed${seed}${SUFFIX}"
    echo "=== $run ==="
    # shellcheck disable=SC2086
    python -m jose.train_state_estimator --adapter ppo_walk --task "$TASK" \
      --teacher-checkpoint "$TEACHER" --seed "$seed" --run-name "$run" --headless \
      ${ARMS[$arm]} ${EXTRA[@]+"${EXTRA[@]}"}
  done
done
