#!/usr/bin/env bash
# Prints how run_one.sh parses each method name (q, d, warmup, zero-init, base row). No training.
#   bash experiments/tests/parse_method_names.sh [method ...]
RUN_ONE="$(cd "$(dirname "$0")/../osc" && pwd)/run_one.sh"
METHOD_BLOCK=$(sed -n '/^# ---------------- method overrides/,/^OV+=(/p' "$RUN_ONE")
METHODS=("$@"); [ ${#METHODS[@]} -eq 0 ] && METHODS=(softmax entmax asentmax asentmax_zi_w20k stieltjes stieltjes_q16 wstieltjes wstieltjes_d4 aswstieltjes_q2 aswstieltjes_q2_d1 aswstieltjes_q2_d4_w20k asstieltjes_q8)
log() { :; }
for method in "${METHODS[@]}"; do
  eval "$METHOD_BLOCK" || { echo "$method: PARSE FAILED"; continue; }
  printf "%-26s " "$method"
  printf "%s\n" "${OV[@]}" | grep -oE "stieltjes_q=[0-9.]+|stieltjes_window=[0-9.]+|attn_scale_type=[a-z-]+|attn_scale_zero_init=[0-9.]+|num_warmup_steps=[0-9]+|entmax_alpha=[0-9.]+" | tr '\n' ' '; echo
done
