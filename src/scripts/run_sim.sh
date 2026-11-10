#!/usr/bin/env bash
# run_sim.sh
# Compiles and runs every testbench with Icarus Verilog:
#   1. the unit benches (time integrator, event scheduler, LIF neuron)
#   2. the snn_top golden-trace tests (RTL vs Python reference, per image)
# Golden inputs come from data/golden/ (event ROM, base table, golden file);
# if they are missing they are regenerated from the checkpoint with
# src/model/export_mem.py (uses the THETA recorded in theta_report.txt when
# available, otherwise 16).
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p impl/temp

PKG="src/rtl/util/pkg_snn.sv"
CORES="src/rtl/core/time_integrator.sv src/rtl/core/event_sched.sv \
src/rtl/core/lif_layer.sv src/rtl/core/lif_layer_parallel.sv \
src/rtl/core/lif_neuron.sv src/rtl/core/spike_counter.sv"
MEMS="src/rtl/mem/stream_rom.sv src/rtl/mem/base_rom.sv src/rtl/mem/weight_rom.sv"
UTILS="src/rtl/util/argmax.sv"
TOP="src/rtl/top/snn_top.sv"

echo "== tb_time_integrator =="
iverilog -g2012 -o impl/temp/tb_time_integrator.vvp \
  "$PKG" src/rtl/core/time_integrator.sv src/tb/tb_time_integrator.sv
vvp impl/temp/tb_time_integrator.vvp

echo "== tb_event_sched =="
iverilog -g2012 -o impl/temp/tb_event_sched.vvp \
  "$PKG" src/rtl/core/event_sched.sv src/tb/tb_event_sched.sv
vvp impl/temp/tb_event_sched.vvp

echo "== tb_lif_neuron =="
iverilog -g2012 -o impl/temp/tb_lif_neuron.vvp \
  "$PKG" src/rtl/core/lif_neuron.sv src/tb/tb_lif_neuron.sv
vvp impl/temp/tb_lif_neuron.vvp

echo "== golden-trace tests (snn_top vs Python reference) =="
GOLDEN_DIR=data/golden
if ! ls "$GOLDEN_DIR"/*_golden.txt >/dev/null 2>&1; then
  echo "golden inputs missing; exporting with src/model/export_mem.py ..."
  THETA=16
  if [ -f "$GOLDEN_DIR/theta_report.txt" ]; then
    THETA=$(awk '/^THETA chosen/ {print $NF}' "$GOLDEN_DIR/theta_report.txt")
  fi
  echo "using THETA=$THETA"
  PYTHONPATH=src:. python3 src/model/export_mem.py --theta "$THETA"
fi

iverilog -g2012 -DNO_VCD -o impl/temp/tb_snn_top.vvp \
  "$PKG" $CORES $MEMS $UTILS "$TOP" src/tb/tb_snn_top.sv

status=0
for golden in "$GOLDEN_DIR"/*_golden.txt; do
  stem=$(basename "$golden" _golden.txt)
  echo "-- $stem --"
  vvp impl/temp/tb_snn_top.vvp \
      +STREAM_FILE="$GOLDEN_DIR/${stem}_stream.mem" \
      +BASE_FILE="$GOLDEN_DIR/${stem}_base.mem" \
      > "impl/temp/${stem}_rtl.txt"
  python3 src/scripts/compare_golden.py \
      "impl/temp/${stem}_rtl.txt" "$golden" || status=1
done

if [ "$status" -ne 0 ]; then
  echo "run_sim.sh: FAILURES detected"
  exit 1
fi
echo "run_sim.sh: all testbenches passed"
