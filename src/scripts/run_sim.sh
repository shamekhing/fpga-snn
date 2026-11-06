#!/usr/bin/env bash
# run_sim.sh
# Compiles and runs each testbench with Icarus Verilog.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p ../ignore

SRC="rtl/util/pkg_snn.sv rtl/core/*.sv rtl/mem/*.sv rtl/util/argmax.sv rtl/top/*.sv"

echo "== tb_time_integrator =="
iverilog -g2012 -o "impl/tb_time_integrator.vvp" \
  rtl/util/pkg_snn.sv rtl/core/time_integrator.sv tb/tb_time_integrator.sv
vvp "impl/tb_time_integrator.vvp"

echo "== tb_event_sched =="
iverilog -g2012 -o "impl/tb_event_sched.vvp" \
  rtl/util/pkg_snn.sv rtl/core/event_sched.sv tb/tb_event_sched.sv
vvp "impl/tb_event_sched.vvp"

for tb in tb/tb_lif_neuron.sv tb/tb_snn_top.sv; do
  name=$(basename "$tb" .sv)
  echo "== $name =="
  iverilog -g2012 -o "impl/$name.vvp" $SRC "$tb"
  vvp "impl/$name.vvp"
done
