#!/usr/bin/env bash
# run_sim.sh
# Compiles and runs each testbench with Icarus Verilog.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p impl/temp

SRC="src/rtl/util/pkg_snn.sv src/rtl/core/*.sv src/rtl/mem/*.sv src/rtl/util/argmax.sv src/rtl/top/*.sv"

echo "== tb_time_integrator =="
iverilog -g2012 -o "impl/temp/tb_time_integrator.vvp" \
  src/rtl/util/pkg_snn.sv src/rtl/core/time_integrator.sv src/tb/tb_time_integrator.sv
vvp "impl/temp/tb_time_integrator.vvp"

echo "== tb_event_sched =="
iverilog -g2012 -o "impl/temp/tb_event_sched.vvp" \
  src/rtl/util/pkg_snn.sv src/rtl/core/event_sched.sv src/tb/tb_event_sched.sv
vvp "impl/temp/tb_event_sched.vvp"

for tb in src/tb/tb_lif_neuron.sv src/tb/tb_snn_top.sv; do
  name=$(basename "$tb" .sv)
  echo "== $name =="
  iverilog -g2012 -o "impl/temp/$name.vvp" $SRC "$tb"
  vvp "impl/temp/$name.vvp"
done
