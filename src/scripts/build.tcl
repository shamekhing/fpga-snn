# build.tcl
# Gowin EDA build flow for Tang Primer 20K.
# Adapt paths and pin names from tang_primer_20k.cst to your board.
set device "GW2A-LV18PG256C8/I7"
set top_module "snn_top"
set sources [glob -nocomplain src/rtl/util/pkg_snn.sv src/rtl/core/*.sv src/rtl/mem/*.sv src/rtl/util/argmax.sv src/rtl/top/*.sv]
set constraints src/constraints/tang_primer_20k.cst

# Create project, add sources and constraints, run synthesis, PnR, bitstream.
# Uncomment and adapt the gw_sh commands below once the CST pin assignments
# are complete:

# create_project -device $device -top $top_module -work_dir impl
# add_file -type verilog $sources
# add_file -type constraint $constraints
# run synthesis
# run place_and_route
# generate_bitstream -o impl/$top_module.fs

# TODO: Complete CST pin assignments (reset, buttons, LEDs) before uncommenting.
# TODO: Run Gowin EDA (gw_ide or gw_sh) with this script to produce impl/$top_module.fs
