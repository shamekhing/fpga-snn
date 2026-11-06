# build.tcl
# Gowin EDA build flow for Tang Primer 20K.
# TEMPLATE: fill in the project settings from your Gowin EDA project.
set device "GW2A-LV18PG256C8/I7"
set top_module "snn_top"
set sources [glob -nocomplain rtl/util/pkg_snn.sv rtl/core/*.sv rtl/mem/*.sv rtl/util/argmax.sv rtl/top/*.sv]
set constraints constraints/tang_primer_20k.cst
# TODO: create project, add sources and constraints, run synthesis, PnR, bitstream to impl/
