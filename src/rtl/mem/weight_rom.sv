// weight_rom.sv
// Hidden-layer weights, one row per input pixel, one column per hidden neuron.
// Initialized from a hex file exported by model/export_mem.py.
import pkg_snn::*;

module weight_rom #(
    parameter string INIT_FILE = "data/weights/hidden_w.mem",
    parameter int    N_NEURONS = N_HIDDEN,
    parameter int    DEPTH     = N_INPUT,
    parameter int    ADDR_W    = 10
) (
    input  logic                  clk,
    input  logic [ADDR_W-1:0]     addr,
    output logic [N_NEURONS*W_W-1:0] data
);

  logic [N_NEURONS*W_W-1:0] mem [DEPTH];

  initial $readmemh(INIT_FILE, mem);

  always_ff @(posedge clk) data <= mem[addr];

endmodule
