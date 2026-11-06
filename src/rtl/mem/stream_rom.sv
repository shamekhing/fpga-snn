// stream_rom.sv
// Encoded spike streams (delta-time events) and labels for test images.
// Initialized from files written by model/export_mem.py.
import pkg_snn::*;

module stream_rom #(
    parameter string INIT_FILE  = "data/streams/stream.mem",
    parameter int    DEPTH      = 65536
) (
    input  logic                clk,
    input  logic [15:0]         addr,
    output logic [DT_W-1:0]     data
);

  logic [DT_W-1:0] mem [DEPTH];

  initial $readmemh(INIT_FILE, mem);

  always_ff @(posedge clk) data <= mem[addr];

endmodule
