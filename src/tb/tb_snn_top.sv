// tb_snn_top.sv
// Golden-trace testbench. Runs one exported test image through snn_top and
// prints the same fields the Python reference writes to its golden file:
//   TRACE <time> <hidden_spikes> <output_spikes>   (per event time, in order)
//   DIGIT <d>
//   COUNTS <c0> ... <c9>
//   OPOT <p0> ... <p9>            (signed)
//   HPOT <h0> ... <h127>          (signed)
// A Python compare step (src/scripts/compare_golden.py) checks this output
// against the reference's golden file; any mismatch fails the run.
//
// File selection (runtime plusargs, consumed by the ROM models):
//   +STREAM_FILE=<path>  +BASE_FILE=<path>
// The weight ROMs keep their default paths (data/weights/*.mem), which the
// exporter regenerates from the checkpoint before the run.
`timescale 1ns/1ps
import pkg_snn::*;

module tb_snn_top;

  logic clk = 0;
  logic rst_n = 0;
  logic start = 0;
  logic done;
  logic [3:0] digit;

  snn_top dut (
    .clk   (clk),
    .rst_n (rst_n),
    .start (start),
    .done  (done),
    .digit (digit)
  );

  always #5 clk = ~clk;

  // Trace sampling on the falling edge: trace_* are registered and stable.
  always @(negedge clk) begin
    if (dut.trace_strobe)
      $display("TRACE %0d %0d %0d", dut.trace_time, dut.trace_hspk,
               dut.trace_ospk);
  end

  initial begin
`ifndef NO_VCD
    $dumpfile("ignore/tb_snn_top.vcd");
    $dumpvars(0, tb_snn_top);
`endif
    rst_n = 0;
    repeat (4) @(negedge clk);
    rst_n = 1;
    repeat (2) @(negedge clk);
    start = 1;
    @(negedge clk);
    start = 0;

    // Watchdog: far above a full-image run (~20k cycles).
    begin : watchdog
      integer cycles;
      cycles = 0;
      while (!done && cycles < 2000000) begin
        @(posedge clk);
        cycles = cycles + 1;
      end
      if (!done) begin
        $display("ERROR: timeout after %0d cycles without done", cycles);
        $finish;
      end
    end

    @(negedge clk);
    $write("DIGIT %0d\n", digit);
    $write("COUNTS");
    for (int k = 0; k < N_OUTPUT; k++)
      $write(" %0d", dut.u_counter.count[16*k +: 16]);
    $write("\n");
    $write("OPOT");
    for (int k = 0; k < N_OUTPUT; k++)
      $write(" %0d", $signed(dut.u_output.pot_r[k]));
    $write("\n");
    $write("HPOT");
    for (int n = 0; n < N_HIDDEN; n++)
      $write(" %0d", $signed(dut.u_hidden.pot_r[n]));
    $write("\n");
    $finish;
  end

endmodule
