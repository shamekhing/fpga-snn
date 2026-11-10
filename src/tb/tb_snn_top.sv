// tb_snn_top.sv
// Full-image testbench. Loads one golden image and compares spike traces with Python.
// TEMPLATE: skeleton only.
`timescale 1ns/1ps
import pkg_snn::*;

module tb_snn_top;

  logic clk = 0, rst_n = 0, start = 0;
  logic done;
  logic [3:0] digit;

  snn_top dut (.*);

  always #5 clk = ~clk;

  initial begin
    $dumpfile("ignore/tb_snn_top.vcd");
    $dumpvars(0, tb_snn_top);
    #20 rst_n = 1;
    @(posedge clk); start = 1;
    @(posedge clk); start = 0;
    wait (done);
    $display("predicted digit = %0d", digit);
    $finish;
  end

endmodule
