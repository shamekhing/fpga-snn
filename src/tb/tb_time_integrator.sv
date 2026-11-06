`timescale 1ns/1ps
import pkg_snn::*;

module tb_time_integrator;

  logic clk = 0;
  logic rst_n = 0;
  logic clear = 0;
  logic step = 0;
  logic [DT_W-1:0] delta = '0;
  logic [TIME_W-1:0] abs_time;
  logic is_overflow;

  integer checks = 0;

  time_integrator dut (.*);

  always #5 clk = ~clk;

  task automatic check_time(
      input logic [TIME_W-1:0] expected,
      input string label
  );
    begin
      checks = checks + 1;
      if (abs_time !== expected)
        $fatal(1, "%s: expected abs_time=%0d, got %0d",
               label, expected, abs_time);
    end
  endtask

  task automatic check_overflow(
      input logic expected,
      input string label
  );
    begin
      checks = checks + 1;
      if (is_overflow !== expected)
        $fatal(1, "%s: expected is_overflow=%0b, got %0b",
               label, expected, is_overflow);
    end
  endtask

  initial begin
    @(posedge clk);
    #1;
    check_time('0, "initial reset");
    check_overflow(1'b0, "ordinary delta");

    @(negedge clk);
    rst_n = 1;
    step = 1;
    delta = 8'd5;
    @(posedge clk);
    #1 check_time(16'd5, "first step");

    @(negedge clk);
    step = 0;
    delta = 8'd7;
    @(posedge clk);
    #1 check_time(16'd5, "step disabled");

    @(negedge clk);
    step = 1;
    delta = 8'd2;
    @(posedge clk);
    #1 check_time(16'd7, "accumulated step");

    @(negedge clk);
    delta = 8'hff;
    check_overflow(1'b1, "overflow delta");
    @(posedge clk);
    #1 check_time(16'd262, "overflow advances time");

    @(negedge clk);
    step = 0;
    check_overflow(1'b1, "overflow indication while stalled");
    @(posedge clk);
    #1 check_time(16'd262, "overflow does not advance while stalled");

    @(negedge clk);
    clear = 1;
    step = 1;
    delta = 8'd4;
    @(posedge clk);
    #1 check_time('0, "clear takes priority over step");

    @(negedge clk);
    clear = 0;
    delta = 8'd1;
    @(posedge clk);
    #1 check_time(16'd1, "step after clear");

    @(negedge clk);
    rst_n = 0;
    step = 0;
    #1 check_time('0, "asynchronous reset");

    @(negedge clk);
    rst_n = 1;
    step = 1;
    delta = 8'd3;
    @(posedge clk);
    #1 check_time(16'd3, "step after reset");

    $display("PASS: tb_time_integrator (%0d checks)", checks);
    $finish;
  end

endmodule