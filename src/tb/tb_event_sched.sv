`timescale 1ns/1ps

module tb_event_sched;

  localparam int N = 4;

  logic clk = 0;
  logic rst_n = 0;
  logic scan_req = 0;
  logic ev_ready = 0;
  logic [15:0] next_t [N];
  logic valid [N];
  logic negative [N];
  logic scan_busy;
  logic scan_done;
  logic ev_valid;
  logic [15:0] ev_time;
  logic [N-1:0] ev_mask;
  logic [N-1:0] ev_negative;
  integer cycles;

  event_sched #(.N(N)) dut (.*);

  always #5 clk = ~clk;

  task automatic check_scan(
      input logic expected_valid,
      input logic [15:0] expected_time,
      input logic [N-1:0] expected_mask,
      input logic [N-1:0] expected_negative,
      input string label
  );
    begin
      @(negedge clk);
      scan_req = 1;
      @(negedge clk);
      scan_req = 0;

      cycles = 0;
      while (!scan_done && cycles < N + 3) begin
        @(posedge clk);
        #1 cycles = cycles + 1;
      end
      if (!scan_done)
        $fatal(1, "%s: scan did not complete", label);
      if (ev_valid !== expected_valid
          || ev_time !== expected_time
          || ev_mask !== expected_mask
          || ev_negative !== expected_negative)
        $fatal(1, "%s: valid=%0b time=%0d mask=%b negative=%b",
               label, ev_valid, ev_time, ev_mask, ev_negative);

      if (expected_valid) begin
        repeat (2) begin
          @(posedge clk);
          #1;
          if (!ev_valid || ev_time !== expected_time
              || ev_mask !== expected_mask
              || ev_negative !== expected_negative)
            $fatal(1, "%s: event changed before ready", label);
        end
        @(negedge clk);
        ev_ready = 1;
        @(negedge clk);
        ev_ready = 0;
        #1;
        if (ev_valid || scan_busy)
          $fatal(1, "%s: event was not released after ready", label);
      end else if (scan_busy) begin
        $fatal(1, "%s: empty scan left scheduler busy", label);
      end
    end
  endtask

  initial begin
    for (int i = 0; i < N; i++) begin
      next_t[i] = '0;
      valid[i] = 1'b0;
      negative[i] = 1'b0;
    end

    repeat (2) @(posedge clk);
    #1;
    if (scan_busy || ev_valid || scan_done)
      $fatal(1, "reset state is not idle");
    @(negedge clk);
    rst_n = 1;

    valid[0] = 1;
    next_t[0] = 16'd9;
    negative[0] = 1;
    valid[1] = 1;
    next_t[1] = 16'd4;
    negative[1] = 0;
    valid[2] = 1;
    next_t[2] = 16'd4;
    negative[2] = 1;
    check_scan(1, 16'd4, 4'b0110, 4'b0100, "minimum with tied events");

    for (int i = 0; i < N; i++) valid[i] = 1'b0;
    check_scan(0, '0, '0, '0, "no pending events");

    valid[3] = 1;
    next_t[3] = 16'hffff;
    negative[3] = 0;
    check_scan(1, 16'hffff, 4'b1000, '0, "maximum representable time");

    $display("PASS: tb_event_sched (3 scans)");
    $finish;
  end

endmodule
