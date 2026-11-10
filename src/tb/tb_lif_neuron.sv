`timescale 1ns/1ps
import pkg_snn::*;

module tb_lif_neuron;

  // Shims for the bundled event (event_in_t) and result (neuron_out_t).
  logic clk = 0;
  logic rst_n = 0;
  logic clear = 0;
  event_in_t ev_in;
  neuron_out_t spk_positive;
  neuron_out_t spk_bidirectional;
  logic signed [POT_W-1:0] synaptic_input = '0;

  lif_neuron dut_positive (
    .clk,
    .rst_n,
    .clear,
    .ev            (ev_in),
    .synaptic_input (synaptic_input),
    .spk           (spk_positive)
  );

  lif_neuron #(.ENABLE_NEGATIVE_SPIKES(1'b1)) dut_bidirectional (
    .clk,
    .rst_n,
    .clear,
    .ev            (ev_in),
    .synaptic_input (synaptic_input),
    .spk           (spk_bidirectional)
  );

  always #5 clk = ~clk;

  task automatic check_state(
      input logic signed [POT_W-1:0] expected_positive_pot,
      input logic expected_positive_spike,
      input logic expected_positive_sign,
      input logic signed [POT_W-1:0] expected_bidirectional_pot,
      input logic expected_bidirectional_spike,
      input logic expected_bidirectional_sign,
      input string label
  );
    begin
      if (spk_positive.pot !== expected_positive_pot
          || spk_positive.spk_valid !== expected_positive_spike
          || spk_positive.spk_negative !== expected_positive_sign
          || spk_bidirectional.pot !== expected_bidirectional_pot
          || spk_bidirectional.spk_valid !== expected_bidirectional_spike
          || spk_bidirectional.spk_negative !== expected_bidirectional_sign)
        $fatal(1, "%s: positive=(%0d,%0b,%0b) bidirectional=(%0d,%0b,%0b)",
               label,
               spk_positive.pot, spk_positive.spk_valid, spk_positive.spk_negative,
               spk_bidirectional.pot, spk_bidirectional.spk_valid,
               spk_bidirectional.spk_negative);
    end
  endtask

  task automatic apply_event(
      input logic [DT_W-1:0] event_dt,
      input logic signed [POT_W-1:0] event_input,
      input logic signed [POT_W-1:0] expected_positive_pot,
      input logic expected_positive_spike,
      input logic expected_positive_sign,
      input logic signed [POT_W-1:0] expected_bidirectional_pot,
      input logic expected_bidirectional_spike,
      input logic expected_bidirectional_sign,
      input string label
  );
    begin
      @(negedge clk);
      ev_in.dt = event_dt;
      ev_in.valid = 1;
      ev_in.negative = 0;
      synaptic_input = event_input;
      @(posedge clk);
      #1 check_state(expected_positive_pot, expected_positive_spike,
                     expected_positive_sign, expected_bidirectional_pot,
                     expected_bidirectional_spike, expected_bidirectional_sign,
                     label);
      @(negedge clk);
      ev_in.valid = 0;
      ev_in.negative = 0;
    end
  endtask

  initial begin
    @(posedge clk);
    #1 check_state('0, 1'b0, 1'b0, '0, 1'b0, 1'b0, "reset");

    @(negedge clk);
    rst_n = 1;

    apply_event(0, THETA + 4, 4, 1, 0, 4, 1, 0, "positive threshold");
    apply_event(1, (THETA - 1) - 2, THETA - 1, 0, 0, THETA - 1, 0, 0,
                "decay before event");
    apply_event(0, 1, 0, 1, 0, 0, 1, 0, "threshold after decay");
    apply_event(0, -THETA, -THETA, 0, 0, 0, 1, 1, "negative threshold option");

    @(negedge clk);
    clear = 1;
    ev_in.valid = 1;
    ev_in.negative = 0;
    synaptic_input = 20;
    @(posedge clk);
    #1 check_state('0, 1'b0, 1'b0, '0, 1'b0, 1'b0, "clear priority");
    @(negedge clk);
    clear = 0;
    ev_in.valid = 0;

    apply_event(0, THETA - 4, THETA - 4, 0, 0, THETA - 4, 0, 0,
                "subthreshold input");
    @(negedge clk);
    ev_in.dt = DT_W'(OVERFLOW_CODE);
    ev_in.valid = 0;
    @(posedge clk);
    #1 check_state(THETA - 4, 0, 0, THETA - 4, 0, 0,
                   "overflow does not update potential");
    apply_event(DT_W'(OVERFLOW_CODE), 0, 0, 0, 0, 0, 0, 0,
                "elapsed overflow time decays on next event");

    @(negedge clk);
    rst_n = 0;
    #1 check_state('0, 1'b0, 1'b0, '0, 1'b0, 1'b0,
                   "asynchronous reset");

    $display("PASS: tb_lif_neuron (8 event checks)");
    $finish;
  end

endmodule
