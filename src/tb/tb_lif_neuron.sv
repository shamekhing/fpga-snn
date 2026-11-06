`timescale 1ns/1ps
import pkg_snn::*;

module tb_lif_neuron;

  logic clk = 0;
  logic rst_n = 0;
  logic clear = 0;
  logic ev_valid = 0;
  logic [DT_W-1:0] dt = '0;
  logic signed [POT_W-1:0] synaptic_input = '0;

  logic spike_valid_positive;
  logic spike_negative_positive;
  logic signed [POT_W-1:0] pot_positive;
  logic spike_valid_bidirectional;
  logic spike_negative_bidirectional;
  logic signed [POT_W-1:0] pot_bidirectional;

  lif_neuron dut_positive (
    .clk,
    .rst_n,
    .clear,
    .ev_valid,
    .dt,
    .synaptic_input,
    .spike_valid   (spike_valid_positive),
    .spike_negative(spike_negative_positive),
    .pot           (pot_positive)
  );

  lif_neuron #(.ENABLE_NEGATIVE_SPIKES(1'b1)) dut_bidirectional (
    .clk,
    .rst_n,
    .clear,
    .ev_valid,
    .dt,
    .synaptic_input,
    .spike_valid   (spike_valid_bidirectional),
    .spike_negative(spike_negative_bidirectional),
    .pot           (pot_bidirectional)
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
      if (pot_positive !== expected_positive_pot
          || spike_valid_positive !== expected_positive_spike
          || spike_negative_positive !== expected_positive_sign
          || pot_bidirectional !== expected_bidirectional_pot
          || spike_valid_bidirectional !== expected_bidirectional_spike
          || spike_negative_bidirectional !== expected_bidirectional_sign)
        $fatal(1, "%s: positive=(%0d,%0b,%0b) bidirectional=(%0d,%0b,%0b)",
               label,
               pot_positive, spike_valid_positive, spike_negative_positive,
               pot_bidirectional, spike_valid_bidirectional,
               spike_negative_bidirectional);
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
      dt = event_dt;
      synaptic_input = event_input;
      ev_valid = 1;
      @(posedge clk);
      #1 check_state(expected_positive_pot, expected_positive_spike,
                     expected_positive_sign, expected_bidirectional_pot,
                     expected_bidirectional_spike,
                     expected_bidirectional_sign, label);
      @(negedge clk);
      ev_valid = 0;
    end
  endtask

  initial begin
    @(posedge clk);
    #1 check_state('0, 1'b0, 1'b0, '0, 1'b0, 1'b0, "reset");

    @(negedge clk);
    rst_n = 1;

    apply_event(0, 20, 4, 1, 0, 4, 1, 0, "positive threshold");
    apply_event(1, 13, 15, 0, 0, 15, 0, 0, "decay before event");
    apply_event(0, 1, 0, 1, 0, 0, 1, 0, "threshold after decay");
    apply_event(0, -16, -16, 0, 0, 0, 1, 1, "negative threshold option");

    @(negedge clk);
    clear = 1;
    ev_valid = 1;
    synaptic_input = 20;
    @(posedge clk);
    #1 check_state('0, 1'b0, 1'b0, '0, 1'b0, 1'b0, "clear priority");
    @(negedge clk);
    clear = 0;
    ev_valid = 0;

    apply_event(0, 12, 12, 0, 0, 12, 0, 0, "subthreshold input");
    @(negedge clk);
    dt = DT_W'(OVERFLOW_CODE);
    ev_valid = 0;
    @(posedge clk);
    #1 check_state(12, 0, 0, 12, 0, 0, "overflow does not update potential");
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
