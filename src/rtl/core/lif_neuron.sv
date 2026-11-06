// Single signed-event LIF neuron with reset-to-mod thresholding.
import pkg_snn::*;

module lif_neuron #(
    parameter bit ENABLE_NEGATIVE_SPIKES = 1'b0,
    parameter int THETA_LOW = -THETA
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     clear,
    input  logic                     ev_valid,
    input  logic [DT_W-1:0]          dt,
    input  logic signed [POT_W-1:0]  synaptic_input,
    output logic                     spike_valid,
    output logic                     spike_negative,
    output logic signed [POT_W-1:0]  pot
);

  localparam logic signed [POT_W-1:0] THETA_VALUE = POT_W'(THETA);
  localparam logic signed [POT_W-1:0] THETA_LOW_VALUE = POT_W'(THETA_LOW);

  logic signed [POT_W-1:0] pot_r;
  logic signed [POT_W-1:0] decayed;
  logic signed [POT_W-1:0] summed;
  logic                    fire_positive;
  logic                    fire_negative;

  always_comb begin
    decayed       = pot_r >>> dt;
    summed        = decayed + synaptic_input;
    fire_positive = ev_valid && (summed >= THETA_VALUE);
    fire_negative = ev_valid && ENABLE_NEGATIVE_SPIKES
                    && (summed <= THETA_LOW_VALUE);
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pot_r         <= '0;
      spike_valid   <= 1'b0;
      spike_negative <= 1'b0;
    end else if (clear) begin
      pot_r         <= '0;
      spike_valid   <= 1'b0;
      spike_negative <= 1'b0;
    end else begin
      spike_valid    <= fire_positive || fire_negative;
      spike_negative <= fire_negative;
      if (ev_valid) begin
        if (fire_positive)
          pot_r <= summed - THETA_VALUE;
        else if (fire_negative)
          pot_r <= summed - THETA_LOW_VALUE;
        else
          pot_r <= summed;
      end
    end
  end

  assign pot = pot_r;

endmodule
