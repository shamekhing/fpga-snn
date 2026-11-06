// Parallel LIF layer used for the small output layer.
import pkg_snn::*;

module lif_layer_parallel #(
    parameter int N = N_OUTPUT,
    parameter bit ENABLE_NEGATIVE_SPIKES = 1'b0,
    parameter int THETA_LOW = -THETA
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     clear,
    input  logic                     ev_valid,
    input  logic [DT_W-1:0]          dt,
    input  logic                     ev_negative,
    input  logic signed [W_W-1:0]    w_row [0:N-1],
    output logic                     spk_valid [0:N-1],
    output logic                     spk_negative [0:N-1],
    output logic signed [POT_W-1:0]  pot [0:N-1]
);

  generate
    for (genvar i = 0; i < N; i++) begin : g_neuron
      logic signed [POT_W-1:0] weight_extended;
      logic signed [POT_W-1:0] signed_weight;

      assign weight_extended = POT_W'(w_row[i]);
      assign signed_weight = ev_negative ? -weight_extended : weight_extended;

      lif_neuron #(
        .ENABLE_NEGATIVE_SPIKES (ENABLE_NEGATIVE_SPIKES),
        .THETA_LOW              (THETA_LOW)
      ) u_neuron (
        .clk             (clk),
        .rst_n           (rst_n),
        .clear           (clear),
        .ev_valid        (ev_valid),
        .dt              (dt),
        .synaptic_input  (signed_weight),
        .spike_valid     (spk_valid[i]),
        .spike_negative  (spk_negative[i]),
        .pot             (pot[i])
      );
    end
  endgenerate

endmodule
