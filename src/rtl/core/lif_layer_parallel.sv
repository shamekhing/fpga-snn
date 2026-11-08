// Parallel LIF layer used for the small output layer.
// One `lif_neuron` per output neuron, all driven from the same accepted event.
import pkg_snn::*;

module lif_layer_parallel #(
    parameter int N = N_OUTPUT,
    parameter bit ENABLE_NEGATIVE_SPIKES = 1'b0,
    parameter int THETA_LOW = -THETA
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     clear,
    input  event_in_t                ev,
    input  logic signed [W_W-1:0]    w_row [0:N-1],
    output logic                     spk_valid [0:N-1],
    output logic                     spk_negative [0:N-1],
    output logic signed [POT_W-1:0]  pot [0:N-1]
);

  // One `lif_neuron` per output neuron. Each core is a clean source of
  // `neuron_out_t`; the generate block exposes its fields on the flat result
  // bus. `neuron_out_t` is kept as a single-neuron bundle.
  generate
    for (genvar i = 0; i < N; i++) begin : g_neuron
      logic signed [POT_W-1:0] signed_weight;
      neuron_out_t spk_n;

      assign signed_weight = ev.negative ? -w_row[i] : w_row[i];

      lif_neuron #(
        .ENABLE_NEGATIVE_SPIKES (ENABLE_NEGATIVE_SPIKES),
        .THETA_LOW (THETA_LOW)
      ) u_neuron (
        .clk             (clk),
        .rst_n           (rst_n),
        .clear           (clear),
        .ev              (ev),
        .synaptic_input  (signed_weight),
        .spk             (spk_n)
      );

      assign spk_valid[i] = spk_n.spk_valid;
      assign spk_negative[i] = spk_n.spk_negative;
      assign pot[i] = spk_n.pot;
    end
  endgenerate

endmodule
