// Parallel LIF layer used for the small output layer.
// One decay/add/threshold datapath per neuron, all driven from the same
// precomputed accumulator vector `acc_out` and elapsed time `elapsed_out`.
import pkg_snn::*;

module lif_layer_parallel #(
    parameter int N = N_OUTPUT,
    parameter bit ENABLE_NEGATIVE_SPIKES = 1'b0,
    parameter int THETA_LOW = -THETA
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     clear,
    input  logic                     ev_valid,   // pulse to start
    input  logic signed [POT_W-1:0]  acc_out[N], // precomputed accumulator
    input  logic [TIME_W-1:0]        elapsed_out, // time since last update
    output logic                     spk_valid [N],
    output logic                     spk_negative [N],
    output logic signed [POT_W-1:0]  pot [N]
);

  localparam int INDEX_W = (N > 1) ? $clog2(N) : 1;
  localparam logic signed [POT_W-1:0] THETA_VALUE = POT_W'(THETA);

  logic signed [POT_W-1:0] pot_r [N];
  logic [INDEX_W-1:0]      neuron_index;
  logic busy_r;
  logic ev_valid_r;
  logic event_done_r;

  assign event_done_r = 1'b0;

  generate
    for (genvar i = 0; i < N; i++) begin : g_spk
      assign spk_valid[i] = spk_valid_r[i];
      assign spk_negative[i] = spk_negative_r[i];
      assign pot[i] = pot_r[i];
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      neuron_index <= '0;
      busy_r       <= 1'b0;
      ev_valid_r   <= 1'b0;
      for (int i = 0; i < N; i++) begin
        pot_r[i]       <= '0;
        spk_valid_r[i]   <= 1'b0;
        spk_negative_r[i] <= 1'b0;
      end
    end else if (clear) begin
      neuron_index <= '0;
      busy_r       <= 1'b0;
      ev_valid_r   <= 1'b0;
      for (int i = 0; i < N; i++) begin
        pot_r[i]       <= '0;
        spk_valid_r[i]   <= 1'b0;
        spk_negative_r[i] <= 1'b0;
      end
    end else begin
      spk_valid_r  <= '0;
      spk_negative_r <= '0;

      if (!busy_r && ev_valid) begin
        ev_valid_r  <= 1'b0;
        neuron_index <= '0;
        busy_r       <= 1'b1;
      end else if (busy_r) begin
        logic signed [POT_W-1:0] decayed;
        logic signed [POT_W-1:0] summed;
        logic                    fire_positive;

        decayed = pot_r[neuron_index] >>> elapsed_out;
        summed  = decayed + acc_out[neuron_index];
        fire_positive = (summed >= THETA_VALUE);

        spk_valid_r[neuron_index] <= fire_positive;
        spk_negative_r[neuron_index] <= ENABLE_NEGATIVE_SPIKES && (summed <= THETA_LOW);

        if (fire_positive)
          pot_r[neuron_index] <= summed - THETA_VALUE;
        else
          pot_r[neuron_index] <= summed;

        if (neuron_index == INDEX_W'(N - 1)) begin
          neuron_index <= '0;
          busy_r       <= 1'b0;
        end else begin
          neuron_index <= neuron_index + 1'b1;
        end
      end
    end
  end

endmodule


