// lif_layer.sv
// Time-multiplexed LIF layer sharing one signed event datapath.
// TEMPLATE: currently instantiates N_HIDDEN neurons in parallel for simplicity.
// Replace with a time-multiplexed scheme (a few lanes + state BRAM) once the
// resource estimate says it is needed.
import pkg_snn::*;

module lif_layer #(
    parameter int N = N_HIDDEN,
    parameter bit ENABLE_NEGATIVE_SPIKES = 1'b0,
    parameter int THETA_LOW = -THETA
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     clear,
    input  logic                     ev_valid,
    input  logic [DT_W-1:0]          dt,
    input  logic                     ev_negative,
    input  logic signed [W_W-1:0]    w_row [0:N-1],   // weight per neuron for this input
    output logic                     busy,
    output logic                     event_done,
    output logic                     spk_valid [0:N-1],
    output logic                     spk_negative [0:N-1],
    output logic signed [POT_W-1:0]  pot       [0:N-1]
);

  localparam int INDEX_W = (N > 1) ? $clog2(N) : 1;
  localparam logic signed [POT_W-1:0] THETA_VALUE = POT_W'(THETA);
  localparam logic signed [POT_W-1:0] THETA_LOW_VALUE = POT_W'(THETA_LOW);

  logic signed [POT_W-1:0] pot_r [0:N-1];
  logic [INDEX_W-1:0] neuron_index;
  logic busy_r;
  logic [DT_W-1:0] dt_r;
  logic ev_negative_r;
  logic signed [POT_W-1:0] weight_extended;
  logic signed [POT_W-1:0] signed_weight;
  logic signed [POT_W-1:0] decayed;
  logic signed [POT_W-1:0] summed;
  logic fire_positive;
  logic fire_negative;

  assign busy = busy_r;
  assign weight_extended = POT_W'(w_row[neuron_index]);
  assign signed_weight = ev_negative_r ? -weight_extended : weight_extended;
  assign decayed = pot_r[neuron_index] >>> dt_r;
  assign summed = decayed + signed_weight;
  assign fire_positive = (summed >= THETA_VALUE);
  assign fire_negative = ENABLE_NEGATIVE_SPIKES && (summed <= THETA_LOW_VALUE);

  generate
    for (genvar i = 0; i < N; i++) begin : g_outputs
      assign pot[i] = pot_r[i];
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      neuron_index <= '0;
      busy_r       <= 1'b0;
      dt_r         <= '0;
      ev_negative_r <= 1'b0;
      event_done   <= 1'b0;
      for (int i = 0; i < N; i++) begin
        pot_r[i]       <= '0;
        spk_valid[i]   <= 1'b0;
        spk_negative[i] <= 1'b0;
      end
    end else if (clear) begin
      neuron_index <= '0;
      busy_r       <= 1'b0;
      dt_r         <= '0;
      ev_negative_r <= 1'b0;
      event_done   <= 1'b0;
      for (int i = 0; i < N; i++) begin
        pot_r[i]       <= '0;
        spk_valid[i]   <= 1'b0;
        spk_negative[i] <= 1'b0;
      end
    end else begin
      event_done <= 1'b0;
      for (int i = 0; i < N; i++) begin
        spk_valid[i]    <= 1'b0;
        spk_negative[i] <= 1'b0;
      end

      if (!busy && !event_done && ev_valid) begin
        dt_r          <= dt;
        ev_negative_r <= ev_negative;
        neuron_index  <= '0;
        busy_r        <= 1'b1;
      end else if (busy) begin
        spk_valid[neuron_index]    <= fire_positive || fire_negative;
        spk_negative[neuron_index] <= fire_negative;
        if (fire_positive)
          pot_r[neuron_index] <= summed - THETA_VALUE;
        else if (fire_negative)
          pot_r[neuron_index] <= summed - THETA_LOW_VALUE;
        else
          pot_r[neuron_index] <= summed;

        if (neuron_index == INDEX_W'(N - 1)) begin
          neuron_index <= '0;
          busy_r       <= 1'b0;
          event_done   <= 1'b1;
        end else begin
          neuron_index <= neuron_index + 1'b1;
        end
      end
    end
  end

endmodule
