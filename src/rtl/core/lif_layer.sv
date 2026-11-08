// lif_layer.sv
// Time-multiplexed hidden LIF layer. One decay/add/threshold datapath
// shares 128 neuron state registers. On event acceptance it latches the
// event delta and polarity, then processes neuron indices 0..N-1
// sequentially; `event_done` pulses after the final neuron update.
import pkg_snn::*;

module lif_layer #(
    parameter int N = N_HIDDEN,
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
    output logic signed [POT_W-1:0]  pot [0:N-1],
    output logic                     busy,
    output logic                     event_done
);

  localparam int INDEX_W = (N > 1) ? $clog2(N) : 1;
  localparam logic signed [POT_W-1:0] THETA_VALUE = POT_W'(THETA);
  localparam logic signed [POT_W-1:0] THETA_LOW_VALUE = POT_W'(THETA_LOW);

  logic signed [POT_W-1:0] pot_r [0:N-1];
  logic [INDEX_W-1:0] neuron_index;
  logic busy_r;
  logic [DT_W-1:0] dt_r;
  logic ev_negative_r;

  // Registered one-cycle spike indications. These drive the flat result bus;
  // the plumbing that maps them onto the port lives in the generate block
  // below (constant indices), so Icarus can expand it.
  logic spk_valid_r [0:N-1];
  logic spk_negative_r [0:N-1];

  assign busy = busy_r;

  // Build the `neuron_out_t` result bank from the scalar registers at
  // constant (compile-time) indices. `neuron_out_t` is the typed single
  // neuron result; the layer copies it through a flat bus.
  generate
    for (genvar i = 0; i < N; i++) begin : g_spk_acc
      assign spk_valid[i] = spk_valid_r[i];
      assign spk_negative[i] = spk_negative_r[i];
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
        spk_valid_r[i]   <= 1'b0;
        spk_negative_r[i] <= 1'b0;
      end
    end else if (clear) begin
      neuron_index <= '0;
      busy_r       <= 1'b0;
      dt_r         <= '0;
      ev_negative_r <= 1'b0;
      event_done   <= 1'b0;
      for (int i = 0; i < N; i++) begin
        pot_r[i]       <= '0;
        spk_valid_r[i]   <= 1'b0;
        spk_negative_r[i] <= 1'b0;
      end
    end else begin
      event_done <= 1'b0;
      // One-cycle pulse: silence every lane until the in-flight neuron fires.
      for (int i = 0; i < N; i++) begin
        spk_valid_r[i] <= 1'b0;
        spk_negative_r[i] <= 1'b0;
      end

      if (!busy_r && !event_done && ev.valid) begin
        dt_r          <= ev.dt;
        ev_negative_r <= ev.negative;
        neuron_index  <= '0;
        busy_r        <= 1'b1;
      end else if (busy_r) begin
        logic signed [POT_W-1:0] weight_extended;
        logic signed [POT_W-1:0] signed_weight;
        logic signed [POT_W-1:0] decayed;
        logic signed [POT_W-1:0] summed;
        logic                    fire_positive;
        logic                    fire_negative;

        weight_extended = POT_W'(w_row[neuron_index]);
        signed_weight = ev_negative_r ? -weight_extended : weight_extended;
        decayed = pot_r[neuron_index] >>> dt_r;
        summed = decayed + signed_weight;
        fire_positive = (summed >= THETA_VALUE);
        fire_negative = ENABLE_NEGATIVE_SPIKES && (summed <= THETA_LOW_VALUE);

        spk_valid_r[neuron_index] <= fire_positive || fire_negative;
        spk_negative_r[neuron_index] <= fire_negative;

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
