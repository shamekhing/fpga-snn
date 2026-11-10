// lif_layer.sv
// Time-multiplexed hidden LIF layer. One decay/add/threshold datapath
// shares N neuron state registers. `acc` is a precomputed accumulator vector
// (all events at one event time are summed into it by the row sequencer),
// and `elapsed` is the time since the last hidden update. On `ev_valid`, the
// layer starts processing neurons 0..N-1 sequentially; `event_done` pulses
// after the final neuron update.
import pkg_snn::*;

module lif_layer #(
    parameter int N = N_HIDDEN,
    parameter bit ENABLE_NEGATIVE_SPIKES = 1'b0,
    parameter int THETA_LOW = -THETA
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     clear,
    input  logic                     ev_valid,   // pulse to start
    input  logic signed [POT_W-1:0]  acc [N],    // precomputed accumulator
    input  logic [TIME_W-1:0]        elapsed,    // time since last update
    output logic                     spk_valid [N],
    output logic                     spk_negative [N],
    output logic signed [POT_W-1:0]  pot [N],
    output logic                     event_done
);

  localparam int INDEX_W = (N > 1) ? $clog2(N) : 1;
  localparam logic signed [POT_W-1:0] THETA_VALUE = POT_W'(THETA);

  logic signed [POT_W-1:0] pot_r [N];
  logic [INDEX_W-1:0]      neuron_index;
  logic busy_r;
  logic ev_valid_r;
  logic event_done_r;

  assign event_done = event_done_r;

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
      event_done_r <= 1'b0;
      for (int i = 0; i < N; i++) begin
        pot_r[i]       <= '0;
        spk_valid_r[i]   <= 1'b0;
        spk_negative_r[i] <= 1'b0;
      end
    end else if (clear) begin
      neuron_index <= '0;
      busy_r       <= 1'b0;
      ev_valid_r   <= 1'b0;
      event_done_r <= 1'b0;
      for (int i = 0; i < N; i++) begin
        pot_r[i]       <= '0;
        spk_valid_r[i]   <= 1'b0;
        spk_negative_r[i] <= 1'b0;
      end
    end else begin
      event_done_r <= 1'b0;
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

        decayed = pot_r[neuron_index] >>> elapsed;
        summed  = decayed + acc[neuron_index];
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
          event_done_r <= 1'b1;
        end else begin
          neuron_index <= neuron_index + 1'b1;
        end
      end
    end
  end

  // One-cycle pulse: acc is valid exactly during the cycle in which event_done
  // is asserted (i.e., while the layer is processing the in-flight event).
  logic acc_valid_r;
  assign acc_valid = acc_valid_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc_valid_r <= 1'b0;
    end else if (clear) begin
      acc_valid_r <= 1'b0;
    end else begin
      acc_valid_r <= ev_valid;
    end
  end

endmodule

