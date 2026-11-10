// lif_layer_parallel.sv
// Parallel output LIF layer: one decay/add/threshold datapath per output
// neuron, all updated in the same cycle. `acc_out` is the precomputed output
// accumulator (contributions of every hidden neuron that spiked at this event
// time, accumulated by the top-level sequencer); `elapsed_out` is the time
// since the last output update. Output neurons use only positive spikes.
import pkg_snn::*;

module lif_layer_parallel #(
    parameter int N = N_OUTPUT,
    parameter bit ENABLE_NEGATIVE_SPIKES = 1'b0,
    parameter int THETA_LOW = -THETA
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     clear,
    input  logic                     ev_valid,       // one-cycle start pulse
    input  logic signed [POT_W-1:0]  acc_out [N],    // precomputed accumulator
    input  logic [TIME_W-1:0]        elapsed_out,    // time since last update
    output logic [N-1:0]             spk_valid,      // one-cycle spike pulse
    output logic [N-1:0]             spk_negative,
    output logic signed [POT_W-1:0]  pot [N],
    output logic                     event_done      // one-cycle pulse
);

  localparam logic signed [POT_W-1:0] THETA_VALUE = POT_W'(THETA);
  localparam logic signed [POT_W-1:0] THETA_LOW_VALUE = POT_W'(THETA_LOW);

  logic signed [POT_W-1:0] pot_r [N];
  logic [N-1:0]            spk_valid_r;
  logic [N-1:0]            spk_negative_r;
  logic                    event_done_r;

  // Shifts of POT_W or more saturate to the sign bit: clamp the shift amount.
  logic [4:0] shift_amt;
  assign shift_amt = (|elapsed_out[TIME_W-1:5]) ? 5'd31 : elapsed_out[4:0];

  assign event_done = event_done_r;

  generate
    for (genvar k = 0; k < N; k++) begin : g_neuron
      logic signed [POT_W-1:0] decayed;
      logic signed [POT_W-1:0] summed;
      logic                    fire_positive;
      logic                    fire_negative;

      always_comb begin
        decayed       = pot_r[k] >>> shift_amt;
        summed        = decayed + acc_out[k];
        fire_positive = ev_valid && (summed >= THETA_VALUE);
        fire_negative = ev_valid && ENABLE_NEGATIVE_SPIKES
                        && (summed <= THETA_LOW_VALUE);
      end

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          pot_r[k]          <= '0;
          spk_valid_r[k]    <= 1'b0;
          spk_negative_r[k] <= 1'b0;
        end else if (clear) begin
          pot_r[k]          <= '0;
          spk_valid_r[k]    <= 1'b0;
          spk_negative_r[k] <= 1'b0;
        end else begin
          spk_valid_r[k]    <= fire_positive;
          spk_negative_r[k] <= fire_negative;
          if (fire_positive)
            pot_r[k] <= summed - THETA_VALUE;
          else if (fire_negative)
            pot_r[k] <= summed - THETA_LOW_VALUE;
          else if (ev_valid)
            pot_r[k] <= summed;
        end
      end
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)       event_done_r <= 1'b0;
    else if (clear)   event_done_r <= 1'b0;
    else              event_done_r <= ev_valid;
  end

  assign spk_valid    = spk_valid_r;
  assign spk_negative = spk_negative_r;

endmodule
