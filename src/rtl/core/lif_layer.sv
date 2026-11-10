// lif_layer.sv
// Time-multiplexed hidden LIF layer. One decay/add/threshold datapath shares
// N neuron state registers. `acc` is the precomputed accumulator vector: all
// events of all row streams at one event time were summed into it by the
// top-level sequencer before this update starts. `elapsed` is the time since
// the last hidden update.
//
// On `ev_valid` the layer processes neurons 0..N-1 sequentially, one per
// clock. Every neuron is updated at every event time, including neurons with
// zero accumulated input. The spike bank accumulates across the update and,
// together with the one-cycle `event_done` pulse, presents the complete spike
// vector for this event time.
import pkg_snn::*;

module lif_layer #(
    parameter int N = N_HIDDEN,
    parameter bit ENABLE_NEGATIVE_SPIKES = 1'b0,
    parameter int THETA_LOW = -THETA
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     clear,
    input  logic                     ev_valid,      // one-cycle start pulse
    input  logic signed [POT_W-1:0]  acc [N],       // precomputed accumulator
    input  logic [TIME_W-1:0]        elapsed,       // time since last update
    output logic [N-1:0]             spk_valid,     // spike bank (valid with event_done)
    output logic [N-1:0]             spk_negative,  // negative-threshold bank
    output logic signed [POT_W-1:0]  pot [N],
    output logic                     event_done     // one-cycle pulse
);

  localparam int INDEX_W = (N > 1) ? $clog2(N) : 1;
  localparam logic signed [POT_W-1:0] THETA_VALUE = POT_W'(THETA);
  localparam logic signed [POT_W-1:0] THETA_LOW_VALUE = POT_W'(THETA_LOW);

  logic signed [POT_W-1:0] pot_r [N];
  logic [INDEX_W-1:0]      neuron_index;
  logic                    busy_r;
  logic                    event_done_r;
  logic [N-1:0]            spk_valid_r;
  logic [N-1:0]            spk_negative_r;

  // Shifts of POT_W or more saturate to the sign bit: clamp the shift amount.
  logic [4:0] shift_amt;
  assign shift_amt = (|elapsed[TIME_W-1:5]) ? 5'd31 : elapsed[4:0];

  assign event_done = event_done_r;

  generate
    for (genvar i = 0; i < N; i++) begin : g_result
      assign pot[i] = pot_r[i];
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      neuron_index  <= '0;
      busy_r        <= 1'b0;
      event_done_r  <= 1'b0;
      spk_valid_r   <= '0;
      spk_negative_r<= '0;
      for (int i = 0; i < N; i++) pot_r[i] <= '0;
    end else if (clear) begin
      neuron_index  <= '0;
      busy_r        <= 1'b0;
      event_done_r  <= 1'b0;
      spk_valid_r   <= '0;
      spk_negative_r<= '0;
      for (int i = 0; i < N; i++) pot_r[i] <= '0;
    end else begin
      event_done_r <= 1'b0;

      if (!busy_r && ev_valid) begin
        neuron_index <= '0;
        busy_r       <= 1'b1;
      end else if (busy_r) begin
        logic signed [POT_W-1:0] decayed;
        logic signed [POT_W-1:0] summed;
        logic                    fire_positive;
        logic                    fire_negative;

        decayed = pot_r[neuron_index] >>> shift_amt;
        summed  = decayed + acc[neuron_index];
        fire_positive = (summed >= THETA_VALUE);
        fire_negative = ENABLE_NEGATIVE_SPIKES && (summed <= THETA_LOW_VALUE);

        spk_valid_r[neuron_index]    <= fire_positive;
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
          event_done_r <= 1'b1;
        end else begin
          neuron_index <= neuron_index + 1'b1;
        end
      end
    end
  end

  assign spk_valid    = spk_valid_r;
  assign spk_negative = spk_negative_r;

endmodule
