// Single signed-event LIF neuron with reset-to-mod thresholding.
// Consumes one `event_in_t` and produces one `neuron_out_t` so the core is a
// clean 1:1 source/destination of the neuron result bus.
import pkg_snn::*;

module lif_neuron #(
    parameter bit ENABLE_NEGATIVE_SPIKES = 1'b0,
    parameter int THETA_LOW = -THETA
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     clear,
    input  event_in_t                ev,
    input  logic signed [POT_W-1:0]    synaptic_input,
    output neuron_out_t              spk
);

  localparam logic signed [POT_W-1:0] THETA_VALUE = POT_W'(THETA);
  localparam logic signed [POT_W-1:0] THETA_LOW_VALUE = POT_W'(THETA_LOW);

  logic signed [POT_W-1:0] pot_r;
  neuron_out_t spk_r; // registered spike indications + current state
  logic signed [POT_W-1:0] decayed;
  logic signed [POT_W-1:0] summed;
  logic                    fire_positive;
  logic                    fire_negative;

  // Combinational read of the current state into the result struct.
  always_comb begin
    spk_r.pot = pot_r;
  end

  always_comb begin
    logic [DT_W-1:0] ev_dt;
    ev_dt = ev.dt;
    decayed       = pot_r >>> ev_dt;
    summed        = decayed + synaptic_input;
    fire_positive = ev.valid && (summed >= THETA_VALUE);
    fire_negative = ev.valid && ENABLE_NEGATIVE_SPIKES
                    && (summed <= THETA_LOW_VALUE);
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pot_r <= '0;
      spk_r.spk_valid   <= 1'b0;
      spk_r.spk_negative <= 1'b0;
    end else if (clear) begin
      pot_r <= '0;
      spk_r.spk_valid   <= 1'b0;
      spk_r.spk_negative <= 1'b0;
    end else begin
      if (ev.valid) begin
        if (fire_positive)
          pot_r <= summed - THETA_VALUE;
        else if (fire_negative)
          pot_r <= summed - THETA_LOW_VALUE;
        else
          pot_r <= summed;
      end
      spk_r.spk_valid      <= fire_positive || fire_negative;
      spk_r.spk_negative   <= fire_negative;
    end
  end

  assign spk = spk_r;

endmodule
