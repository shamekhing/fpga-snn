// spike_counter.sv
// Counts output spikes per output neuron for the current image.
import pkg_snn::*;

module spike_counter (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    clear,
    input  logic [N_OUTPUT-1:0]     spk_in,       // one spike pulse per output neuron
    output logic [15:0]             count [N_OUTPUT]
);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int k = 0; k < N_OUTPUT; k++) count[k] <= '0;
    end else if (clear) begin
      for (int k = 0; k < N_OUTPUT; k++) count[k] <= '0;
    end else begin
      for (int k = 0; k < N_OUTPUT; k++)
        if (spk_in[k]) count[k] <= count[k] + 16'd1;
    end
  end

endmodule
