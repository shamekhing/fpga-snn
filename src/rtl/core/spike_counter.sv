// spike_counter.sv
// Counts output spikes per output neuron for the current image.
// The count bus is a flat packed vector (16 bits per neuron, neuron k in
// bits [16k+15:16k]) so it passes through module boundaries cleanly.
import pkg_snn::*;

module spike_counter (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    clear,
    input  logic [N_OUTPUT-1:0]     spk_in,       // one spike pulse per output neuron
    output logic [16*N_OUTPUT-1:0]  count         // flat: neuron k in [16k+15:16k]
);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      count <= '0;
    end else if (clear) begin
      count <= '0;
    end else begin
      for (int k = 0; k < N_OUTPUT; k++)
        if (spk_in[k]) count[16*k +: 16] <= count[16*k +: 16] + 16'd1;
    end
  end

endmodule

