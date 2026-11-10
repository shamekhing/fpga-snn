// argmax.sv
// Selects the output neuron with the highest spike count.
// Ties resolve to the lowest index. `count` is the flat packed spike-count
// vector from spike_counter (neuron k in bits [16k+15:16k]).
import pkg_snn::*;

module argmax (
    input  logic [16*N_OUTPUT-1:0]  count,
    output logic [3:0]              digit
);

  always_comb begin
    digit = '0;
    for (int k = 1; k < N_OUTPUT; k++)
      if (count[16*k +: 16] > count[16*digit +: 16]) digit = 4'(k);
  end

endmodule

