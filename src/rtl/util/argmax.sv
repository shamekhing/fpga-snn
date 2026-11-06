// argmax.sv
// Selects the output neuron with the highest spike count.
// Ties resolve to the lowest index.
import pkg_snn::*;

module argmax (
    input  logic [15:0]             count [N_OUTPUT],
    output logic [3:0]              digit
);

  always_comb begin
    digit = '0;
    for (int k = 1; k < N_OUTPUT; k++)
      if (count[k] > count[digit]) digit = 4'(k);
  end

endmodule
