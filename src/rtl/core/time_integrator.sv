// time_integrator.sv
// Converts one input's differential (delta) time stream into absolute time.
// Overflow spikes advance time without generating a spike.
// TEMPLATE: interface and basic accumulation only.
import pkg_snn::*;

module time_integrator (
    input  logic                clk,
    input  logic                rst_n,
    input  logic                clear,
    input  logic                step,          // consume next delta from stream
    input  logic [DT_W-1:0]     delta,         // delta time from stream
    output logic [TIME_W-1:0]   abs_time,      // accumulated absolute time
    output logic                is_overflow    // delta == OVERFLOW_CODE
);

  logic [TIME_W-1:0] acc;

  assign is_overflow = (delta == DT_W'(OVERFLOW_CODE));

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)       acc <= '0;
    else if (clear)   acc <= '0;
    else if (step)    acc <= acc + TIME_W'(delta);
  end

  assign abs_time = acc;

endmodule
