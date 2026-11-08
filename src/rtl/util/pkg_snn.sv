// pkg_snn.sv
// Central parameters and shared types for the SNN accelerator.
// Change sizes and widths here first; then review every address width and
// testbench assumption that depends on them.
package pkg_snn;

  // ---- Network size ----
  localparam int IMAGE_ROWS = 28;
  localparam int IMAGE_COLS = 28;
  localparam int N_INPUT   = IMAGE_ROWS * IMAGE_COLS; // flattened input weights
  localparam int N_INPUT_STREAMS = IMAGE_ROWS; // one differential stream per row
  localparam int N_HIDDEN  = 128;   // hidden LIF neurons (time-multiplexed)
  localparam int N_OUTPUT  = 10;    // one per digit

  // ---- Bit widths (from Python model's observed maxima) ----
  localparam int W_W       = 4;     // synaptic weight width (signed)
  localparam int POT_W     = 16;    // membrane potential width (signed)
  localparam int TIME_W    = 16;    // absolute-time register width
  localparam int DT_W      = 8;     // delta-time field of a spike event

  // ---- Neuron parameters ----
  localparam int THETA     = 16;    // firing threshold (integer, same units as weights)

  // ---- Event & result types ----
  // A single accepted event handed to a LIF layer (the scheduler's output).
  // Bundling {valid, dt, negative} into one port instead of three.
  typedef struct packed {
    logic                       valid;
    logic [DT_W-1:0]            dt;
    logic                       negative;
  } event_in_t;

  // One neuron's result: the registered spike indications plus the current
  // membrane potential. Bundling {spk_valid, spk_negative, pot} into one
  // `neuron_out_t` turns the result bus from three separate ports into one.
  typedef struct packed {
    logic                     spk_valid;
    logic                     spk_negative;
    logic signed [POT_W-1:0]  pot;
  } neuron_out_t;

  // ---- Stream ----
  localparam int OVERFLOW_CODE = 8'hFF; // reserved all-ones delta = timer overflow spike

endpackage
