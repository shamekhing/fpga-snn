// pkg_snn.sv
// Central parameters for the SNN accelerator. Change sizes and widths here only.
package pkg_snn;

  // ---- Network size ----
  localparam int IMAGE_ROWS = 28;
  localparam int IMAGE_COLS = 28;
  localparam int N_INPUT   = IMAGE_ROWS * IMAGE_COLS; // input pixel weights
  localparam int N_INPUT_STREAMS = IMAGE_ROWS; // one differential stream per row
  localparam int N_HIDDEN  = 128;   // hidden LIF neurons (time-multiplexed)
  localparam int N_OUTPUT  = 10;    // one per digit

  // ---- Bit widths (size from Python model's observed maxima) ----
  localparam int W_W       = 4;     // synaptic weight width (signed)
  localparam int POT_W     = 16;    // membrane potential width (signed)
  localparam int TIME_W    = 16;    // absolute time register width
  localparam int DT_W      = 8;     // delta-time field in spike events

  // ---- Neuron parameters ----
  localparam int THETA     = 16;    // firing threshold (integer, same units as weights)

  // ---- Stream ----
  localparam int OVERFLOW_CODE = 8'hFF; // reserved all-ones delta = timer overflow spike

endpackage
