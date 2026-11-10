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

  // ---- Bit widths ----
  localparam int W_W       = 4;     // synaptic weight width (signed)
  localparam int POT_W     = 16;    // membrane potential width (signed)
  localparam int TIME_W    = 16;    // absolute-time register width
  localparam int DT_W      = 8;     // delta-time field of a spike event

  // ---- Neuron parameters ----
  // THETA = 26 was selected by sweeping integer thresholds 4..32 on the first
  // 2,000 training images with the quantized no-bias reference (see
  // data/golden/theta_report.txt and the README "Accuracy" section).
  localparam int THETA     = 26;    // firing threshold (integer, same units as weights)

  // ---- Legacy single-neuron bundles (kept for lif_neuron + tb_lif_neuron) ----
  // A single accepted event handed to a LIF core (the scheduler's output).
  typedef struct packed {
    logic                       valid;
    logic [DT_W-1:0]            dt;
    logic                       negative;
  } event_in_t;

  // One neuron's result: the registered spike indications plus the current
  // membrane potential.
  typedef struct packed {
    logic                     spk_valid;
    logic                     spk_negative;
    logic signed [POT_W-1:0]  pot;
  } neuron_out_t;

  // ---- Event word: 14 bits stored in a 16-bit word ----
  // {neg[0], column[4:0], dt[7:0]}: neg is the MSB (bit 13), column sits in
  // bits [12:8], dt occupies bits [7:0]; bits [15:14] are zero.
  // Overflow: dt == 8'hFF with column and neg as 0.
  typedef struct packed {
    logic        neg;
    logic [4:0]  column;
    logic [7:0]  dt;
  } event_word_t;

  // Row-stream count word (one per row, before its event words).
  // Width chosen from the data's maximum per-row event count so it can
  // represent values beyond 255.
  localparam int COUNT_W = 10;

  // Reserved all-ones delta = timer overflow (time-only) entry.
  localparam int OVERFLOW_CODE = 8'hFF;

endpackage
