// Top-level SNN datapath for the differential-time row-stream architecture.
//
// Per the paper, the datapath is: 28 differential-time row streams -> time
// integrators -> a common minimum-time stage (event_sched) -> a weight
// accumulator -> a time-multiplexed hidden LIF layer -> an output LIF layer.
//
// The control is a lightweight FSM. `start` begins a single image; `done` is
// asserted for one cycle when all 28 rows have been exhausted.
//
// Event word format (16 bits): {neg[0], column[4:0], dt[7:0]}
//   - overflow: dt == 0xFF with column and neg as 0
//   - each row begins with a 10-bit count word
import pkg_snn::*;

module snn_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        done,
    output logic [3:0]  digit
);

  // ---- Control state machine ----
  typedef enum logic [2:0] {
    STATE_RESET,
    STATE_IDLE,
    STATE_SCAN,
    STATE_ACCUMULATE,
    STATE_HIDDEN_UPDATE,
    STATE_OUTPUT_UPDATE,
    STATE_DONE
  } state_e;

  state_e state_r, state_n;

  // ---- Scheduler interface ----
  logic scan_req;
  logic ev_ready;
  logic [TIME_W-1:0] next_t [N_INPUT_STREAMS];
  logic valid [N_INPUT_STREAMS];
  logic negative [N_INPUT_STREAMS];
  logic scan_busy;
  logic scan_done;
  logic ev_valid;
  logic [TIME_W-1:0] ev_time;
  logic [N_INPUT_STREAMS-1:0] ev_mask;
  logic [N_INPUT_STREAMS-1:0] ev_negative;

  // ---- Accumulators ----
  // 128 hidden neurons, 16-bit signed each.
  logic signed [POT_W-1:0] acc [N_HIDDEN];
  // 10 output neurons, 16-bit signed each.
  logic signed [POT_W-1:0] acc_out [N_OUTPUT];

  // ---- Hidden layer ----
  logic [N_HIDDEN-1:0] hidden_spk_valid;
  logic [N_HIDDEN-1:0] hidden_spk_negative;
  logic signed [POT_W-1:0] hidden_pot [N_HIDDEN];
  logic hidden_event_done;

  // ---- Output layer ----
  logic [N_OUTPUT-1:0] output_spk_valid;
  logic [N_OUTPUT-1:0] output_spk_negative;
  logic signed [POT_W-1:0] output_pot [N_OUTPUT];

  // ---- Output counting ----
  logic [15:0] output_count [N_OUTPUT];

  // ---- Results ----
  logic [3:0] digit_r;
  logic done_r;

  // ---- FSM ----
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_r      <= STATE_RESET;
      start_r      <= 1'b0;
      done_r       <= 1'b0;
      scan_req     <= 1'b0;
      ev_ready     <= 1'b0;
      acc_ready    <= 1'b0;
      hidden_done  <= 1'b0;
    end else begin
      state_r      <= state_n;
      scan_req     <= 1'b0;
      ev_ready     <= 1'b0;
      acc_ready    <= 1'b0;
      done_r       <= 1'b0;
    end
  end

  // ---- Combinational state update ----
  always_comb begin
    state_n         = state_r;
    scan_req        = 1'b0;
    ev_ready        = 1'b0;
    acc_ready       = 1'b0;
    done_r          = 1'b0;
    ev_valid_hidden = 1'b0;
    ev_valid_out    = 1'b0;

    case (state_r)
      STATE_RESET: begin
        state_n = STATE_IDLE;
      end

      STATE_IDLE: begin
        if (start) begin
          state_n = STATE_SCAN;
          scan_req = 1'b1;
        end
      end

      STATE_SCAN: begin
        if (scan_done && ev_valid) begin
          state_n = STATE_ACCUMULATE;
        end
      end

      STATE_ACCUMULATE: begin
        // For each tied row, read the event and accumulate into `acc`.
        if (/* all tied rows processed */) begin
          state_n = STATE_HIDDEN_UPDATE;
        end
      end

      STATE_HIDDEN_UPDATE: begin
        if (hidden_event_done) begin
          state_n = STATE_OUTPUT_UPDATE;
        end
      end

      STATE_OUTPUT_UPDATE: begin
        if (/* output layer done */) begin
          state_n = STATE_IDLE;
          done_r = 1'b1;
        end
      end

      STATE_DONE: begin
        if (start) begin
          state_n = STATE_SCAN;
          scan_req = 1'b1;
        end
      end

      default: state_n = STATE_RESET;
    endcase
  end

  // ---- Module instantiations (see RTL source files) ----
  //
  // 1. 28 time_integrator instances (one per row)
  // 2. 28 row sequencers (ROM + overflow handling)
  // 3. event_sched #(.N(N_INPUT_STREAMS)) u_event_sched (...)
  // 4. Accumulator (acc [N_HIDDEN], acc_out [N_OUTPUT])
  // 5. lif_layer #(.N(N_HIDDEN)) u_hidden_layer (...)
  // 6. lif_layer_parallel #(.N(N_OUTPUT)) u_output_layer (...)
  // 7. spike_counter u_output_counter (...)
  // 8. argmax u_argmax (...)
  //
  // The full implementation is in the accompanying paper. The testbench
  // (tb_snn_top.sv) drives the control signals and reads the outputs.

endmodule
