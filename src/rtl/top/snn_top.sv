// snn_top.sv
// Top-level SNN datapath: 28 differential-time row streams -> time
// integrators -> common minimum-time stage (event_sched) -> weight
// accumulator -> time-multiplexed hidden LIF layer (128) -> parallel output
// LIF layer (10) -> spike counters -> argmax.
//
// A centralized sequencer walks the row streams: per-row head registers feed
// the scheduler; tied rows are consumed one at a time (each event costs one
// hidden-weight ROM read with one cycle of latency), all contributions of one
// event time land in `acc` before the single hidden update, and hidden spikes
// feed `acc_out` before the single output update. `done` pulses for one cycle
// once every row stream is exhausted.
import pkg_snn::*;

module snn_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        done,
    output logic [3:0]  digit
);

  // ============================== ROMs ==============================
  localparam int STREAM_DEPTH = 16384;
  localparam int HW_WORD_W    = N_HIDDEN * W_W;  // 512
  localparam int OW_WORD_W    = N_OUTPUT * W_W;  // 40

  stream_rom #(
    .DEPTH (STREAM_DEPTH)
  ) u_stream (
    .addr (stream_addr),
    .data (stream_word)
  );

  base_rom u_base (
    .addr (base_addr),
    .data (base_word)
  );

  weight_rom #(
    .INIT_FILE ("data/weights/hidden_w.mem"),
    .N_NEURONS (N_HIDDEN),
    .DEPTH     (N_INPUT),
    .ADDR_W    (10)
  ) u_hidden_w (
    .clk  (clk),
    .addr (hw_addr),
    .data (hw_word)
  );

  weight_rom #(
    .INIT_FILE ("data/weights/output_w.mem"),
    .N_NEURONS (N_OUTPUT),
    .DEPTH     (N_HIDDEN),
    .ADDR_W    (7)
  ) u_output_w (
    .clk  (clk),
    .addr (ow_addr),
    .data (ow_word)
  );

  // ====================== per-row sequencer state ======================
  logic [15:0]        stream_addr;
  logic [15:0]        stream_word;
  logic [4:0]         base_addr;
  logic [15:0]        base_word;

  logic [15:0]        ptr        [N_INPUT_STREAMS];
  logic [COUNT_W-1:0] remaining  [N_INPUT_STREAMS];
  logic               head_valid [N_INPUT_STREAMS];
  logic [DT_W-1:0]    head_dt    [N_INPUT_STREAMS];
  logic [4:0]         head_col   [N_INPUT_STREAMS];
  logic               head_neg   [N_INPUT_STREAMS];

  // Time integrators: one per row.
  logic               ti_step  [N_INPUT_STREAMS];
  logic [DT_W-1:0]    ti_delta [N_INPUT_STREAMS];
  logic [TIME_W-1:0]  ti_time  [N_INPUT_STREAMS];
  logic               ti_ovf   [N_INPUT_STREAMS];

  // Scheduler inputs: next pending event time per row.
  logic [TIME_W-1:0]       next_t   [N_INPUT_STREAMS];
  logic                    sched_valid [N_INPUT_STREAMS];
  logic                    sched_negative [N_INPUT_STREAMS];

  generate
    for (genvar r = 0; r < N_INPUT_STREAMS; r++) begin : g_row
      time_integrator u_ti (
        .clk        (clk),
        .rst_n      (rst_n),
        .clear      (clear_state),
        .step       (ti_step[r]),
        .delta      (ti_delta[r]),
        .abs_time   (ti_time[r]),
        .is_overflow(ti_ovf[r])
      );
      assign next_t[r]        = ti_time[r] + TIME_W'({8'h00, head_dt[r]});
      assign sched_valid[r]   = head_valid[r];
      assign sched_negative[r]= head_neg[r];
    end
  endgenerate

  // ============================ accumulators ============================
  logic signed [POT_W-1:0] acc     [N_HIDDEN];
  logic signed [POT_W-1:0] acc_out [N_OUTPUT];

  // Weight words unpacked into per-neuron signed values.
  logic [HW_WORD_W-1:0] hw_word;
  logic [OW_WORD_W-1:0] ow_word;
  logic [9:0]           hw_addr;
  logic [6:0]           ow_addr;
  logic signed [POT_W-1:0] hw_ext [N_HIDDEN];
  logic signed [POT_W-1:0] ow_ext [N_OUTPUT];

  generate
    for (genvar n = 0; n < N_HIDDEN; n++) begin : g_hw
      logic [W_W-1:0] w4;
      assign w4 = hw_word[n*W_W +: W_W];
      assign hw_ext[n] = {{(POT_W-W_W){w4[W_W-1]}}, w4};
    end
  endgenerate

  generate
    for (genvar k = 0; k < N_OUTPUT; k++) begin : g_ow
      logic [W_W-1:0] w4;
      assign w4 = ow_word[k*W_W +: W_W];
      assign ow_ext[k] = {{(POT_W-W_W){w4[W_W-1]}}, w4};
    end
  endgenerate

  // ============================== layers ==============================
  logic [N_HIDDEN-1:0] hidden_spk_valid;
  logic [N_HIDDEN-1:0] hidden_spk_negative;
  logic signed [POT_W-1:0] hidden_pot [N_HIDDEN];
  logic hidden_event_done;

  logic [N_OUTPUT-1:0] output_spk_valid;
  logic [N_OUTPUT-1:0] output_spk_negative;
  logic signed [POT_W-1:0] output_pot [N_OUTPUT];
  logic output_event_done;

  logic [16*N_OUTPUT-1:0] output_count_flat;

  spike_counter u_counter (
    .clk   (clk),
    .rst_n (rst_n),
    .clear (clear_state),
    .spk_in(output_spk_valid),
    .count (output_count_flat)
  );

  argmax u_argmax (
    .count (output_count_flat),
    .digit (digit)
  );

  lif_layer #(
    .N (N_HIDDEN),
    .ENABLE_NEGATIVE_SPIKES (1'b0)
  ) u_hidden (
    .clk          (clk),
    .rst_n        (rst_n),
    .clear        (clear_state),
    .ev_valid     (hidden_start),
    .acc          (acc),
    .elapsed      (elapsed_hidden),
    .spk_valid    (hidden_spk_valid),
    .spk_negative (hidden_spk_negative),
    .pot          (hidden_pot),
    .event_done   (hidden_event_done)
  );

  lif_layer_parallel #(
    .N (N_OUTPUT),
    .ENABLE_NEGATIVE_SPIKES (1'b0)
  ) u_output (
    .clk          (clk),
    .rst_n        (rst_n),
    .clear        (clear_state),
    .ev_valid     (output_start),
    .acc_out      (acc_out),
    .elapsed_out  (elapsed_out),
    .spk_valid    (output_spk_valid),
    .spk_negative (output_spk_negative),
    .pot          (output_pot),
    .event_done   (output_event_done)
  );

  // ============================ control ============================
  logic clear_state;
  logic scan_req;
  logic ev_ready;
  logic hidden_start;
  logic output_start;
  logic [TIME_W-1:0] elapsed_hidden;
  logic [TIME_W-1:0] elapsed_out;

  logic [TIME_W-1:0] ev_time_r;
  logic [N_INPUT_STREAMS-1:0] ev_mask_r;

  logic [TIME_W-1:0] t_last_hidden;
  logic [TIME_W-1:0] t_last_out;

  logic [4:0] init_row;   // 0..27 during INIT
  logic [4:0] acc_row;    // row currently consuming its event group
  logic [6:0] out_scan;   // hidden-neuron scan for the output accumulator

  logic [DT_W-1:0] acc_dt_r;
  logic [4:0]      acc_col_r;
  logic            acc_neg_r;

  logic [N_HIDDEN-1:0] hidden_spk_latched;

  // Golden-trace observation registers (sampled by the testbench).
  logic [TIME_W-1:0] trace_time;
  logic [7:0]        trace_hspk;
  logic [7:0]        trace_ospk;
  logic              trace_strobe;

  // ---- scheduler ----
  logic scan_done;
  logic sched_ev_valid;
  logic [TIME_W-1:0] sched_ev_time;
  logic [N_INPUT_STREAMS-1:0] sched_ev_mask;

  event_sched #(
    .N (N_INPUT_STREAMS)
  ) u_sched (
    .clk         (clk),
    .rst_n       (rst_n),
    .scan_req    (scan_req),
    .ev_ready    (ev_ready),
    .next_t      (next_t),
    .valid       (sched_valid),
    .negative    (sched_negative),
    .scan_busy   (),
    .scan_done   (scan_done),
    .ev_valid    (sched_ev_valid),
    .ev_time     (sched_ev_time),
    .ev_mask     (sched_ev_mask),
    .ev_negative ()
  );

  // ---- helpers ----
  // Lowest set bit of m at index >= start, or -1 when none.
  function automatic int next_set_bit(input logic [N_INPUT_STREAMS-1:0] m,
                                      input int start);
    int found;
    begin
      found = -1;
      for (int r = N_INPUT_STREAMS-1; r >= start; r--)
        if (m[r]) found = r;
      next_set_bit = found;
    end
  endfunction

  function automatic logic [7:0] popcount_n(input logic [N_HIDDEN-1:0] v);
    int c;
    begin
      c = 0;
      for (int i = 0; i < N_HIDDEN; i++) if (v[i]) c = c + 1;
      popcount_n = c[7:0];
    end
  endfunction

  function automatic logic [7:0] popcount_o(input logic [N_OUTPUT-1:0] v);
    int c;
    begin
      c = 0;
      for (int i = 0; i < N_OUTPUT; i++) if (v[i]) c = c + 1;
      popcount_o = c[7:0];
    end
  endfunction

  // ---- FSM ----
  typedef enum logic [4:0] {
    S_IDLE, S_INIT_BASE, S_INIT_COUNT, S_INIT_WORD, S_INIT_NEXT,
    S_SCAN, S_ACC_WADDR, S_ACC_WWAIT, S_ACC_WADD, S_ACC_WORD, S_ACC_NEXT,
    S_HIDDEN_START, S_HIDDEN_WAIT,
    S_OUT_ADDR, S_OUT_WAIT, S_OUT_USE, S_OUT_START, S_OUT_DONE,
    S_DONE_PULSE
  } state_e;

  state_e state;
  state_e state_n;

  // Lowest tied row (scheduler mask) for the S_SCAN handoff.
  logic [4:0] first_tied;
  logic       first_tied_found;
  always_comb begin
    int b;
    b = next_set_bit(sched_ev_mask, 0);
    if (b < 0) begin
      first_tied       = 5'd0;
      first_tied_found = 1'b0;
    end else begin
      first_tied       = b;
      first_tied_found = 1'b1;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state              <= S_IDLE;
      clear_state        <= 1'b0;
      scan_req           <= 1'b0;
      ev_ready           <= 1'b0;
      hidden_start       <= 1'b0;
      output_start       <= 1'b0;
      done               <= 1'b0;
      init_row           <= '0;
      acc_row            <= '0;
      out_scan           <= '0;
      ev_time_r          <= '0;
      ev_mask_r          <= '0;
      t_last_hidden      <= '0;
      t_last_out         <= '0;
      elapsed_hidden     <= '0;
      elapsed_out        <= '0;
      acc_dt_r           <= '0;
      acc_col_r          <= '0;
      acc_neg_r          <= 1'b0;
      hidden_spk_latched <= '0;
      trace_time         <= '0;
      trace_hspk         <= '0;
      trace_ospk         <= '0;
      trace_strobe       <= 1'b0;
      stream_addr        <= '0;
      base_addr          <= '0;
      hw_addr            <= '0;
      ow_addr            <= '0;
      for (int r = 0; r < N_INPUT_STREAMS; r++) begin
        ptr[r]        <= '0;
        remaining[r]  <= '0;
        head_valid[r] <= 1'b0;
        head_dt[r]    <= '0;
        head_col[r]   <= '0;
        head_neg[r]   <= 1'b0;
        ti_step[r]    <= 1'b0;
        ti_delta[r]   <= '0;
      end
      for (int n = 0; n < N_HIDDEN; n++) acc[n] <= '0;
      for (int k = 0; k < N_OUTPUT; k++) acc_out[k] <= '0;
    end else begin
      clear_state  <= 1'b0;
      scan_req     <= 1'b0;
      ev_ready     <= 1'b0;
      hidden_start <= 1'b0;
      output_start <= 1'b0;
      done         <= 1'b0;
      trace_strobe <= 1'b0;
      for (int r = 0; r < N_INPUT_STREAMS; r++) ti_step[r] <= 1'b0;

      case (state)
        // ---- idle / start ----
        S_IDLE: begin
          state_n = S_IDLE;
          if (start) begin
            clear_state <= 1'b1;
            init_row    <= '0;
            base_addr   <= '0;      // base_word = base_mem[0] next cycle
            state_n     = S_INIT_BASE;
          end
        end

        // S_INIT_BASE: base_word now holds row init_row's base address.
        S_INIT_BASE: begin
          stream_addr <= base_word;   // point the stream ROM at the count word
          state_n     = S_INIT_COUNT;
        end

        // S_INIT_COUNT: stream_word = row init_row's count word.
        S_INIT_COUNT: begin
          remaining[init_row] <= stream_word[COUNT_W-1:0];
          ptr[init_row]       <= base_word + 16'd1;
          if (stream_word[COUNT_W-1:0] != '0) begin
            stream_addr <= base_word + 16'd1;
            state_n     = S_INIT_WORD;
          end else begin
            head_valid[init_row] <= 1'b0;
            state_n = S_INIT_NEXT;
          end
        end

        // S_INIT_WORD: stream_word = next unread entry of row init_row.
        S_INIT_WORD: begin
          if (stream_word[7:0] == DT_W'(OVERFLOW_CODE)) begin
            ti_step[init_row]   <= 1'b1;
            ti_delta[init_row]  <= stream_word[7:0];
            ptr[init_row]       <= ptr[init_row] + 16'd1;
            remaining[init_row] <= remaining[init_row] - 10'd1;
            stream_addr         <= ptr[init_row] + 16'd1;
            if (remaining[init_row] == 10'd1)
              state_n = S_INIT_NEXT;          // exhausted; head stays invalid
            else
              state_n = S_INIT_WORD;
          end else begin
            head_dt[init_row]    <= stream_word[7:0];
            head_col[init_row]   <= stream_word[12:8];
            head_neg[init_row]   <= stream_word[13];
            head_valid[init_row] <= 1'b1;
            ptr[init_row]        <= ptr[init_row] + 16'd1;
            remaining[init_row]  <= remaining[init_row] - 10'd1;
            state_n = S_INIT_NEXT;
          end
        end

        // S_INIT_NEXT: advance to the next row (or start scanning).
        S_INIT_NEXT: begin
          if (init_row == 5'd27) begin
            scan_req <= 1'b1;
            state_n  = S_SCAN;
          end else begin
            init_row  <= init_row + 5'd1;
            base_addr <= init_row + 5'd1;
            state_n   = S_INIT_BASE;
          end
        end

        // ---- scheduler handshake ----
        S_SCAN: begin
          state_n = S_SCAN;
          if (scan_done) begin
            if (sched_ev_valid && first_tied_found) begin
              ev_time_r <= sched_ev_time;
              ev_mask_r <= sched_ev_mask;
              ev_ready  <= 1'b1;
              for (int n = 0; n < N_HIDDEN; n++) acc[n] <= '0;
              acc_row   <= first_tied;
              acc_col_r <= head_col[first_tied];
              acc_neg_r <= head_neg[first_tied];
              acc_dt_r  <= head_dt[first_tied];
              state_n   = S_ACC_WADDR;
            end else begin
              state_n = S_DONE_PULSE;          // all rows exhausted
            end
          end
        end

        // ---- consume one row's group of same-time events ----
        // acc_col_r/acc_neg_r/acc_dt_r hold the pending event to accumulate.
        S_ACC_WADDR: begin
          hw_addr <= acc_row * 10'd28 + {5'h00, acc_col_r};
          state_n = S_ACC_WWAIT;
        end

        S_ACC_WWAIT: begin
          state_n = S_ACC_WADD;      // weight ROM latches data at this edge
        end

        S_ACC_WADD: begin
          for (int n = 0; n < N_HIDDEN; n++)
            acc[n] <= acc[n] + (acc_neg_r ? -hw_ext[n] : hw_ext[n]);
          ti_step[acc_row]  <= 1'b1;
          ti_delta[acc_row] <= acc_dt_r;
          if (remaining[acc_row] != '0) begin
            stream_addr <= ptr[acc_row];
            state_n     = S_ACC_WORD;
          end else begin
            head_valid[acc_row] <= 1'b0;
            state_n = S_ACC_NEXT;
          end
        end

        // S_ACC_WORD: stream_word = entry at ptr[acc_row] (ptr not yet past it).
        S_ACC_WORD: begin
          if (stream_word[7:0] == 8'h00) begin
            // same-time repeat: consume and accumulate
            acc_col_r <= stream_word[12:8];
            acc_neg_r <= stream_word[13];
            acc_dt_r  <= 8'h00;
            ptr[acc_row]       <= ptr[acc_row] + 16'd1;
            remaining[acc_row] <= remaining[acc_row] - 10'd1;
            stream_addr        <= ptr[acc_row] + 16'd1;
            state_n = S_ACC_WADDR;
          end else if (stream_word[7:0] == DT_W'(OVERFLOW_CODE)) begin
            ti_step[acc_row]   <= 1'b1;
            ti_delta[acc_row]  <= stream_word[7:0];
            ptr[acc_row]       <= ptr[acc_row] + 16'd1;
            remaining[acc_row] <= remaining[acc_row] - 10'd1;
            stream_addr        <= ptr[acc_row] + 16'd1;
            if (remaining[acc_row] == 10'd1) begin
              head_valid[acc_row] <= 1'b0;
              state_n = S_ACC_NEXT;
            end else begin
              state_n = S_ACC_WORD;
            end
          end else begin
            // regular event at a later time: becomes the new head
            head_dt[acc_row]    <= stream_word[7:0];
            head_col[acc_row]   <= stream_word[12:8];
            head_neg[acc_row]   <= stream_word[13];
            head_valid[acc_row] <= 1'b1;
            ptr[acc_row]        <= ptr[acc_row] + 16'd1;
            remaining[acc_row]  <= remaining[acc_row] - 10'd1;
            state_n = S_ACC_NEXT;
          end
        end

        // S_ACC_NEXT: next tied row, or the hidden update.
        S_ACC_NEXT: begin
          int nxt;
          nxt = next_set_bit(ev_mask_r, acc_row + 1);
          if (nxt >= 0) begin
            acc_row   <= nxt[4:0];
            acc_col_r <= head_col[nxt];
            acc_neg_r <= head_neg[nxt];
            acc_dt_r  <= head_dt[nxt];
            state_n   = S_ACC_WADDR;
          end else begin
            elapsed_hidden <= ev_time_r - t_last_hidden;
            state_n = S_HIDDEN_START;
          end
        end

        // ---- hidden layer update (one per event time) ----
        S_HIDDEN_START: begin
          hidden_start <= 1'b1;
          state_n      = S_HIDDEN_WAIT;
        end

        S_HIDDEN_WAIT: begin
          state_n = S_HIDDEN_WAIT;
          if (hidden_event_done) begin
            t_last_hidden      <= ev_time_r;
            hidden_spk_latched <= hidden_spk_valid;
            trace_time         <= ev_time_r;
            trace_hspk         <= popcount_n(hidden_spk_valid);
            if (|hidden_spk_valid) begin
              for (int k = 0; k < N_OUTPUT; k++) acc_out[k] <= '0;
              out_scan <= '0;
              state_n  = S_OUT_ADDR;
            end else begin
              trace_ospk   <= 8'h00;
              trace_strobe <= 1'b1;
              scan_req     <= 1'b1;
              state_n      = S_SCAN;
            end
          end
        end

        // ---- output accumulator: W2[n] for every spiking hidden neuron ----
        S_OUT_ADDR: begin
          ow_addr <= out_scan;
          state_n = S_OUT_WAIT;
        end

        S_OUT_WAIT: begin
          state_n = S_OUT_USE;
        end

        S_OUT_USE: begin
          if (hidden_spk_latched[out_scan]) begin
            for (int k = 0; k < N_OUTPUT; k++)
              acc_out[k] <= acc_out[k] + ow_ext[k];
          end
          out_scan <= out_scan + 7'd1;
          if (out_scan == 7'd127) begin
            elapsed_out <= ev_time_r - t_last_out;
            state_n     = S_OUT_START;
          end else begin
            state_n = S_OUT_ADDR;
          end
        end

        S_OUT_START: begin
          output_start <= 1'b1;
          state_n      = S_OUT_DONE;
        end

        S_OUT_DONE: begin
          state_n = S_OUT_DONE;
          if (output_event_done) begin
            t_last_out   <= ev_time_r;
            trace_ospk   <= popcount_o(output_spk_valid);
            trace_strobe <= 1'b1;
            scan_req     <= 1'b1;
            state_n      = S_SCAN;
          end
        end

        // ---- completion ----
        S_DONE_PULSE: begin
          done    <= 1'b1;
          state_n = S_IDLE;
        end

        default: state_n = S_IDLE;
      endcase

      // Registered after the case so the NBA samples the freshly computed
      // next state (a transfer placed before the case would latch the
      // previous cycle's state_n and stall the FSM).
      state <= state_n;
    end
  end

endmodule



  // (state machine in the second half of this file)
