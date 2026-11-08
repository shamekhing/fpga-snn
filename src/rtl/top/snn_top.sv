// Top-level event-driven SNN datapath.
// The control window is deliberately bounded while the stream end protocol is
// finalized; all accelerator layers are nevertheless instantiated and clocked.
import pkg_snn::*;

module snn_top (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    output logic        done,
    output logic [3:0]  digit
);

  localparam int RUN_CYCLES = 1024;
  localparam int RUN_W = $clog2(RUN_CYCLES + 1);

  logic clear_state;
  logic running;
  logic [RUN_W-1:0] run_count;
  logic [15:0] stream_addr;
  logic [DT_W-1:0] stream_data;

  // Per-stream next event time used by the scheduler's minimum search.
  logic [TIME_W-1:0] next_time [0:N_INPUT_STREAMS-1];

  // Per-stream event valid and negative masks for the scheduler's minimum
  // search. The top-level broadcasts the stream ROM word to every row; the
  // scheduler reduces these to a single accepted event on `ev_valid`.
  logic stream_valid [0:N_INPUT_STREAMS-1];
  logic stream_negative [0:N_INPUT_STREAMS-1];

  // Scheduler connection state (passthrough, documented here for clarity).
  logic scan_busy;
  logic scan_done;
  logic hidden_busy;
  logic hidden_event_done;
  logic ev_valid;
  logic [TIME_W-1:0] ev_time;
  logic [N_INPUT_STREAMS-1:0] ev_mask;
  logic [N_INPUT_STREAMS-1:0] ev_negative;

  // One event bus from the scheduler into the hidden layer.
  // Bundling valid/dt/negative collapses three wires into one port.
  event_in_t ev_in_hidden;

  // Hidden-layer result bank. `hidden_pot` is a per-neuron state readout;
  // the classification path consumes `hidden_spike`/`hidden_negative`.
  // Result bank: flat arrays (not packed struct arrays) are used because
  // Icarus cannot expand `neuron_out_t` array indexing in generate blocks.
  // `hidden_pot` is a per-neuron state readout; the spike bank is packed into
  // `hidden_spike_p`/`hidden_negative_p` purely for the `|` reduction, which
  // Icarus only accepts on packed vectors.
  logic hidden_spike [0:N_HIDDEN-1];
  logic hidden_negative [0:N_HIDDEN-1];
  logic signed [POT_W-1:0] hidden_pot [0:N_HIDDEN-1];
  logic hidden_any_spike;
  logic hidden_any_negative;
  // Packed intermediate for the OR reduction below.
  logic [N_HIDDEN-1:0] hidden_spike_p;
  logic [N_HIDDEN-1:0] hidden_negative_p;

  // Output layer: event bus in, flat result bus out.
  event_in_t ev_in_output;
  logic output_spike [0:N_OUTPUT-1];
  logic output_negative [0:N_OUTPUT-1];
  logic signed [POT_W-1:0] output_pot [0:N_OUTPUT-1];
  logic [15:0] output_count [0:N_OUTPUT-1];

  // Packed copy of output_spike for spike_counter's packed spk_in port.
  logic [N_OUTPUT-1:0] output_spike_p;
  // Weight storage (ROM data is unsigned; the slices are reinterpreted as
  // signed per the model's 2's-complement packing).
  logic [N_HIDDEN*W_W-1:0] hidden_weight_data;
  logic signed [W_W-1:0] hidden_w_row [0:N_HIDDEN-1];

  logic [N_OUTPUT*W_W-1:0] output_weight_data;
  logic signed [W_W-1:0] output_w_row [0:N_OUTPUT-1];

  // ---- Stream / scheduler wiring ----
  stream_rom #(
    .INIT_FILE ("data/streams/stream.mem")
  ) u_stream (
    .clk  (clk),
    .addr (stream_addr),
    .data (stream_data)
  );

  event_sched #(
    .N (N_INPUT_STREAMS)
  ) u_input_scheduler (
    .clk         (clk),
    .rst_n       (rst_n),
    .scan_req    (running && !scan_busy && !hidden_busy && !ev_valid),
    .ev_ready    (hidden_event_done),
    .next_t      (next_time),
    .valid       (stream_valid),
    .negative    (stream_negative),
    .scan_busy   (scan_busy),
    .scan_done   (scan_done),
    .ev_valid    (ev_valid),
    .ev_time     (ev_time),
    .ev_mask     (ev_mask),
    .ev_negative (ev_negative)
  );

  // Every row stream carries the same delta word; the scheduler reduces these
  // to a single accepted event on `ev_valid`.
  generate
    for (genvar s = 0; s < N_INPUT_STREAMS; s++) begin : g_stream
      assign next_time[s] = TIME_W'(stream_data);
      assign stream_valid[s] = running;
      assign stream_negative[s] = 1'b0;
    end
  endgenerate

  // Pack the scheduler's event into one bus for the hidden layer.
  assign ev_in_hidden.valid = ev_valid;
  assign ev_in_hidden.dt = ev_time[DT_W-1:0];
  assign ev_in_hidden.negative = |ev_negative;

  // ---- Hidden layer ----
  lif_layer #(
    .N (N_HIDDEN)
  ) u_hidden_layer (
    .clk           (clk),
    .rst_n         (rst_n),
    .clear         (clear_state),
    .ev            (ev_in_hidden),
    .w_row         (hidden_w_row),
    .spk_valid     (hidden_spike),
    .spk_negative  (hidden_negative),
    .pot           (hidden_pot),
    .busy          (hidden_busy),
    .event_done    (hidden_event_done)
  );

  generate
    for (genvar s = 0; s < N_HIDDEN; s++) begin : g_pack_sp
      assign hidden_spike_p[s] = hidden_spike[s];
      assign hidden_negative_p[s] = hidden_negative[s];
    end
  endgenerate

  assign hidden_any_spike = |hidden_spike_p;
  assign hidden_any_negative = |hidden_negative_p;

  // Pack the output spike bank into a packed vector for spike_counter.
  generate
    for (genvar s = 0; s < N_OUTPUT; s++) begin : g_pack_out
      assign output_spike_p[s] = output_spike[s];
    end
  endgenerate

  // ---- Output layer ----
  // The output layer reuses the same accepted event, but with the hidden
  // layer's combined spike polarity and a unit delta.
  assign ev_in_output.valid = hidden_any_spike;
  assign ev_in_output.dt = {{(DT_W-1){1'b0}}, 1'b1};
  assign ev_in_output.negative = hidden_any_negative;

  lif_layer_parallel #(
    .N (N_OUTPUT)
  ) u_output_layer (
    .clk           (clk),
    .rst_n         (rst_n),
    .clear         (clear_state),
    .ev            (ev_in_output),
    .w_row         (output_w_row),
    .spk_valid     (output_spike),
    .spk_negative  (output_negative),
    .pot           (output_pot)
  );

  // ---- Output counting / classification ----
  spike_counter u_output_counter (
    .clk     (clk),
    .rst_n   (rst_n),
    .clear   (clear_state),
    .spk_in  (output_spike_p),
    .count   (output_count)
  );

  argmax u_argmax (
    .count (output_count),
    .digit (digit)
  );

  // ---- Control ----
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      clear_state <= 1'b0;
      running     <= 1'b0;
      run_count   <= '0;
      stream_addr <= '0;
      done        <= 1'b0;
    end else begin
      clear_state <= 1'b0;
      done        <= 1'b0;
      if (!running) begin
        if (start) begin
          clear_state <= 1'b1;
          running     <= 1'b1;
          run_count   <= '0;
          stream_addr <= '0;
        end
      end else if (run_count == RUN_CYCLES - 1) begin
        running <= 1'b0;
        done    <= 1'b1;
      end else begin
        run_count <= run_count + 1'b1;
        if (hidden_event_done)
          stream_addr <= stream_addr + 1'b1;
      end
    end
  end

endmodule
