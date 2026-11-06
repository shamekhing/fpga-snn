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
  logic [TIME_W-1:0] next_time [0:N_INPUT_STREAMS-1];
  logic stream_valid [0:N_INPUT_STREAMS-1];
  logic stream_negative [0:N_INPUT_STREAMS-1];

  logic scan_busy;
  logic scan_done;
  logic ev_valid;
  logic [TIME_W-1:0] ev_time;
  logic [N_INPUT_STREAMS-1:0] ev_mask;
  logic [N_INPUT_STREAMS-1:0] ev_negative;

  logic [N_HIDDEN*W_W-1:0] hidden_weight_data;
  logic signed [W_W-1:0] hidden_w_row [0:N_HIDDEN-1];
  logic hidden_spike [0:N_HIDDEN-1];
  logic hidden_negative [0:N_HIDDEN-1];
  logic signed [POT_W-1:0] hidden_pot [0:N_HIDDEN-1];
  logic hidden_any_spike;
  logic hidden_any_negative;

  logic [N_OUTPUT*W_W-1:0] output_weight_data;
  logic signed [W_W-1:0] output_w_row [0:N_OUTPUT-1];
  logic output_spike [0:N_OUTPUT-1];
  logic output_negative [0:N_OUTPUT-1];
  logic signed [POT_W-1:0] output_pot [0:N_OUTPUT-1];
  logic [N_OUTPUT-1:0] output_spike_bus;
  logic [15:0] output_count [0:N_OUTPUT-1];

  stream_rom #(
    .INIT_FILE ("data/streams/stream.mem")
  ) u_stream (
    .clk  (clk),
    .addr (stream_addr),
    .data (stream_data)
  );

  genvar s;
  generate
    for (s = 0; s < N_INPUT_STREAMS; s++) begin : g_stream
      assign next_time[s] = TIME_W'(stream_data);
      assign stream_valid[s] = running;
      assign stream_negative[s] = 1'b0;
    end
  endgenerate

  always_comb begin
    hidden_any_spike = 1'b0;
    hidden_any_negative = 1'b0;
    for (int h = 0; h < N_HIDDEN; h++) begin
      hidden_any_spike = hidden_any_spike || hidden_spike[h];
      hidden_any_negative = hidden_any_negative || hidden_negative[h];
    end
  end

  weight_rom #(
    .INIT_FILE  ("data/weights/hidden_w.mem"),
    .N_NEURONS  (N_HIDDEN),
    .DEPTH      (N_INPUT),
    .ADDR_W     (10)
  ) u_hidden_weights (
    .clk  (clk),
    .addr (stream_addr[9:0]),
    .data (hidden_weight_data)
  );

  weight_rom #(
    .INIT_FILE  ("data/weights/output_w.mem"),
    .N_NEURONS  (N_OUTPUT),
    .DEPTH      (N_HIDDEN),
    .ADDR_W     (7)
  ) u_output_weights (
    .clk  (clk),
    .addr (stream_addr[6:0]),
    .data (output_weight_data)
  );

  generate
    for (s = 0; s < N_HIDDEN; s++) begin : g_hidden_weights
      assign hidden_w_row[s] = hidden_weight_data[s*W_W +: W_W];
    end
    for (s = 0; s < N_OUTPUT; s++) begin : g_output_weights
      assign output_w_row[s] = output_weight_data[s*W_W +: W_W];
    end
    for (s = 0; s < N_OUTPUT; s++) begin : g_output_bus
      assign output_spike_bus[s] = output_spike[s];
    end
  endgenerate

  event_sched #(
    .N (N_INPUT_STREAMS)
  ) u_input_scheduler (
    .clk         (clk),
    .rst_n       (rst_n),
    .scan_req    (running && !scan_busy && !ev_valid),
    .ev_ready    (1'b1),
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

  lif_layer #(
    .N (N_HIDDEN)
  ) u_hidden_layer (
    .clk           (clk),
    .rst_n         (rst_n),
    .clear         (clear_state),
    .ev_valid      (ev_valid),
    .dt            (ev_time[DT_W-1:0]),
    .ev_negative   (|ev_negative),
    .w_row         (hidden_w_row),
    .spk_valid     (hidden_spike),
    .spk_negative  (hidden_negative),
    .pot           (hidden_pot)
  );

  lif_layer #(
    .N (N_OUTPUT)
  ) u_output_layer (
    .clk           (clk),
    .rst_n         (rst_n),
    .clear         (clear_state),
    .ev_valid      (hidden_any_spike),
    .dt            ({{(DT_W-1){1'b0}}, 1'b1}),
    .ev_negative   (hidden_any_negative),
    .w_row         (output_w_row),
    .spk_valid     (output_spike),
    .spk_negative  (output_negative),
    .pot           (output_pot)
  );

  spike_counter u_output_counter (
    .clk     (clk),
    .rst_n   (rst_n),
    .clear   (clear_state),
    .spk_in  (output_spike_bus),
    .count   (output_count)
  );

  argmax u_argmax (
    .count (output_count),
    .digit (digit)
  );

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
        if (ev_valid)
          stream_addr <= stream_addr + 1'b1;
      end
    end
  end

endmodule
