// Serial minimum-time search for one layer's pending input events.
import pkg_snn::*;

module event_sched #(
    parameter int N = N_INPUT_STREAMS
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    scan_req,    // pulse while idle to scan once
    input  logic                    ev_ready,    // accept a held event
    input  logic [TIME_W-1:0]       next_t [N],   // hold stable until scan_done
    input  logic                    valid [N],
    input  logic                    negative [N],// 1 denotes a negative spike
    output logic                    scan_busy,
    output logic                    scan_done,   // pulse, including an empty scan
    output logic                    ev_valid,    // held until ev_ready
    output logic [TIME_W-1:0]       ev_time,
    output logic [N-1:0]            ev_mask,
    output logic [N-1:0]            ev_negative
);

  localparam int INDEX_W = (N > 1) ? $clog2(N) : 1;

  typedef enum logic [1:0] {
    IDLE,
    SCAN,
    FINALIZE,
    HOLD
  } state_e;

  state_e state;
  logic [INDEX_W-1:0] scan_index;
  logic [TIME_W-1:0] min_time_r;
  logic [N-1:0] mask_r;
  logic [N-1:0] negative_r;
  logic found_r;

  assign scan_busy = (state != IDLE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= IDLE;
      scan_index  <= '0;
      min_time_r  <= '0;
      mask_r      <= '0;
      negative_r  <= '0;
      found_r     <= 1'b0;
      scan_done   <= 1'b0;
      ev_valid    <= 1'b0;
      ev_time     <= '0;
      ev_mask     <= '0;
      ev_negative <= '0;
    end else begin
      scan_done <= 1'b0;
      case (state)
        IDLE: begin
          if (scan_req) begin
            scan_index <= '0;
            min_time_r <= '0;
            mask_r     <= '0;
            negative_r <= '0;
            found_r    <= 1'b0;
            ev_valid   <= 1'b0;
            ev_time    <= '0;
            ev_mask    <= '0;
            ev_negative <= '0;
            state      <= SCAN;
          end
        end

        SCAN: begin
          if (valid[scan_index]) begin
            if (!found_r || (next_t[scan_index] < min_time_r)) begin
              min_time_r <= next_t[scan_index];
              mask_r     <= {{(N-1){1'b0}}, 1'b1} << scan_index;
              negative_r <= {{(N-1){1'b0}}, negative[scan_index]} << scan_index;
              found_r    <= 1'b1;
            end else if (next_t[scan_index] == min_time_r) begin
              mask_r[scan_index]     <= 1'b1;
              negative_r[scan_index] <= negative[scan_index];
            end
          end

          if (scan_index == INDEX_W'(N - 1)) begin
            state <= FINALIZE;
          end else begin
            scan_index <= scan_index + 1'b1;
          end
        end

        FINALIZE: begin
          scan_done <= 1'b1;
          if (found_r) begin
            ev_valid    <= 1'b1;
            ev_time     <= min_time_r;
            ev_mask     <= mask_r;
            ev_negative <= negative_r;
            state       <= HOLD;
          end else begin
            ev_valid    <= 1'b0;
            ev_time     <= '0;
            ev_mask     <= '0;
            ev_negative <= '0;
            state       <= IDLE;
          end
        end

        HOLD: begin
          if (ev_ready) begin
            ev_valid    <= 1'b0;
            ev_time     <= '0;
            ev_mask     <= '0;
            ev_negative <= '0;
            state       <= IDLE;
          end
        end

        default: begin
          state       <= IDLE;
          ev_valid    <= 1'b0;
          ev_time     <= '0;
          ev_mask     <= '0;
          ev_negative <= '0;
        end
      endcase
    end
  end

endmodule
