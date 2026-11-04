module sesr_image_proc(
  input          clock,
  input          reset,
  input  [31:0]  ctrl_reg,
  input  [31:0]  shape_reg,
  input  [31:0]  scale_reg,
  input  [63:0]  src0_addr_reg,
  input  [63:0]  src1_addr_reg,
  input  [63:0]  src2_addr_reg,
  input  [63:0]  dst0_addr_reg,
  input  [63:0]  dst1_addr_reg,
  input  [63:0]  dst2_addr_reg,
  input  [63:0]  dst3_addr_reg,
  output         active,
  output reg     task_done,
  output reg [63:0] perf_cycles,
  output reg [31:0] perf_read_beats,
  output reg [31:0] perf_write_beats,
  output reg [31:0] perf_pixels,
  output reg [63:0] perf_axi_wait_cycles,
  output reg [31:0] perf_tiles,
  input          stream_ch0_valid,
  input  [127:0] stream_ch0_data,
  input  [4:0]  stream_ch0_oc,
  input  [3:0]  stream_ch0_group,
  input          stream_ch1_valid,
  input  [127:0] stream_ch1_data,
  input  [4:0]  stream_ch1_oc,
  input  [3:0]  stream_ch1_group,
  output         stream_ready,
  output         stream_throttle,
  output         stream_active,
  output         axi_arid,
  output reg [63:0] axi_araddr,
  output reg [7:0] axi_arlen,
  output [2:0]   axi_arsize,
  output [1:0]   axi_arburst,
  output         axi_arlock,
  output [3:0]   axi_arcache,
  output [2:0]   axi_arprot,
  output [3:0]   axi_arqos,
  output reg     axi_arvalid,
  input          axi_arready,
  input  [127:0] axi_rdata,
  input          axi_rlast,
  input          axi_rvalid,
  output         axi_rready,
  output         axi_awid,
  output reg [63:0] axi_awaddr,
  output reg [7:0] axi_awlen,
  output [2:0]   axi_awsize,
  output [1:0]   axi_awburst,
  output         axi_awlock,
  output [3:0]   axi_awcache,
  output [2:0]   axi_awprot,
  output [3:0]   axi_awqos,
  output reg     axi_awvalid,
  input          axi_awready,
  output         axi_wid,
  output reg [127:0] axi_wdata,
  output [15:0]  axi_wstrb,
  output reg     axi_wlast,
  output reg     axi_wvalid,
  input          axi_wready,
  input          axi_bvalid,
  output         axi_bready
);

  localparam OP_PREPROCESS = 3'd0;
  localparam OP_POSTPROCESS = 3'd1;
  localparam OP_SPLIT = 3'd2;
  localparam OP_TAIL_FUSED = 3'd3;
  localparam OP_TAIL_ADD_FUSED = 3'd4;
  localparam OP_TAIL_ADD_STREAM_FUSED = 3'd5;

  localparam S_IDLE = 6'd0;
  localparam S_PRE_R0 = 6'd1;
  localparam S_PRE_R1 = 6'd2;
  localparam S_PRE_R2 = 6'd3;
  localparam S_PRE_CALC = 6'd4;
  localparam S_PRE_WY = 6'd5;
  localparam S_PRE_WCB = 6'd6;
  localparam S_PRE_WCR = 6'd7;
  localparam S_PRE_WF0 = 6'd8;
  localparam S_PRE_WF1 = 6'd9;
  localparam S_PRE_WF2 = 6'd10;
  localparam S_PRE_WF3 = 6'd11;
  localparam S_POST_RY0 = 6'd12;
  localparam S_POST_RY1 = 6'd13;
  localparam S_POST_RY2 = 6'd14;
  localparam S_POST_RY3 = 6'd15;
  localparam S_POST_RCB = 6'd16;
  localparam S_POST_RCR = 6'd17;
  localparam S_POST_CALC = 6'd18;
  localparam S_POST_W0 = 6'd19;
  localparam S_POST_W1 = 6'd20;
  localparam S_POST_W2 = 6'd21;
  localparam S_DONE = 6'd22;
  localparam S_POST_ADDR = 6'd23;
  localparam S_POST_RCB_REQ = 6'd24;
  localparam S_POST_W3 = 6'd25;
  localparam S_POST_CALC0 = 6'd26;
  localparam S_POST_CALC1 = 6'd27;
  localparam S_POST_CALC2 = 6'd28;
  localparam S_POST_CALC3 = 6'd29;
  localparam S_POST_NEXT = 6'd30;
  localparam S_TAIL_ADDR = 6'd31;
  localparam S_TAIL_R0 = 6'd32;
  localparam S_TAIL_R1 = 6'd33;
  localparam S_TAIL_R2 = 6'd34;
  localparam S_TAIL_R3 = 6'd35;
  localparam S_TAIL_PACK = 6'd36;
  localparam S_TAIL_SKIP0 = 6'd37;
  localparam S_TAIL_SKIP1 = 6'd38;
  localparam S_TAIL_ADD_PREP = 6'd39;
  localparam S_TAIL_ADD_FIRE = 6'd40;
  localparam S_TAIL_ADD_WAIT = 6'd41;
  localparam S_TAIL_ADD_PACK = 6'd42;
  localparam S_STREAM_WAIT = 6'd43;
  localparam S_STREAM_PACK = 6'd44;
  localparam S_SETUP = 6'd45;
  localparam S_LAUNCH = 6'd46;

  reg [5:0] state;
  reg [2:0] op;
  reg [15:0] width;
  reg [15:0] height;
  reg [3:0] scale;
  reg [63:0] src0_addr;
  reg [63:0] src1_addr;
  reg [63:0] src2_addr;
  reg [63:0] dst0_addr;
  reg [63:0] dst1_addr;
  reg [63:0] dst2_addr;
  reg [63:0] dst3_addr;
  reg [63:0] skip_y_addr;
  reg [31:0] total_pixels;
  reg [31:0] lr_pixels;
  reg [31:0] pixel_base;
  reg [383:0] pre_bgr_blk;
  reg [511:0] post_y_blk;
  reg [127:0] cb_blk;
  reg [127:0] cr_blk;
  reg [127:0] y_pack;
  reg [127:0] cb_pack;
  reg [127:0] cr_pack;
  reg [127:0] y_float_pack0;
  reg [127:0] y_float_pack1;
  reg [127:0] y_float_pack2;
  reg [127:0] y_float_pack3;
  reg [383:0] bgr_pack;
  reg [31:0] out_w;
  reg [31:0] out_x_base;
  reg [31:0] out_y_base;
  reg [31:0] cb_index_base;
  reg [31:0] cb_addr_aligned;
  reg [3:0] cb_offset_base;
  reg [63:0] pre_rgb_addr;
  reg [63:0] pre_y_addr;
  reg [63:0] pre_cb_addr;
  reg [63:0] pre_cr_addr;
  reg [63:0] pre_y_float_addr;
  reg [63:0] post_y_addr;
  reg [63:0] post_rgb_addr;
  reg [31:0] post_out_x;
  reg [1:0] post_row_phase;
  reg [31:0] post_cb_row_base;
  reg [31:0] post_cb_x_base;
  reg cbcr_cache_valid;
  reg [31:0] cbcr_cache_addr;
  reg [127:0] pre_y_pack_q;
  reg [127:0] pre_cb_pack_q;
  reg [127:0] pre_cr_pack_q;
  reg [127:0] pre_y_float_pack0_q;
  reg [127:0] pre_y_float_pack1_q;
  reg [127:0] pre_y_float_pack2_q;
  reg [127:0] pre_y_float_pack3_q;
  reg [383:0] post_bgr_pack_q;
  reg pre_save_y;
  reg [31:0] tail_cstep;
  reg [63:0] tail_addr0;
  reg [63:0] tail_addr1;
  reg [63:0] tail_addr2;
  reg [63:0] tail_addr3;
  reg [127:0] tail_src0;
  reg [127:0] tail_src1;
  reg [127:0] tail_src2;
  reg [127:0] tail_src3;
  reg [63:0] tail_skip_addr0;
  reg [63:0] tail_skip_addr1;
  reg [127:0] tail_skip0;
  reg [127:0] tail_skip1;
  reg [511:0] tail_raw_pack;
  reg [511:0] tail_skip_pack;
  reg [511:0] tail_add_x_pack;
  reg [511:0] tail_add_y_pack;
  reg tail_add_valid;
  reg [3:0] tail_add_wait;
  reg [127:0] stream_buf0 [0:255];
  reg [127:0] stream_buf1 [0:255];
  reg stream_fill_sel;
  reg stream_proc_sel;
  reg stream_buf_ready0;
  reg stream_buf_ready1;
  reg [31:0] stream_buf_base0;
  reg [31:0] stream_buf_base1;
  reg [31:0] stream_fill_base;
  reg [31:0] stream_proc_base;
  reg [8:0] stream_fill_count;
  reg [8:0] stream_expected_words;
  reg [3:0] stream_proc_group;
  reg [1:0] stream_proc_phase;
  reg [31:0] stream_lr_total;
  reg [31:0] stream_lr_index;
  reg [31:0] stream_out_row;
  reg [31:0] stream_out_col;
  reg [31:0] post_cb_row_base_next;
  reg [31:0] post_cb_x_base_next;
  reg [1:0] post_row_phase_next;
  reg read_pending;
  reg read_burst_mode;
  reg write_aw_pending;
  reg write_w_pending;
  reg write_burst_mode;
  reg [1:0] post_write_beat;

  wire [31:0] cfg_width32 = {16'd0, shape_reg[31:16]};
  wire [31:0] cfg_height32 = {16'd0, shape_reg[15:0]};
  wire [31:0] cfg_scale32 = scale_reg[3:0] == 4'd0 ? 32'd1 : {28'd0, scale_reg[3:0]};
  wire [31:0] scale32 = {28'd0, scale};

  reg [31:0] u8_to_float_rom [0:255];
  initial begin
    $readmemh("vsrc/u8_to_float_unit.mem", u8_to_float_rom);
  end

  wire stream_fill_buffer_ready = stream_fill_sel ? stream_buf_ready1 : stream_buf_ready0;
  wire stream_other_buffer_ready = stream_fill_sel ? stream_buf_ready0 : stream_buf_ready1;
  wire [8:0] stream_throttle_margin = scale == 4 ? 9'd64 : 9'd16;
  wire stream_almost_full = stream_other_buffer_ready &&
                            stream_expected_words > stream_throttle_margin &&
                            stream_fill_count >= stream_expected_words - stream_throttle_margin;
  wire stream_accepting = op == OP_TAIL_ADD_STREAM_FUSED &&
                          state != S_IDLE && state != S_DONE &&
                          state != S_SETUP && state != S_LAUNCH;
  wire [1:0] stream_valid_count = {1'b0, (stream_ch0_valid && stream_ready)} + {1'b0, (stream_ch1_valid && stream_ready)};
  assign stream_ready = stream_accepting && !stream_fill_buffer_ready;
  assign stream_throttle = stream_accepting && (stream_fill_buffer_ready || stream_almost_full);
  assign stream_active = stream_accepting;
  assign active = state != S_IDLE && state != S_DONE && state != S_STREAM_WAIT;
  assign axi_arid = 1'b0;
  assign axi_arsize = 3'd4;
  assign axi_arburst = 2'd1;
  assign axi_arlock = 1'b0;
  assign axi_arcache = 4'd0;
  assign axi_arprot = 3'd0;
  assign axi_arqos = 4'd0;
  assign axi_rready = 1'b1;
  assign axi_awid = 1'b0;
  assign axi_awsize = 3'd4;
  assign axi_awburst = 2'd1;
  assign axi_awlock = 1'b0;
  assign axi_awcache = 4'd0;
  assign axi_awprot = 3'd0;
  assign axi_awqos = 4'd0;
  assign axi_wid = 1'b0;
  assign axi_wstrb = 16'hffff;
  assign axi_bready = 1'b1;

  wire [511:0] tail_add_z_pack;
  genvar add_i;
  generate
    for (add_i = 0; add_i < 16; add_i = add_i + 1)
    begin: gen_tail_add
      FP32_Adder u_tail_add (
        .clock(clock),
        .reset(reset),
        .io_x(tail_add_x_pack[add_i * 32 +: 32]),
        .io_y(tail_add_y_pack[add_i * 32 +: 32]),
        .io_z(tail_add_z_pack[add_i * 32 +: 32]),
        .io_valid_in(tail_add_valid)
      );
    end
  endgenerate

  function [7:0] clip_u8_signed;
    input signed [31:0] v;
    reg signed [31:0] rounded;
    begin
      rounded = (v + 32'sd32768) >>> 16;
      if (rounded < 0)
        clip_u8_signed = 8'd0;
      else if (rounded > 255)
        clip_u8_signed = 8'd255;
      else
        clip_u8_signed = rounded[7:0];
    end
  endfunction

  function [7:0] bgr_byte_at;
    input [383:0] block;
    input integer index;
    reg [383:0] shifted;
    begin
      shifted = block >> (index * 8);
      bgr_byte_at = shifted[7:0];
    end
  endfunction

  function [31:0] y_word_at;
    input [511:0] block;
    input integer index;
    reg [511:0] shifted;
    begin
      shifted = block >> (index * 32);
      y_word_at = shifted[31:0];
    end
  endfunction

  function [7:0] byte_at_128;
    input [127:0] block;
    input [3:0] index;
    reg [127:0] shifted;
    begin
      shifted = block >> ({28'd0, index} * 32'd8);
      byte_at_128 = shifted[7:0];
    end
  endfunction

  function [31:0] word_at_128;
    input [127:0] block;
    input [1:0] index;
    reg [127:0] shifted;
    begin
      shifted = block >> ({30'd0, index} * 32'd32);
      word_at_128 = shifted[31:0];
    end
  endfunction

  function [63:0] tail_chan_byte_offset;
    input [3:0] chan;
    input [63:0] cstep_bytes;
    begin
      case (chan)
        4'd0: tail_chan_byte_offset = 64'd0;
        4'd1: tail_chan_byte_offset = cstep_bytes;
        4'd2: tail_chan_byte_offset = cstep_bytes << 1;
        4'd3: tail_chan_byte_offset = (cstep_bytes << 1) + cstep_bytes;
        4'd4: tail_chan_byte_offset = cstep_bytes << 2;
        4'd5: tail_chan_byte_offset = (cstep_bytes << 2) + cstep_bytes;
        4'd6: tail_chan_byte_offset = (cstep_bytes << 2) + (cstep_bytes << 1);
        4'd7: tail_chan_byte_offset = (cstep_bytes << 2) + (cstep_bytes << 1) + cstep_bytes;
        4'd8: tail_chan_byte_offset = cstep_bytes << 3;
        4'd9: tail_chan_byte_offset = (cstep_bytes << 3) + cstep_bytes;
        4'd10: tail_chan_byte_offset = (cstep_bytes << 3) + (cstep_bytes << 1);
        4'd11: tail_chan_byte_offset = (cstep_bytes << 3) + (cstep_bytes << 1) + cstep_bytes;
        4'd12: tail_chan_byte_offset = (cstep_bytes << 3) + (cstep_bytes << 2);
        4'd13: tail_chan_byte_offset = (cstep_bytes << 3) + (cstep_bytes << 2) + cstep_bytes;
        4'd14: tail_chan_byte_offset = (cstep_bytes << 3) + (cstep_bytes << 2) + (cstep_bytes << 1);
        default: tail_chan_byte_offset = (cstep_bytes << 3) + (cstep_bytes << 2) + (cstep_bytes << 1) + cstep_bytes;
      endcase
    end
  endfunction

  function [63:0] tail_src_addr;
    input [3:0] chan;
    input [31:0] elem_index;
    reg [63:0] cstep_bytes;
    begin
      cstep_bytes = {32'd0, tail_cstep} << 2;
      tail_src_addr = src0_addr + tail_chan_byte_offset(chan, cstep_bytes) + ({32'd0, elem_index} << 2);
    end
  endfunction

  function [511:0] tail_pack_x2;
    input [127:0] ch0_lo;
    input [127:0] ch0_hi;
    input [127:0] ch1_lo;
    input [127:0] ch1_hi;
    integer i;
    reg [3:0] pair_idx;
    reg [31:0] yword;
    reg [511:0] pack;
    begin
      pack = 512'd0;
      for (i = 0; i < 16; i = i + 1)
      begin
        pair_idx = {1'b0, i[3:1]};
        if (i[0] == 1'b0)
          yword = pair_idx[2] ? word_at_128(ch0_hi, pair_idx[1:0]) : word_at_128(ch0_lo, pair_idx[1:0]);
        else
          yword = pair_idx[2] ? word_at_128(ch1_hi, pair_idx[1:0]) : word_at_128(ch1_lo, pair_idx[1:0]);
        pack = pack | ({480'd0, yword} << (i * 32));
      end
      tail_pack_x2 = pack;
    end
  endfunction

  function [511:0] tail_pack_x4;
    input [127:0] ch0;
    input [127:0] ch1;
    input [127:0] ch2;
    input [127:0] ch3;
    integer i;
    reg [1:0] phase;
    reg [1:0] elem_idx;
    reg [31:0] yword;
    reg [511:0] pack;
    begin
      pack = 512'd0;
      for (i = 0; i < 16; i = i + 1)
      begin
        phase = i[1:0];
        elem_idx = i[3:2];
        case (phase)
          2'd0: yword = word_at_128(ch0, elem_idx);
          2'd1: yword = word_at_128(ch1, elem_idx);
          2'd2: yword = word_at_128(ch2, elem_idx);
          default: yword = word_at_128(ch3, elem_idx);
        endcase
        pack = pack | ({480'd0, yword} << (i * 32));
      end
      tail_pack_x4 = pack;
    end
  endfunction

  function [511:0] tail_skip_pack_x2;
    input [127:0] skip_lo;
    input [127:0] skip_hi;
    integer i;
    reg [3:0] pair_idx;
    reg [31:0] yword;
    reg [511:0] pack;
    begin
      pack = 512'd0;
      for (i = 0; i < 16; i = i + 1)
      begin
        pair_idx = {1'b0, i[3:1]};
        yword = pair_idx[2] ? word_at_128(skip_hi, pair_idx[1:0]) : word_at_128(skip_lo, pair_idx[1:0]);
        pack = pack | ({480'd0, yword} << (i * 32));
      end
      tail_skip_pack_x2 = pack;
    end
  endfunction

  function [511:0] tail_skip_pack_x4;
    input [127:0] skip;
    integer i;
    reg [1:0] elem_idx;
    reg [31:0] yword;
    reg [511:0] pack;
    begin
      pack = 512'd0;
      for (i = 0; i < 16; i = i + 1)
      begin
        elem_idx = i[3:2];
        yword = word_at_128(skip, elem_idx);
        pack = pack | ({480'd0, yword} << (i * 32));
      end
      tail_skip_pack_x4 = pack;
    end
  endfunction

  function [3:0] stream_phase_chan_base;
    input [1:0] phase;
    begin
      case (phase)
        2'd0: stream_phase_chan_base = 4'd0;
        2'd1: stream_phase_chan_base = 4'd2;
        2'd2: stream_phase_chan_base = 4'd8;
        default: stream_phase_chan_base = 4'd10;
      endcase
    end
  endfunction

  function [63:0] stream_rgb_addr;
    input [31:0] lr_index;
    input [1:0] phase;
    reg [31:0] lr_row;
    reg [31:0] lr_col;
    reg [31:0] out_index;
    reg [31:0] width32;
    begin
      width32 = {16'd0, width};
      lr_row = width32 == 32'd0 ? 32'd0 : lr_index / width32;
      lr_col = width32 == 32'd0 ? 32'd0 : lr_index - lr_row * width32;
      if (scale == 4)
        out_index = ((lr_row << 2) + {30'd0, phase}) * out_w + (lr_col << 2);
      else
        out_index = ((lr_row << 1) + {31'd0, phase[0]}) * out_w + (lr_col << 1);
      stream_rgb_addr = dst0_addr + ({32'd0, out_index} << 1) + {32'd0, out_index};
    end
  endfunction

  function [63:0] stream_rgb_addr_rc;
    input [31:0] lr_row;
    input [31:0] lr_col;
    input [1:0] phase;
    reg [31:0] out_index;
    begin
      if (scale == 4)
        out_index = ((lr_row << 2) + {30'd0, phase}) * out_w + (lr_col << 2);
      else
        out_index = ((lr_row << 1) + {31'd0, phase[0]}) * out_w + (lr_col << 1);
      stream_rgb_addr_rc = dst0_addr + ({32'd0, out_index} << 1) + {32'd0, out_index};
    end
  endfunction

  function [31:0] stream_group_lr_index;
    input [31:0] base;
    input [3:0] group;
    begin
      stream_group_lr_index = base + {26'd0, group, 2'b00};
    end
  endfunction

  function [7:0] float_unit_to_u8_bits;
    input [31:0] bits;
    reg sign;
    reg [7:0] exp;
    reg [23:0] mant;
    reg [39:0] prod;
    integer shift;
    reg [39:0] rounded;
    begin
      sign = bits[31];
      exp = bits[30:23];
      mant = {1'b1, bits[22:0]};
      if (sign || exp == 8'd0 || exp < 8'd119)
      begin
        float_unit_to_u8_bits = 8'd0;
      end
      else if (exp >= 8'd127)
      begin
        float_unit_to_u8_bits = 8'd255;
      end
      else
      begin
        prod = mant * 8'd255;
        shift = 32'sd150 - $signed({24'd0, exp});
        rounded = (prod + (40'd1 << (shift - 1))) >> shift;
        if (rounded > 40'd255)
          float_unit_to_u8_bits = 8'd255;
        else
          float_unit_to_u8_bits = rounded[7:0];
      end
    end
  endfunction

  function burst_cross_4k;
    input [63:0] addr;
    input [7:0] len;
    reg [12:0] end_offset;
    begin
      end_offset = {1'b0, addr[11:0]} + ({5'd0, len} << 4) + 13'd15;
      burst_cross_4k = end_offset[12];
    end
  endfunction

  function [7:0] ycbcr_b;
    input [7:0] yv;
    input [7:0] cbv;
    reg signed [31:0] yy;
    reg signed [31:0] cb;
    begin
      yy = $signed({8'd0, yv, 16'd0});
      cb = $signed({24'd0, cbv}) - 32'sd128;
      ycbcr_b = clip_u8_signed(yy + 32'sd116130 * cb);
    end
  endfunction

  function [7:0] ycbcr_g;
    input [7:0] yv;
    input [7:0] cbv;
    input [7:0] crv;
    reg signed [31:0] yy;
    reg signed [31:0] cb;
    reg signed [31:0] cr;
    begin
      yy = $signed({8'd0, yv, 16'd0});
      cb = $signed({24'd0, cbv}) - 32'sd128;
      cr = $signed({24'd0, crv}) - 32'sd128;
      ycbcr_g = clip_u8_signed(yy - 32'sd22554 * cb - 32'sd46802 * cr);
    end
  endfunction

  function [7:0] ycbcr_r;
    input [7:0] yv;
    input [7:0] crv;
    reg signed [31:0] yy;
    reg signed [31:0] cr;
    begin
      yy = $signed({8'd0, yv, 16'd0});
      cr = $signed({24'd0, crv}) - 32'sd128;
      ycbcr_r = clip_u8_signed(yy + 32'sd91881 * cr);
    end
  endfunction

  function [95:0] post_bgr_four;
    input [3:0] base;
    integer i;
    reg [3:0] idx;
    reg [31:0] coff;
    reg [7:0] py;
    reg [7:0] pcb;
    reg [7:0] pcr;
    reg [95:0] pack;
    begin
      pack = 96'd0;
      for (i = 0; i < 4; i = i + 1)
      begin
        idx = base + i[3:0];
        if (scale == 4)
          coff = {28'd0, cb_offset_base} + ({28'd0, idx} >> 2);
        else
          coff = {28'd0, cb_offset_base} + ({28'd0, idx} >> 1);
        py = float_unit_to_u8_bits(y_word_at(post_y_blk, {28'd0, idx}));
        pcb = byte_at_128(cb_blk, coff[3:0]);
        pcr = byte_at_128(cr_blk, coff[3:0]);
        pack = pack | ({88'd0, ycbcr_b(py, pcb)} << ((i * 3 + 0) * 8));
        pack = pack | ({88'd0, ycbcr_g(py, pcb, pcr)} << ((i * 3 + 1) * 8));
        pack = pack | ({88'd0, ycbcr_r(py, pcr)} << ((i * 3 + 2) * 8));
      end
      post_bgr_four = pack;
    end
  endfunction

  task start_read;
    input [63:0] addr;
    begin
      axi_araddr <= addr;
      axi_arlen <= 8'd0;
      axi_arvalid <= 1'b1;
      read_pending <= 1'b1;
      read_burst_mode <= 1'b0;
    end
  endtask

  task start_read_burst;
    input [63:0] addr;
    input [7:0] len;
    begin
      axi_araddr <= addr;
      axi_arlen <= len;
      axi_arvalid <= 1'b1;
      read_pending <= 1'b1;
      read_burst_mode <= 1'b1;
    end
  endtask

  task start_read_burst_safe;
    input [63:0] addr;
    input [7:0] len;
    begin
      if (burst_cross_4k(addr, len))
        start_read(addr);
      else
        start_read_burst(addr, len);
    end
  endtask

  task start_write;
    input [63:0] addr;
    input [127:0] data;
    begin
      axi_awaddr <= addr;
      axi_awlen <= 8'd0;
      axi_wdata <= data;
      axi_wlast <= 1'b1;
      axi_awvalid <= 1'b1;
      axi_wvalid <= 1'b1;
      write_aw_pending <= 1'b1;
      write_w_pending <= 1'b1;
      write_burst_mode <= 1'b0;
    end
  endtask

  task start_write_burst;
    input [63:0] addr;
    input [127:0] data;
    input [7:0] len;
    input last;
    begin
      axi_awaddr <= addr;
      axi_awlen <= len;
      axi_wdata <= data;
      axi_wlast <= last;
      axi_awvalid <= 1'b1;
      axi_wvalid <= 1'b1;
      write_aw_pending <= 1'b1;
      write_w_pending <= 1'b1;
      write_burst_mode <= 1'b1;
    end
  endtask

  task start_write_burst_safe;
    input [63:0] addr;
    input [127:0] data;
    input [7:0] len;
    input last;
    begin
      if (burst_cross_4k(addr, len))
        start_write(addr, data);
      else
        start_write_burst(addr, data, len, last);
    end
  endtask

  task send_write_beat;
    input [127:0] data;
    input last;
    begin
      axi_wdata <= data;
      axi_wlast <= last;
      axi_wvalid <= 1'b1;
      write_w_pending <= 1'b1;
    end
  endtask

integer i;
    reg [7:0] bv;
    reg [7:0] gv;
    reg [7:0] rv;
    reg signed [31:0] yyv;
    reg signed [31:0] cbv;
    reg signed [31:0] crv;
    reg [7:0] yu8;
    reg [7:0] cbu8;
    reg [7:0] cru8;
  always @* begin
    

    y_pack = 128'd0;
    cb_pack = 128'd0;
    cr_pack = 128'd0;
    y_float_pack0 = 128'd0;
    y_float_pack1 = 128'd0;
    y_float_pack2 = 128'd0;
    y_float_pack3 = 128'd0;
    bgr_pack = 384'd0;

    for (i = 0; i < 16; i = i + 1)
    begin
      bv = bgr_byte_at(pre_bgr_blk, i * 3 + 0);
      gv = bgr_byte_at(pre_bgr_blk, i * 3 + 1);
      rv = bgr_byte_at(pre_bgr_blk, i * 3 + 2);
      yyv = 32'sd19595 * $signed({24'd0, rv}) + 32'sd38470 * $signed({24'd0, gv}) + 32'sd7471 * $signed({24'd0, bv});
      cbv = 32'sd8388608 - 32'sd11058 * $signed({24'd0, rv}) - 32'sd21710 * $signed({24'd0, gv}) + 32'sd32768 * $signed({24'd0, bv});
      crv = 32'sd8388608 + 32'sd32768 * $signed({24'd0, rv}) - 32'sd27439 * $signed({24'd0, gv}) - 32'sd5329 * $signed({24'd0, bv});
      yu8 = clip_u8_signed(yyv);
      cbu8 = clip_u8_signed(cbv);
      cru8 = clip_u8_signed(crv);
      y_pack = y_pack | ({120'd0, yu8} << (i * 8));
      cb_pack = cb_pack | ({120'd0, cbu8} << (i * 8));
      cr_pack = cr_pack | ({120'd0, cru8} << (i * 8));
      if (i < 4)
        y_float_pack0 = y_float_pack0 | ({96'd0, u8_to_float_rom[yu8]} << (i * 32));
      else if (i < 8)
        y_float_pack1 = y_float_pack1 | ({96'd0, u8_to_float_rom[yu8]} << ((i - 4) * 32));
      else if (i < 12)
        y_float_pack2 = y_float_pack2 | ({96'd0, u8_to_float_rom[yu8]} << ((i - 8) * 32));
      else
        y_float_pack3 = y_float_pack3 | ({96'd0, u8_to_float_rom[yu8]} << ((i - 12) * 32));
    end
  end

  always @(posedge clock) begin
    if (reset)
    begin
      state <= S_IDLE;
      task_done <= 1'b0;
      axi_arvalid <= 1'b0;
      axi_awvalid <= 1'b0;
      axi_wvalid <= 1'b0;
      axi_arlen <= 8'd0;
      axi_awlen <= 8'd0;
      axi_wlast <= 1'b1;
      read_pending <= 1'b0;
      read_burst_mode <= 1'b0;
      write_aw_pending <= 1'b0;
      write_w_pending <= 1'b0;
      write_burst_mode <= 1'b0;
      post_write_beat <= 2'd0;
      perf_cycles <= 64'd0;
      perf_read_beats <= 32'd0;
      perf_write_beats <= 32'd0;
      perf_pixels <= 32'd0;
      perf_axi_wait_cycles <= 64'd0;
      perf_tiles <= 32'd0;
      total_pixels <= 32'd0;
      lr_pixels <= 32'd0;
      pixel_base <= 32'd0;
      pre_bgr_blk <= 384'd0;
      post_y_blk <= 512'd0;
      cb_blk <= 128'd0;
      cr_blk <= 128'd0;
      out_w <= 32'd0;
      cb_addr_aligned <= 32'd0;
      cb_offset_base <= 4'd0;
      pre_rgb_addr <= 64'd0;
      pre_y_addr <= 64'd0;
      pre_cb_addr <= 64'd0;
      pre_cr_addr <= 64'd0;
      pre_y_float_addr <= 64'd0;
      post_y_addr <= 64'd0;
      post_rgb_addr <= 64'd0;
      post_out_x <= 32'd0;
      post_row_phase <= 2'd0;
      post_cb_row_base <= 32'd0;
      post_cb_x_base <= 32'd0;
      cbcr_cache_valid <= 1'b0;
      cbcr_cache_addr <= 32'd0;
      pre_y_pack_q <= 128'd0;
      pre_cb_pack_q <= 128'd0;
      pre_cr_pack_q <= 128'd0;
      pre_y_float_pack0_q <= 128'd0;
      pre_y_float_pack1_q <= 128'd0;
      pre_y_float_pack2_q <= 128'd0;
      pre_y_float_pack3_q <= 128'd0;
      post_bgr_pack_q <= 384'd0;
      pre_save_y <= 1'b0;
      tail_cstep <= 32'd0;
      tail_addr0 <= 64'd0;
      tail_addr1 <= 64'd0;
      tail_addr2 <= 64'd0;
      tail_addr3 <= 64'd0;
      tail_src0 <= 128'd0;
      tail_src1 <= 128'd0;
      tail_src2 <= 128'd0;
      tail_src3 <= 128'd0;
      tail_skip_addr0 <= 64'd0;
      tail_skip_addr1 <= 64'd0;
      tail_skip0 <= 128'd0;
      tail_skip1 <= 128'd0;
      tail_raw_pack <= 512'd0;
      tail_skip_pack <= 512'd0;
      tail_add_x_pack <= 512'd0;
      tail_add_y_pack <= 512'd0;
      tail_add_valid <= 1'b0;
      tail_add_wait <= 4'd0;
      stream_fill_sel <= 1'b0;
      stream_proc_sel <= 1'b0;
      stream_buf_ready0 <= 1'b0;
      stream_buf_ready1 <= 1'b0;
      stream_buf_base0 <= 32'd0;
      stream_buf_base1 <= 32'd0;
      stream_fill_base <= 32'd0;
      stream_proc_base <= 32'd0;
      stream_fill_count <= 9'd0;
      stream_expected_words <= 9'd0;
      stream_proc_group <= 4'd0;
      stream_proc_phase <= 2'd0;
      stream_lr_total <= 32'd0;
      stream_lr_index <= 32'd0;
      stream_out_row <= 32'd0;
      stream_out_col <= 32'd0;
      post_cb_row_base_next <= 32'd0;
      post_cb_x_base_next <= 32'd0;
      post_row_phase_next <= 2'd0;
    end
    else
    begin
      if (stream_ready && stream_ch0_valid)
      begin
        if (stream_fill_sel)
          stream_buf1[{stream_ch0_oc[3:0], stream_ch0_group}] <= stream_ch0_data;
        else
          stream_buf0[{stream_ch0_oc[3:0], stream_ch0_group}] <= stream_ch0_data;
      end
      if (stream_ready && stream_ch1_valid)
      begin
        if (stream_fill_sel)
          stream_buf1[{stream_ch1_oc[3:0], stream_ch1_group}] <= stream_ch1_data;
        else
          stream_buf0[{stream_ch1_oc[3:0], stream_ch1_group}] <= stream_ch1_data;
      end
      if (stream_accepting && stream_valid_count != 2'd0)
      begin
        if (stream_fill_count + {7'd0, stream_valid_count} >= stream_expected_words)
        begin
          if (stream_fill_sel)
          begin
            stream_buf_ready1 <= 1'b1;
            stream_buf_base1 <= stream_fill_base;
          end
          else
          begin
            stream_buf_ready0 <= 1'b1;
            stream_buf_base0 <= stream_fill_base;
          end
          stream_fill_sel <= ~stream_fill_sel;
          stream_fill_base <= stream_fill_base + 32'd64;
          stream_fill_count <= 9'd0;
        end
        else
          stream_fill_count <= stream_fill_count + {7'd0, stream_valid_count};
      end

      if (state != S_IDLE && state != S_DONE)
        perf_cycles <= perf_cycles + 64'd1;
      if ((axi_arvalid && !axi_arready) || (axi_awvalid && !axi_awready) || (axi_wvalid && !axi_wready))
        perf_axi_wait_cycles <= perf_axi_wait_cycles + 64'd1;
      if (axi_rvalid && axi_rready)
        perf_read_beats <= perf_read_beats + 32'd1;
      if (axi_wvalid && axi_wready)
        perf_write_beats <= perf_write_beats + 32'd1;
      if (axi_arvalid && axi_arready)
        axi_arvalid <= 1'b0;
      if (axi_awvalid && axi_awready)
      begin
        axi_awvalid <= 1'b0;
        write_aw_pending <= 1'b0;
      end
      if (axi_wvalid && axi_wready)
      begin
        axi_wvalid <= 1'b0;
        write_w_pending <= 1'b0;
      end

      case (state)
        S_IDLE:
        begin
          task_done <= 1'b0;
          if (ctrl_reg[0])
          begin
            op <= ctrl_reg[3:1];
            width <= shape_reg[31:16];
            height <= shape_reg[15:0];
            scale <= cfg_scale32[3:0];
            src0_addr <= src0_addr_reg;
            src1_addr <= src1_addr_reg;
            src2_addr <= src2_addr_reg;
            dst0_addr <= dst0_addr_reg;
            dst1_addr <= dst1_addr_reg;
            dst2_addr <= dst2_addr_reg;
            dst3_addr <= dst3_addr_reg;
            skip_y_addr <= dst1_addr_reg;
            pixel_base <= 32'd0;
            pre_rgb_addr <= src0_addr_reg;
            pre_y_addr <= dst0_addr_reg;
            pre_cb_addr <= dst1_addr_reg;
            pre_cr_addr <= dst2_addr_reg;
            pre_y_float_addr <= dst3_addr_reg;
            pre_save_y <= dst0_addr_reg != 64'd0;
            post_y_addr <= src0_addr_reg;
            post_rgb_addr <= dst0_addr_reg;
            post_out_x <= 32'd0;
            post_row_phase <= 2'd0;
            post_cb_row_base <= 32'd0;
            post_cb_x_base <= 32'd0;
            stream_fill_sel <= 1'b0;
            stream_proc_sel <= 1'b0;
            stream_buf_ready0 <= 1'b0;
            stream_buf_ready1 <= 1'b0;
            stream_fill_base <= 32'd0;
            stream_proc_base <= 32'd0;
            stream_fill_count <= 9'd0;
            stream_proc_group <= 4'd0;
            stream_proc_phase <= 2'd0;
            stream_lr_index <= 32'd0;
            stream_out_row <= 32'd0;
            stream_out_col <= 32'd0;
            post_write_beat <= 2'd0;
            cbcr_cache_valid <= 1'b0;
            cbcr_cache_addr <= 32'd0;
            cb_addr_aligned <= 32'd0;
            cb_offset_base <= 4'd0;
            perf_cycles <= 64'd0;
            perf_read_beats <= 32'd0;
            perf_write_beats <= 32'd0;
            perf_axi_wait_cycles <= 64'd0;
            perf_tiles <= 32'd0;
            state <= S_SETUP;
          end
        end

        S_SETUP:
        begin
          lr_pixels <= {16'd0, width} * {16'd0, height};
          state <= S_LAUNCH;
        end

        S_LAUNCH:
        begin
          tail_cstep <= (lr_pixels + 32'd7) & 32'hfffffff8;
          stream_lr_total <= lr_pixels;
          case (scale)
            4'd4:
            begin
              out_w <= {16'd0, width} << 2;
              total_pixels <= lr_pixels << 4;
              perf_pixels <= lr_pixels << 4;
              stream_expected_words <= 9'd256;
            end
            4'd2:
            begin
              out_w <= {16'd0, width} << 1;
              total_pixels <= lr_pixels << 2;
              perf_pixels <= lr_pixels << 2;
              stream_expected_words <= 9'd64;
            end
            default:
            begin
              out_w <= {16'd0, width};
              total_pixels <= lr_pixels;
              perf_pixels <= lr_pixels;
              stream_expected_words <= 9'd16;
            end
          endcase

          if (op == OP_SPLIT)
          begin
            total_pixels <= lr_pixels;
            perf_pixels <= lr_pixels;
            state <= S_DONE;
          end
          else if (op == OP_POSTPROCESS)
          begin
            start_read_burst_safe(src0_addr, 8'd3);
            state <= S_POST_RY0;
          end
          else if (op == OP_TAIL_FUSED)
            state <= S_TAIL_ADDR;
          else if (op == OP_TAIL_ADD_FUSED)
            state <= S_TAIL_ADDR;
          else if (op == OP_TAIL_ADD_STREAM_FUSED)
            state <= S_STREAM_WAIT;
          else
          begin
            total_pixels <= lr_pixels;
            perf_pixels <= lr_pixels;
            if (scale == 4'd4)
            begin
              out_w <= {16'd0, width} << 2;
              stream_expected_words <= 9'd256;
            end
            else if (scale == 4'd2)
            begin
              out_w <= {16'd0, width} << 1;
              stream_expected_words <= 9'd64;
            end
            else
            begin
              out_w <= {16'd0, width};
              stream_expected_words <= 9'd16;
            end
            start_read_burst_safe(src0_addr, 8'd2);
            state <= S_PRE_R0;
          end
        end

        S_PRE_R0:
          if (read_pending && axi_rvalid)
          begin
            pre_bgr_blk[127:0] <= axi_rdata;
            if (!read_burst_mode)
              start_read(pre_rgb_addr + 64'd16);
            state <= S_PRE_R1;
          end
        S_PRE_R1:
          if (read_pending && axi_rvalid)
          begin
            pre_bgr_blk[255:128] <= axi_rdata;
            if (!read_burst_mode)
              start_read(pre_rgb_addr + 64'd32);
            state <= S_PRE_R2;
          end
        S_PRE_R2:
          if (read_pending && axi_rvalid)
          begin
            pre_bgr_blk[383:256] <= axi_rdata;
            read_pending <= 1'b0;
            state <= S_PRE_CALC;
          end
        S_PRE_CALC:
        begin
          perf_tiles <= perf_tiles + 32'd1;
          pre_y_pack_q <= y_pack;
          pre_cb_pack_q <= cb_pack;
          pre_cr_pack_q <= cr_pack;
          pre_y_float_pack0_q <= y_float_pack0;
          pre_y_float_pack1_q <= y_float_pack1;
          pre_y_float_pack2_q <= y_float_pack2;
          pre_y_float_pack3_q <= y_float_pack3;
          if (pre_save_y)
          begin
            start_write(pre_y_addr, y_pack);
            state <= S_PRE_WY;
          end
          else
          begin
            start_write(pre_cb_addr, cb_pack);
            state <= S_PRE_WCB;
          end
        end
        S_PRE_WY:
          if (!write_aw_pending && !write_w_pending && axi_bvalid)
          begin
            start_write(pre_cb_addr, pre_cb_pack_q);
            state <= S_PRE_WCB;
          end
        S_PRE_WCB:
          if (!write_aw_pending && !write_w_pending && axi_bvalid)
          begin
            start_write(pre_cr_addr, pre_cr_pack_q);
            state <= S_PRE_WCR;
          end
        S_PRE_WCR:
          if (!write_aw_pending && !write_w_pending && axi_bvalid)
          begin
            start_write_burst_safe(pre_y_float_addr, pre_y_float_pack0_q, 8'd3, 1'b0);
            state <= S_PRE_WF0;
          end
        S_PRE_WF0:
          if (write_burst_mode && !write_aw_pending && !write_w_pending)
          begin
            send_write_beat(pre_y_float_pack1_q, 1'b0);
            state <= S_PRE_WF1;
          end
          else if (!write_burst_mode && !write_aw_pending && !write_w_pending && axi_bvalid)
          begin
            start_write(pre_y_float_addr + 64'd16, pre_y_float_pack1_q);
            state <= S_PRE_WF1;
          end
        S_PRE_WF1:
          if (write_burst_mode && !write_aw_pending && !write_w_pending)
          begin
            send_write_beat(pre_y_float_pack2_q, 1'b0);
            state <= S_PRE_WF2;
          end
          else if (!write_burst_mode && !write_aw_pending && !write_w_pending && axi_bvalid)
          begin
            start_write(pre_y_float_addr + 64'd32, pre_y_float_pack2_q);
            state <= S_PRE_WF2;
          end
        S_PRE_WF2:
          if (write_burst_mode && !write_aw_pending && !write_w_pending)
          begin
            send_write_beat(pre_y_float_pack3_q, 1'b1);
            state <= S_PRE_WF3;
          end
          else if (!write_burst_mode && !write_aw_pending && !write_w_pending && axi_bvalid)
          begin
            start_write(pre_y_float_addr + 64'd48, pre_y_float_pack3_q);
            state <= S_PRE_WF3;
          end
        S_PRE_WF3:
          if (!write_aw_pending && !write_w_pending && axi_bvalid)
          begin
            if (pixel_base + 32'd16 >= total_pixels)
              state <= S_DONE;
            else
            begin
              pixel_base <= pixel_base + 32'd16;
              pre_rgb_addr <= pre_rgb_addr + 64'd48;
              pre_y_addr <= pre_y_addr + 64'd16;
              pre_cb_addr <= pre_cb_addr + 64'd16;
              pre_cr_addr <= pre_cr_addr + 64'd16;
              pre_y_float_addr <= pre_y_float_addr + 64'd64;
              start_read_burst_safe(pre_rgb_addr + 64'd48, 8'd2);
              state <= S_PRE_R0;
            end
          end

        S_POST_RY0:
          if (read_pending && axi_rvalid)
          begin
            post_y_blk[127:0] <= axi_rdata;
            if (!read_burst_mode)
              start_read(post_y_addr + 64'd16);
            state <= S_POST_RY1;
          end
        S_POST_RY1:
          if (read_pending && axi_rvalid)
          begin
            post_y_blk[255:128] <= axi_rdata;
            if (!read_burst_mode)
              start_read(post_y_addr + 64'd32);
            state <= S_POST_RY2;
          end
        S_POST_RY2:
          if (read_pending && axi_rvalid)
          begin
            post_y_blk[383:256] <= axi_rdata;
            if (!read_burst_mode)
              start_read(post_y_addr + 64'd48);
            state <= S_POST_RY3;
          end
        S_POST_RY3:
          if (read_pending && axi_rvalid)
          begin
            post_y_blk[511:384] <= axi_rdata;
            read_pending <= 1'b0;
            state <= S_POST_ADDR;
          end
        S_POST_ADDR:
        begin
          if (op == OP_TAIL_ADD_STREAM_FUSED)
            cb_index_base <= stream_lr_index;
          else
            cb_index_base <= post_cb_row_base + post_cb_x_base;
          state <= S_POST_RCB_REQ;
        end
        S_POST_RCB_REQ:
        begin
          cb_addr_aligned <= cb_index_base & 32'hfffffff0;
          cb_offset_base <= cb_index_base[3:0];
          if (cbcr_cache_valid && cbcr_cache_addr == (cb_index_base & 32'hfffffff0))
            state <= S_POST_CALC;
          else
          begin
            start_read(src1_addr + {32'd0, (cb_index_base & 32'hfffffff0)});
            state <= S_POST_RCB;
          end
        end
        S_POST_RCB:
          if (read_pending && axi_rvalid)
          begin
            cb_blk <= axi_rdata;
            read_pending <= 1'b0;
            start_read(src2_addr + {32'd0, cb_addr_aligned});
            state <= S_POST_RCR;
          end
        S_POST_RCR:
          if (read_pending && axi_rvalid)
          begin
            cr_blk <= axi_rdata;
            cbcr_cache_valid <= 1'b1;
            cbcr_cache_addr <= cb_addr_aligned;
            read_pending <= 1'b0;
            state <= S_POST_CALC;
          end
        S_POST_CALC:
        begin
          post_bgr_pack_q <= 384'd0;
          state <= S_POST_CALC0;
        end
        S_POST_CALC0:
        begin
          post_bgr_pack_q[95:0] <= post_bgr_four(4'd0);
          state <= S_POST_CALC1;
        end
        S_POST_CALC1:
        begin
          post_bgr_pack_q[191:96] <= post_bgr_four(4'd4);
          state <= S_POST_CALC2;
        end
        S_POST_CALC2:
        begin
          post_bgr_pack_q[287:192] <= post_bgr_four(4'd8);
          state <= S_POST_CALC3;
        end
        S_POST_CALC3:
        begin
          post_bgr_pack_q[383:288] <= post_bgr_four(4'd12);
          perf_tiles <= perf_tiles + 32'd1;
          state <= S_POST_W0;
        end
        S_POST_W0:
        begin
          start_write_burst_safe(post_rgb_addr, post_bgr_pack_q[127:0], 8'd2, 1'b0);
          post_write_beat <= 2'd1;
          state <= S_POST_W1;
        end
        S_POST_W1:
          if (write_burst_mode && !write_aw_pending && !write_w_pending)
          begin
            send_write_beat(post_bgr_pack_q[255:128], 1'b0);
            post_write_beat <= 2'd2;
            state <= S_POST_W2;
          end
          else if (!write_burst_mode && !write_aw_pending && !write_w_pending && axi_bvalid)
          begin
            start_write(post_rgb_addr + 64'd16, post_bgr_pack_q[255:128]);
            post_write_beat <= 2'd2;
            state <= S_POST_W2;
          end
        S_POST_W2:
          if (write_burst_mode && !write_aw_pending && !write_w_pending)
          begin
            send_write_beat(post_bgr_pack_q[383:256], 1'b1);
            post_write_beat <= 2'd3;
            state <= S_POST_W3;
          end
          else if (!write_burst_mode && !write_aw_pending && !write_w_pending && axi_bvalid)
          begin
            start_write(post_rgb_addr + 64'd32, post_bgr_pack_q[383:256]);
            post_write_beat <= 2'd3;
            state <= S_POST_W3;
          end
        S_POST_W3:
          if (!write_aw_pending && !write_w_pending && axi_bvalid)
          begin
            post_write_beat <= 2'd0;
            if (pixel_base + 32'd16 >= total_pixels)
              state <= S_DONE;
            else
            begin
              pixel_base <= pixel_base + 32'd16;
              post_y_addr <= post_y_addr + 64'd64;
              post_rgb_addr <= post_rgb_addr + 64'd48;
              if (post_out_x + 32'd16 >= out_w)
              begin
                post_out_x <= 32'd0;
                post_cb_x_base <= 32'd0;
                if (scale == 4)
                begin
                  if (post_row_phase == 2'd3)
                  begin
                    post_row_phase <= 2'd0;
                    post_cb_row_base <= post_cb_row_base + {16'd0, width};
                  end
                  else
                    post_row_phase <= post_row_phase + 2'd1;
                end
                else
                begin
                  if (post_row_phase[0])
                  begin
                    post_row_phase <= 2'd0;
                    post_cb_row_base <= post_cb_row_base + {16'd0, width};
                  end
                  else
                    post_row_phase <= post_row_phase + 2'd1;
                end
              end
              else
              begin
                post_out_x <= post_out_x + 32'd16;
                if (scale == 4)
                  post_cb_x_base <= post_cb_x_base + 32'd4;
                else
                  post_cb_x_base <= post_cb_x_base + 32'd8;
              end
              if (op == OP_TAIL_ADD_STREAM_FUSED)
              begin
                if (scale == 4)
                begin
                  if (stream_proc_phase == 2'd3)
                  begin
                    stream_proc_phase <= 2'd0;
                    if (stream_proc_group == 4'd15)
                    begin
                      if (stream_proc_sel)
                        stream_buf_ready1 <= 1'b0;
                      else
                        stream_buf_ready0 <= 1'b0;
                      stream_proc_sel <= ~stream_proc_sel;
                      stream_proc_group <= 4'd0;
                    end
                    else
                      stream_proc_group <= stream_proc_group + 4'd1;
                    stream_lr_index <= stream_lr_index + 32'd4;
                    if (stream_out_col + 32'd4 >= {16'd0, width})
                    begin
                      stream_out_col <= 32'd0;
                      stream_out_row <= stream_out_row + 32'd1;
                    end
                    else
                      stream_out_col <= stream_out_col + 32'd4;
                  end
                  else
                    stream_proc_phase <= stream_proc_phase + 2'd1;
                end
                else
                begin
                  if (stream_proc_phase[0])
                  begin
                    stream_proc_phase <= 2'd0;
                    if (stream_proc_group == 4'd14)
                    begin
                      if (stream_proc_sel)
                        stream_buf_ready1 <= 1'b0;
                      else
                        stream_buf_ready0 <= 1'b0;
                      stream_proc_sel <= ~stream_proc_sel;
                      stream_proc_group <= 4'd0;
                    end
                    else
                      stream_proc_group <= stream_proc_group + 4'd2;
                    stream_lr_index <= stream_lr_index + 32'd8;
                    if (stream_out_col + 32'd8 >= {16'd0, width})
                    begin
                      stream_out_col <= 32'd0;
                      stream_out_row <= stream_out_row + 32'd1;
                    end
                    else
                      stream_out_col <= stream_out_col + 32'd8;
                  end
                  else
                    stream_proc_phase <= 2'd1;
                end
                if (pixel_base + 32'd16 >= total_pixels)
                  state <= S_DONE;
                else
                  state <= S_STREAM_WAIT;
              end
              else if (op == OP_TAIL_FUSED || op == OP_TAIL_ADD_FUSED)
                state <= S_TAIL_ADDR;
              else
                state <= S_POST_NEXT;
            end
          end

        S_POST_NEXT:
        begin
          start_read_burst_safe(post_y_addr, 8'd3);
          state <= S_POST_RY0;
        end

        S_STREAM_WAIT:
        begin
          if (stream_proc_sel ? stream_buf_ready1 : stream_buf_ready0)
            state <= S_STREAM_PACK;
        end

        S_STREAM_PACK:
        begin
          post_rgb_addr <= stream_rgb_addr_rc(stream_out_row, stream_out_col, stream_proc_phase);
          tail_skip_addr0 <= skip_y_addr + ({32'd0, stream_lr_index} << 2);
          tail_skip_addr1 <= skip_y_addr + ({32'd0, stream_lr_index + 32'd4} << 2);
          if (scale == 4)
          begin
            if (stream_proc_sel)
              tail_raw_pack <= tail_pack_x4(
                stream_buf1[{stream_phase_chan_base(stream_proc_phase) + 4'd0, stream_proc_group}],
                stream_buf1[{stream_phase_chan_base(stream_proc_phase) + 4'd1, stream_proc_group}],
                stream_buf1[{stream_phase_chan_base(stream_proc_phase) + 4'd4, stream_proc_group}],
                stream_buf1[{stream_phase_chan_base(stream_proc_phase) + 4'd5, stream_proc_group}]
              );
            else
              tail_raw_pack <= tail_pack_x4(
                stream_buf0[{stream_phase_chan_base(stream_proc_phase) + 4'd0, stream_proc_group}],
                stream_buf0[{stream_phase_chan_base(stream_proc_phase) + 4'd1, stream_proc_group}],
                stream_buf0[{stream_phase_chan_base(stream_proc_phase) + 4'd4, stream_proc_group}],
                stream_buf0[{stream_phase_chan_base(stream_proc_phase) + 4'd5, stream_proc_group}]
              );
            start_read(skip_y_addr + ({32'd0, stream_lr_index} << 2));
            state <= S_TAIL_SKIP0;
          end
          else
          begin
            if (stream_proc_sel)
            begin
              if (stream_proc_phase[0])
                tail_raw_pack <= tail_pack_x2(
                  stream_buf1[{4'd2, stream_proc_group}],
                  stream_buf1[{4'd2, stream_proc_group + 4'd1}],
                  stream_buf1[{4'd3, stream_proc_group}],
                  stream_buf1[{4'd3, stream_proc_group + 4'd1}]
                );
              else
                tail_raw_pack <= tail_pack_x2(
                  stream_buf1[{4'd0, stream_proc_group}],
                  stream_buf1[{4'd0, stream_proc_group + 4'd1}],
                  stream_buf1[{4'd1, stream_proc_group}],
                  stream_buf1[{4'd1, stream_proc_group + 4'd1}]
                );
            end
            else
            begin
              if (stream_proc_phase[0])
                tail_raw_pack <= tail_pack_x2(
                  stream_buf0[{4'd2, stream_proc_group}],
                  stream_buf0[{4'd2, stream_proc_group + 4'd1}],
                  stream_buf0[{4'd3, stream_proc_group}],
                  stream_buf0[{4'd3, stream_proc_group + 4'd1}]
                );
              else
                tail_raw_pack <= tail_pack_x2(
                  stream_buf0[{4'd0, stream_proc_group}],
                  stream_buf0[{4'd0, stream_proc_group + 4'd1}],
                  stream_buf0[{4'd1, stream_proc_group}],
                  stream_buf0[{4'd1, stream_proc_group + 4'd1}]
                );
            end
            start_read(skip_y_addr + ({32'd0, stream_lr_index} << 2));
            state <= S_TAIL_SKIP0;
          end
        end

        S_TAIL_ADDR:
        begin
          if (scale == 4)
          begin
            tail_addr0 <= tail_src_addr({post_row_phase[1], 1'b0, post_row_phase[0], 1'b0}, post_cb_row_base + post_cb_x_base);
            tail_addr1 <= tail_src_addr({post_row_phase[1], 1'b0, post_row_phase[0], 1'b0} + 4'd1, post_cb_row_base + post_cb_x_base);
            tail_addr2 <= tail_src_addr({post_row_phase[1], 1'b0, post_row_phase[0], 1'b0} + 4'd4, post_cb_row_base + post_cb_x_base);
            tail_addr3 <= tail_src_addr({post_row_phase[1], 1'b0, post_row_phase[0], 1'b0} + 4'd5, post_cb_row_base + post_cb_x_base);
          end
          else if (post_row_phase[0])
          begin
            tail_addr0 <= tail_src_addr(4'd2, post_cb_row_base + post_cb_x_base);
            tail_addr1 <= tail_src_addr(4'd3, post_cb_row_base + post_cb_x_base);
            tail_addr2 <= tail_src_addr(4'd2, post_cb_row_base + post_cb_x_base + 32'd4);
            tail_addr3 <= tail_src_addr(4'd3, post_cb_row_base + post_cb_x_base + 32'd4);
          end
          else
          begin
            tail_addr0 <= tail_src_addr(4'd0, post_cb_row_base + post_cb_x_base);
            tail_addr1 <= tail_src_addr(4'd1, post_cb_row_base + post_cb_x_base);
            tail_addr2 <= tail_src_addr(4'd0, post_cb_row_base + post_cb_x_base + 32'd4);
            tail_addr3 <= tail_src_addr(4'd1, post_cb_row_base + post_cb_x_base + 32'd4);
          end
          if (op == OP_TAIL_ADD_FUSED)
          begin
            tail_skip_addr0 <= skip_y_addr + ({32'd0, post_cb_row_base + post_cb_x_base} << 2);
            tail_skip_addr1 <= skip_y_addr + ({32'd0, post_cb_row_base + post_cb_x_base + 32'd4} << 2);
          end
          state <= S_TAIL_R0;
        end

        S_TAIL_R0:
        begin
          start_read(tail_addr0);
          state <= S_TAIL_R1;
        end
        S_TAIL_R1:
          if (read_pending && axi_rvalid)
          begin
            tail_src0 <= axi_rdata;
            read_pending <= 1'b0;
            start_read(tail_addr1);
            state <= S_TAIL_R2;
          end
        S_TAIL_R2:
          if (read_pending && axi_rvalid)
          begin
            tail_src1 <= axi_rdata;
            read_pending <= 1'b0;
            start_read(tail_addr2);
            state <= S_TAIL_R3;
          end
        S_TAIL_R3:
          if (read_pending && axi_rvalid)
          begin
            tail_src2 <= axi_rdata;
            read_pending <= 1'b0;
            start_read(tail_addr3);
            state <= S_TAIL_PACK;
          end
        S_TAIL_PACK:
          if (read_pending && axi_rvalid)
          begin
            tail_src3 <= axi_rdata;
            read_pending <= 1'b0;
            if (op == OP_TAIL_ADD_FUSED)
            begin
              if (scale == 4)
                tail_raw_pack <= tail_pack_x4(tail_src0, tail_src1, tail_src2, axi_rdata);
              else
                tail_raw_pack <= tail_pack_x2(tail_src0, tail_src2, tail_src1, axi_rdata);
              start_read(tail_skip_addr0);
              state <= S_TAIL_SKIP0;
            end
            else
            begin
              if (scale == 4)
                post_y_blk <= tail_pack_x4(tail_src0, tail_src1, tail_src2, axi_rdata);
              else
                post_y_blk <= tail_pack_x2(tail_src0, tail_src2, tail_src1, axi_rdata);
              state <= S_POST_ADDR;
            end
          end

        S_TAIL_SKIP0:
          if (read_pending && axi_rvalid)
          begin
            tail_skip0 <= axi_rdata;
            read_pending <= 1'b0;
            if (scale == 4)
              state <= S_TAIL_ADD_PREP;
            else
            begin
              start_read(tail_skip_addr1);
              state <= S_TAIL_SKIP1;
            end
          end

        S_TAIL_SKIP1:
          if (read_pending && axi_rvalid)
          begin
            tail_skip1 <= axi_rdata;
            read_pending <= 1'b0;
            state <= S_TAIL_ADD_PREP;
          end

        S_TAIL_ADD_PREP:
        begin
          tail_add_x_pack <= tail_raw_pack;
          if (scale == 4)
            tail_add_y_pack <= tail_skip_pack_x4(tail_skip0);
          else
            tail_add_y_pack <= tail_skip_pack_x2(tail_skip0, tail_skip1);
          tail_add_valid <= 1'b0;
          tail_add_wait <= 4'd0;
          state <= S_TAIL_ADD_FIRE;
        end

        S_TAIL_ADD_FIRE:
        begin
          tail_add_valid <= 1'b1;
          tail_add_wait <= 4'd0;
          state <= S_TAIL_ADD_WAIT;
        end

        S_TAIL_ADD_WAIT:
        begin
          tail_add_valid <= 1'b0;
          tail_add_wait <= tail_add_wait + 4'd1;
          if (tail_add_wait >= 4'd5)
            state <= S_TAIL_ADD_PACK;
        end

        S_TAIL_ADD_PACK:
        begin
          post_y_blk <= tail_add_z_pack;
          state <= S_POST_ADDR;
        end

        S_DONE:
        begin
          task_done <= 1'b1;
          if (!ctrl_reg[0])
          begin
            task_done <= 1'b0;
            state <= S_IDLE;
          end
        end

        default:
          state <= S_IDLE;
      endcase
    end
  end

endmodule
