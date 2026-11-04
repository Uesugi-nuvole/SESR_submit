`timescale 1ns/100ps

`ifdef XILINX_SIMULATOR
module short(in1, in1);
  inout in1;
endmodule
`endif

module tb_top;

  initial begin
    $timeformat(-9, 3, " ns", 10);
  end

  // ============================================================
  // DDR4 model parameters
  // ============================================================

  localparam ADDR_WIDTH = 17;
  localparam DQ_WIDTH   = 64;
  localparam DQS_WIDTH  = 8;
  localparam DM_WIDTH   = 8;
  localparam DRAM_WIDTH = 8;

  localparam NUM_PHYSICAL_PARTS = DQ_WIDTH / DRAM_WIDTH;

  parameter RANK_WIDTH = 1;
  parameter CS_WIDTH   = 1;
  parameter ODT_WIDTH  = 1;
  parameter CA_MIRROR  = "OFF";

  localparam MRS = 3'b000;
  localparam REF = 3'b001;
  localparam PRE = 3'b010;
  localparam ACT = 3'b011;
  localparam WR  = 3'b100;
  localparam RD  = 3'b101;
  localparam ZQC = 3'b110;
  localparam NOP = 3'b111;

  import arch_package::*;
  parameter UTYPE_density CONFIGURED_DENSITY = _8G;


  // ============================================================
  // UART / MRD dump parameters
  // ============================================================

  parameter int UART_BAUD = 115200;
  localparam real UART_BIT_NS = 1_000_000_000.0 / UART_BAUD;

  localparam int MRD_DUMP_SIZE = 921600;

  string MRD_TCL_FILE = "mrd_dump.tcl";
  string MRD_BIN_FILE = "output.txt";

  integer uart_log_fd;

  string uart_line;
  bit    mrd_tcl_generated;


  // ============================================================
  // Clock and reset
  // ============================================================

  reg sys_clk_i = 1'b0;

  // 100 MHz input clock = 10 ns period
  always #(10.0 / 2.0) sys_clk_i = ~sys_clk_i;

  wire c0_sys_clk_clk_p = sys_clk_i;
  wire c0_sys_clk_clk_n = ~sys_clk_i;

  reg [0:0] sys_rstn = 1'b0;
  reg uart_rtl_0_rxd = 1'b1;

  wire uart_rtl_0_txd;
  wire c0_init_calib_complete_0;

  wire        C0_DDR4_0_act_n;
  wire [16:0] C0_DDR4_0_adr;
  wire [1:0]  C0_DDR4_0_ba;
  wire [1:0]  C0_DDR4_0_bg;
  wire [0:0]  C0_DDR4_0_ck_c;
  wire [0:0]  C0_DDR4_0_ck_t;
  wire [0:0]  C0_DDR4_0_cke;
  wire [0:0]  C0_DDR4_0_cs_n;
  wire [7:0]  C0_DDR4_0_dm_n;
  wire [63:0] C0_DDR4_0_dq;
  wire [7:0]  C0_DDR4_0_dqs_c;
  wire [7:0]  C0_DDR4_0_dqs_t;
  wire [0:0]  C0_DDR4_0_odt;
  wire        C0_DDR4_0_reset_n;

  bit en_model;
  tri model_enable = en_model;

  initial begin
    sys_rstn = 1'b0;
    en_model = 1'b0;
    uart_rtl_0_rxd = 1'b1;

    #0.005;
    en_model = 1'b1;

    // Hold reset for 200 ns
    #200;
    sys_rstn = 1'b1;
  end


  // ============================================================
  // DUT
  // ============================================================

  system_wrapper system_i (
    .C0_DDR4_0_act_n(C0_DDR4_0_act_n),
    .C0_DDR4_0_adr(C0_DDR4_0_adr),
    .C0_DDR4_0_ba(C0_DDR4_0_ba),
    .C0_DDR4_0_bg(C0_DDR4_0_bg),
    .C0_DDR4_0_ck_c(C0_DDR4_0_ck_c),
    .C0_DDR4_0_ck_t(C0_DDR4_0_ck_t),
    .C0_DDR4_0_cke(C0_DDR4_0_cke),
    .C0_DDR4_0_cs_n(C0_DDR4_0_cs_n),
    .C0_DDR4_0_dm_n(C0_DDR4_0_dm_n),
    .C0_DDR4_0_dq(C0_DDR4_0_dq),
    .C0_DDR4_0_dqs_c(C0_DDR4_0_dqs_c),
    .C0_DDR4_0_dqs_t(C0_DDR4_0_dqs_t),
    .C0_DDR4_0_odt(C0_DDR4_0_odt),
    .C0_DDR4_0_reset_n(C0_DDR4_0_reset_n),
    .c0_init_calib_complete_0(c0_init_calib_complete_0),
    .c0_sys_clk_clk_n(c0_sys_clk_clk_n),
    .c0_sys_clk_clk_p(c0_sys_clk_clk_p),
    .sys_rstn(sys_rstn),
    .uart_rtl_0_rxd(uart_rtl_0_rxd),
    .uart_rtl_0_txd(uart_rtl_0_txd)
  );


  // ============================================================
  // DDR4 model helper signals
  // ============================================================

  reg [16:0] c0_ddr4_adr_sdram[1:0];
  reg [1:0]  c0_ddr4_ba_sdram[1:0];
  reg [1:0]  c0_ddr4_bg_sdram[1:0];

  wire        c0_ddr4_act_n    = C0_DDR4_0_act_n;
  wire [16:0] c0_ddr4_adr      = C0_DDR4_0_adr;
  wire [1:0]  c0_ddr4_ba       = C0_DDR4_0_ba;
  wire [1:0]  c0_ddr4_bg       = C0_DDR4_0_bg;
  wire [0:0]  c0_ddr4_cke      = C0_DDR4_0_cke;
  wire [0:0]  c0_ddr4_odt      = C0_DDR4_0_odt;
  wire [0:0]  c0_ddr4_cs_n     = C0_DDR4_0_cs_n;
  wire        c0_ddr4_ck_t     = C0_DDR4_0_ck_t[0];
  wire        c0_ddr4_ck_c     = C0_DDR4_0_ck_c[0];
  wire        c0_ddr4_reset_n  = C0_DDR4_0_reset_n;
  wire [7:0]  c0_ddr4_dm_dbi_n = C0_DDR4_0_dm_n;
  wire [63:0] c0_ddr4_dq       = C0_DDR4_0_dq;
  wire [7:0]  c0_ddr4_dqs_c    = C0_DDR4_0_dqs_c;
  wire [7:0]  c0_ddr4_dqs_t    = C0_DDR4_0_dqs_t;

  always @(*) begin
    c0_ddr4_adr_sdram[0] = c0_ddr4_adr;
    c0_ddr4_adr_sdram[1] = (CA_MIRROR == "ON") ?
      {c0_ddr4_adr[ADDR_WIDTH-1:14],
       c0_ddr4_adr[11], c0_ddr4_adr[12],
       c0_ddr4_adr[13], c0_ddr4_adr[10:9],
       c0_ddr4_adr[7],  c0_ddr4_adr[8],
       c0_ddr4_adr[5],  c0_ddr4_adr[6],
       c0_ddr4_adr[3],  c0_ddr4_adr[4],
       c0_ddr4_adr[2:0]} :
       c0_ddr4_adr;

    c0_ddr4_ba_sdram[0] = c0_ddr4_ba;
    c0_ddr4_ba_sdram[1] = (CA_MIRROR == "ON") ?
      {c0_ddr4_ba[0], c0_ddr4_ba[1]} :
       c0_ddr4_ba;

    c0_ddr4_bg_sdram[0] = c0_ddr4_bg;
    c0_ddr4_bg_sdram[1] = (CA_MIRROR == "ON" && DRAM_WIDTH != 16) ?
      {c0_ddr4_bg[0], c0_ddr4_bg[1]} :
       c0_ddr4_bg;
  end

  reg [31:0] cmdName;
  reg [ADDR_WIDTH-1:0] DDR4_ADRMOD[RANK_WIDTH-1:0];

  always @(*) begin
    if (&c0_ddr4_cs_n) begin
      cmdName = "DSEL";
    end else if (c0_ddr4_act_n) begin
      casez (DDR4_ADRMOD[0][16:14])
        MRS:     cmdName = "MRS";
        REF:     cmdName = "REF";
        PRE:     cmdName = "PRE";
        WR:      cmdName = "WR";
        RD:      cmdName = "RD";
        ZQC:     cmdName = "ZQC";
        NOP:     cmdName = "NOP";
        default: cmdName = "***";
      endcase
    end else begin
      cmdName = "ACT";
    end
  end

  genvar rnk;
  generate
    for (rnk = 0; rnk < CS_WIDTH; rnk = rnk + 1) begin : rankup
      always @(*) begin
        if (c0_ddr4_act_n) begin
          casez (c0_ddr4_adr_sdram[0][16:14])
            WR, RD:  DDR4_ADRMOD[rnk] = c0_ddr4_adr_sdram[rnk] & 17'h047FF;
            default: DDR4_ADRMOD[rnk] = c0_ddr4_adr_sdram[rnk];
          endcase
        end else begin
          DDR4_ADRMOD[rnk] = c0_ddr4_adr_sdram[rnk];
        end
      end
    end
  endgenerate


  // ============================================================
  // DDR4 x8 memory model
  // 8 physical x8 devices form a 64-bit interface
  // ============================================================

  genvar i;
  genvar r;
  genvar s;

  generate
    DDR4_if #(.CONFIGURED_DQ_BITS(8)) iDDR4[0:(RANK_WIDTH*NUM_PHYSICAL_PARTS)-1]();

    for (r = 0; r < RANK_WIDTH; r = r + 1) begin : memModels_Ri1
      for (i = 0; i < NUM_PHYSICAL_PARTS; i = i + 1) begin : memModel1
        ddr4_model #(
          .CONFIGURED_DQ_BITS(8),
          .CONFIGURED_DENSITY(CONFIGURED_DENSITY)
        ) ddr4_model (
          .model_enable(model_enable),
          .iDDR4(iDDR4[(r*NUM_PHYSICAL_PARTS)+i])
        );
      end
    end

    for (r = 0; r < RANK_WIDTH; r = r + 1) begin : tranDQ2
      for (i = 0; i < NUM_PHYSICAL_PARTS; i = i + 1) begin : tranDQ12
        for (s = 0; s < 8; s = s + 1) begin : tranDQp
          `ifdef XILINX_SIMULATOR
            short bidiDQ(iDDR4[(r*NUM_PHYSICAL_PARTS)+i].DQ[s], c0_ddr4_dq[s+i*8]);
          `else
            tran bidiDQ(iDDR4[(r*NUM_PHYSICAL_PARTS)+i].DQ[s], c0_ddr4_dq[s+i*8]);
          `endif
        end
      end
    end

    for (r = 0; r < RANK_WIDTH; r = r + 1) begin : tranDQS2
      for (i = 0; i < NUM_PHYSICAL_PARTS; i = i + 1) begin : tranDQS12
        `ifdef XILINX_SIMULATOR
          short bidiDQS(iDDR4[(r*NUM_PHYSICAL_PARTS)+i].DQS_t, c0_ddr4_dqs_t[i]);
          short bidiDQS_(iDDR4[(r*NUM_PHYSICAL_PARTS)+i].DQS_c, c0_ddr4_dqs_c[i]);
          short bidiDM(iDDR4[(r*NUM_PHYSICAL_PARTS)+i].DM_n, c0_ddr4_dm_dbi_n[i]);
        `else
          tran bidiDQS(iDDR4[(r*NUM_PHYSICAL_PARTS)+i].DQS_t, c0_ddr4_dqs_t[i]);
          tran bidiDQS_(iDDR4[(r*NUM_PHYSICAL_PARTS)+i].DQS_c, c0_ddr4_dqs_c[i]);
          tran bidiDM(iDDR4[(r*NUM_PHYSICAL_PARTS)+i].DM_n, c0_ddr4_dm_dbi_n[i]);
        `endif
      end
    end

    for (r = 0; r < RANK_WIDTH; r = r + 1) begin : ADDR_RANKS
      for (i = 0; i < NUM_PHYSICAL_PARTS; i = i + 1) begin : ADDR_R
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].BG      = c0_ddr4_bg_sdram[r];
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].BA      = c0_ddr4_ba_sdram[r];
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].ADDR_17 = (ADDR_WIDTH == 18) ? DDR4_ADRMOD[r][ADDR_WIDTH-1] : 1'b0;
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].ADDR    = DDR4_ADRMOD[r][13:0];
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].CS_n    = c0_ddr4_cs_n[r];
      end
    end

    for (r = 0; r < RANK_WIDTH; r = r + 1) begin : tranADCTL_RANKS1
      for (i = 0; i < NUM_PHYSICAL_PARTS; i = i + 1) begin : tranADCTL1
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].CK        = {c0_ddr4_ck_t, c0_ddr4_ck_c};
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].ACT_n     = c0_ddr4_act_n;
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].RAS_n_A16 = DDR4_ADRMOD[r][16];
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].CAS_n_A15 = DDR4_ADRMOD[r][15];
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].WE_n_A14  = DDR4_ADRMOD[r][14];
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].CKE       = c0_ddr4_cke[r];
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].ODT       = c0_ddr4_odt[r];
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].PARITY    = 1'b0;
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].TEN       = 1'b0;
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].ZQ        = 1'b1;
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].PWR       = 1'b1;
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].VREF_CA   = 1'b1;
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].VREF_DQ   = 1'b1;
        assign iDDR4[(r*NUM_PHYSICAL_PARTS)+i].RESET_n   = c0_ddr4_reset_n;
      end
    end
  endgenerate


  // ============================================================
  // UART helper functions/tasks
  // ============================================================

  task automatic append_char_to_line;
    input [7:0] data;
    begin
      uart_line = {uart_line, " "};
      uart_line.putc(uart_line.len() - 1, data);
    end
  endtask


  task automatic handle_uart_char;
    input [7:0] data;
    begin
      if (data >= 8'h20 && data <= 8'h7e) begin
        $write("%c", data);

        if (uart_log_fd != 0)
          $fwrite(uart_log_fd, "%c", data);

        append_char_to_line(data);
      end else if (data == 8'h0a) begin
        $write("\n");

        if (uart_log_fd != 0)
          $fwrite(uart_log_fd, "\n");

        if (uart_line.len() != 0) begin
          process_uart_line(uart_line);
          uart_line = "";
        end
      end else if (data == 8'h0d) begin
        if (uart_line.len() != 0) begin
          process_uart_line(uart_line);
          uart_line = "";
        end
      end else begin
        $write("<0x%02h>", data);

        if (uart_log_fd != 0)
          $fwrite(uart_log_fd, "<0x%02h>", data);
      end

      if (uart_log_fd != 0)
        $fflush(uart_log_fd);
    end
  endtask


  task automatic uart_rx_monitor;
    integer k;
    reg [7:0] data;
    begin
      forever begin
        @(negedge uart_rtl_0_txd);

        #(UART_BIT_NS / 2.0);

        if (uart_rtl_0_txd !== 1'b0) begin
          // False start bit
        end else begin
          for (k = 0; k < 8; k = k + 1) begin
            #(UART_BIT_NS);
            data[k] = uart_rtl_0_txd;
          end

          #(UART_BIT_NS);

          if (uart_rtl_0_txd !== 1'b1) begin
            $display("[%0t] UART framing error, data = 0x%02h", $time, data);
          end else begin
            handle_uart_char(data);
          end
        end
      end
    end
  endtask


  task automatic process_uart_line;
    input string line;

    longint unsigned addr;
    bit found;

    begin
      $display("[%0t] UART LINE: %s", $time, line);

      if (!mrd_tcl_generated) begin
        found = extract_hex_addr(line, addr);

        if (found) begin
          $display("[%0t] Found address from UART: 0x%016h", $time, addr);
          generate_mrd_tcl(addr);
          mrd_tcl_generated = 1'b1;
        end
      end
    end
  endtask


  function automatic bit extract_hex_addr;
    input  string line;
    output longint unsigned addr;

    int idx;
    int j;
    int hex_digit;
    int hex_count;
    byte c;
    bit keep_parse;

    begin
      extract_hex_addr = 1'b0;
      addr = 64'd0;

      for (idx = 0; idx < line.len() - 1; idx = idx + 1) begin
        if (!extract_hex_addr) begin
          if ((line[idx] == "0") &&
             ((line[idx+1] == "x") || (line[idx+1] == "X"))) begin

            addr = 64'd0;
            hex_count = 0;
            j = idx + 2;
            keep_parse = 1'b1;

            while ((j < line.len()) && keep_parse) begin
              c = line[j];

              if ((c >= "0") && (c <= "9")) begin
                hex_digit = c - "0";
              end else if ((c >= "a") && (c <= "f")) begin
                hex_digit = c - "a" + 10;
              end else if ((c >= "A") && (c <= "F")) begin
                hex_digit = c - "A" + 10;
              end else begin
                keep_parse = 1'b0;
              end

              if (keep_parse) begin
                addr = (addr << 4) | hex_digit[3:0];
                hex_count = hex_count + 1;
                j = j + 1;
              end
            end

            if (hex_count > 0) begin
              extract_hex_addr = 1'b1;
            end
          end
        end
      end
    end
  endfunction


  task automatic generate_mrd_tcl;
    input longint unsigned addr;

    integer fd;

    begin
      fd = $fopen(MRD_TCL_FILE, "w");

      if (fd == 0) begin
        $display("[%0t] ERROR: Cannot create %s", $time, MRD_TCL_FILE);
      end else begin
        $fdisplay(fd, "# Auto generated by simulation testbench");
        $fdisplay(fd, "# UART detected DDR address: 0x%016h", addr);
        $fdisplay(fd, "");
        $fdisplay(fd, "mrd -bin -file %s 0x%016h %0d",
                  MRD_BIN_FILE,
                  addr,
                  MRD_DUMP_SIZE);
        $fclose(fd);

        $display("");
        $display("============================================================");
        $display("[%0t] mrd Tcl generated successfully", $time);
        $display("Tcl file : %s", MRD_TCL_FILE);
        $display("Bin file : %s", MRD_BIN_FILE);
        $display("Command  : mrd -bin -file %s 0x%016h %0d",
                 MRD_BIN_FILE,
                 addr,
                 MRD_DUMP_SIZE);
        $display("");
        $display("Next step:");
        $display("  Open XSDB / Vitis Tcl Console, then run:");
        $display("  source %s", MRD_TCL_FILE);
        $display("============================================================");
        $display("");
      end
    end
  endtask


  task automatic uart_send_byte;
    input [7:0] data;
    integer k;
    begin
      uart_rtl_0_rxd = 1'b1;
      #(UART_BIT_NS);

      uart_rtl_0_rxd = 1'b0;
      #(UART_BIT_NS);

      for (k = 0; k < 8; k = k + 1) begin
        uart_rtl_0_rxd = data[k];
        #(UART_BIT_NS);
      end

      uart_rtl_0_rxd = 1'b1;
      #(UART_BIT_NS);
    end
  endtask


  task automatic uart_send_string;
    input string str;
    integer k;
    begin
      for (k = 0; k < str.len(); k = k + 1) begin
        uart_send_byte(str[k]);
      end
    end
  endtask


  // ============================================================
  // Simulation control
  // ============================================================

  initial begin
    uart_log_fd = $fopen("uart_output.log", "w");

    if (uart_log_fd == 0) begin
      $display("[%0t] ERROR: Cannot open uart_output.log", $time);
    end else begin
      $display("[%0t] UART log file opened: uart_output.log", $time);
    end

    uart_line = "";
    mrd_tcl_generated = 1'b0;
  end


  initial begin
    $display("[%0t] UART monitor started, baud = %0d", $time, UART_BAUD);
    uart_rx_monitor();
  end


  initial begin
    uart_rtl_0_rxd = 1'b1;

    wait(sys_rstn === 1'b1);
    wait(c0_init_calib_complete_0 === 1'b1);

    #100000;

    $display("[%0t] Optional UART input send point", $time);

    // Uncomment if needed:
    // uart_send_string("start\r\n");
  end


  initial begin
    wait(c0_init_calib_complete_0 === 1'b1);
    $display("[%0t] DDR4 calibration complete", $time);
  end


  initial begin
    #1000000000;
    $display("[%0t] Simulation timeout", $time);
    $finish;
  end

endmodule