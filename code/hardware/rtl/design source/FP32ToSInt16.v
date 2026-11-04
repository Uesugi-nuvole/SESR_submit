module FP32ToSInt16(
  input      [31:0] in,
  output reg [15:0] out
);
  wire        sign = in[31];
  wire [7:0]  exp = in[30:23];
  wire [22:0] frac = in[22:0];
  reg  [23:0] mant;
  integer     e;
  integer     shift;
  reg  [55:0] mant_ext;
  reg  [55:0] frac_mask;
  reg  [31:0] abs_rounded;
  reg signed [31:0] signed_value;

  always @(*) begin
    mant = exp == 8'h0 ? {1'b0, frac} : {1'b1, frac};
    e = {24'h0, exp};
    e = e - 127;
    mant_ext = {32'h0, mant};
    frac_mask = 56'h0;
    abs_rounded = 32'h0;

    if (exp == 8'h0) begin
      abs_rounded = 32'h0;
    end else if (exp == 8'hff) begin
      abs_rounded = 32'h7fffffff;
    end else if (e >= 31) begin
      abs_rounded = 32'h7fffffff;
    end else if (e >= 23) begin
      abs_rounded = {8'h0, mant} << (e - 23);
    end else if (e >= 0) begin
      shift = 23 - e;
      abs_rounded = {8'h0, mant} >> shift;
      frac_mask = (56'h1 << shift) - 56'h1;
      if ((mant_ext & frac_mask) >= (56'h1 << (shift - 1)))
        abs_rounded = abs_rounded + 32'h1;
    end else if (e == -1) begin
      abs_rounded = 32'h1;
    end

    if (sign) begin
      if (abs_rounded >= 32'd32768)
        signed_value = -32'sd32768;
      else
        signed_value = -$signed({16'h0, abs_rounded[15:0]});
    end else begin
      if (abs_rounded > 32'd32767)
        signed_value = 32'sd32767;
      else
        signed_value = $signed({16'h0, abs_rounded[15:0]});
    end

    out = signed_value[15:0];
  end
endmodule
