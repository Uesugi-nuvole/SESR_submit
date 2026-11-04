#ifndef SESR_HW_COMMAND_SEQUENCER_H
#define SESR_HW_COMMAND_SEQUENCER_H

#include <stdint.h>
#include <stddef.h>
#include <vector>

namespace accel {

static const size_t SESR_SEQ_MAX_COMMANDS = 4096;

enum SesrSequencerReg
{
    SESR_SEQ_REG_CTRL = 117,
    SESR_SEQ_REG_STATUS = 118,
    SESR_SEQ_REG_WR_INDEX = 119,
    SESR_SEQ_REG_WR_HEADER = 120,
    SESR_SEQ_REG_WR_DATA = 121,
    SESR_SEQ_REG_COUNT = 122,
    SESR_SEQ_REG_PC_READBACK = 119,
    SESR_SEQ_REG_EXECUTED_READBACK = 120,
    SESR_SEQ_REG_ERROR_READBACK = 121,
    SESR_SEQ_REG_ELAPSED_LO = 122,
    SESR_SEQ_REG_ELAPSED_HI = 123,
    SESR_SEQ_REG_COUNT_READBACK = 124,
    SESR_SEQ_REG_DONE_MASK = 125,
    SESR_SEQ_REG_ACTIVE = 126,
    SESR_SEQ_REG_VERSION = 127
};

enum SesrSequencerOpcode
{
    SESR_SEQ_OP_END = 0,
    SESR_SEQ_OP_WRITE_REG = 1,
    SESR_SEQ_OP_WAIT_DONE = 2,
    SESR_SEQ_OP_DELAY = 3
};

enum SesrSequencerDoneMask
{
    SESR_SEQ_WAIT_ALU = 1u << 0,
    SESR_SEQ_WAIT_GEMM = 1u << 1,
    SESR_SEQ_WAIT_PIXELSHUFFLE = 1u << 2,
    SESR_SEQ_WAIT_IMAGE = 1u << 3
};

struct SesrSeqCommand
{
    uint32_t header;
    uint32_t data;
};

class SesrSeqProgram
{
public:
    SesrSeqProgram();

    void clear();
    void write_reg(uint32_t reg, uint32_t value);
    void wait_done(uint32_t done_mask);
    void delay(uint32_t cycles);
    void end();

    const std::vector<SesrSeqCommand>& commands() const { return cmds; }
    bool overflowed() const { return overflow; }
    int error_code() const { return error; }
    size_t command_count() const { return cmds.size(); }

private:
    void append(const SesrSeqCommand& cmd);

    std::vector<SesrSeqCommand> cmds;
    bool overflow;
    int error;
};

struct SesrSeqMmio
{
    void* opaque;
    void (*write32)(void* opaque, uint32_t reg, uint32_t value);
    uint32_t (*read32)(void* opaque, uint32_t reg);
};

uint32_t sesr_seq_pack_header(uint32_t opcode, uint32_t target_mask, uint32_t reg);
int sesr_seq_upload(const SesrSeqMmio& mmio, const SesrSeqProgram& program);
int sesr_seq_start_and_wait(const SesrSeqMmio& mmio, uint32_t timeout_polls);

int sesr_seq_set_default_mmio(const SesrSeqMmio& mmio, uint32_t timeout_polls);
bool sesr_seq_default_mmio_available();
const SesrSeqMmio& sesr_seq_default_mmio();
uint32_t sesr_seq_default_timeout_polls();

int sesr_seq_record_begin(SesrSeqProgram* program);
int sesr_seq_record_end();
bool sesr_seq_recording();
int sesr_seq_record_error();
int sesr_seq_record_write_reg(uint32_t reg, uint32_t value);
int sesr_seq_record_write_reg32(uint32_t reg, uint32_t value);
int sesr_seq_record_wait(uint32_t done_mask);
int sesr_seq_record_delay(uint32_t cycles);

} // namespace accel

#endif // SESR_HW_COMMAND_SEQUENCER_H
