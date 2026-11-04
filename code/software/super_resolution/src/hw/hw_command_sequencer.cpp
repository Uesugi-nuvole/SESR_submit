#include "hw_command_sequencer.h"

#include <stdio.h>

namespace accel {

static SesrSeqProgram* g_record_program = 0;
static int g_record_error = 0;
static SesrSeqMmio g_default_mmio = {};
static bool g_default_mmio_valid = false;
static uint32_t g_default_timeout_polls = 100000000u;

uint32_t sesr_seq_pack_header(uint32_t opcode, uint32_t target_mask, uint32_t reg)
{
    return ((opcode & 0xfu) << 28) | ((target_mask & 0xfu) << 16) | (reg & 0x7fu);
}

SesrSeqProgram::SesrSeqProgram()
    : overflow(false), error(0)
{
}

void SesrSeqProgram::clear()
{
    cmds.clear();
    overflow = false;
    error = 0;
}

void SesrSeqProgram::append(const SesrSeqCommand& cmd)
{
    if (cmds.size() >= SESR_SEQ_MAX_COMMANDS)
    {
        overflow = true;
        error = -1;
        return;
    }
    cmds.push_back(cmd);
}

void SesrSeqProgram::write_reg(uint32_t reg, uint32_t value)
{
    SesrSeqCommand cmd;
    cmd.header = sesr_seq_pack_header(SESR_SEQ_OP_WRITE_REG, 0, reg);
    cmd.data = value;
    append(cmd);
}

void SesrSeqProgram::wait_done(uint32_t done_mask)
{
    SesrSeqCommand cmd;
    cmd.header = sesr_seq_pack_header(SESR_SEQ_OP_WAIT_DONE, done_mask, 0);
    cmd.data = 0;
    append(cmd);
}

void SesrSeqProgram::delay(uint32_t cycles)
{
    SesrSeqCommand cmd;
    cmd.header = sesr_seq_pack_header(SESR_SEQ_OP_DELAY, 0, 0);
    cmd.data = cycles;
    append(cmd);
}

void SesrSeqProgram::end()
{
    SesrSeqCommand cmd;
    cmd.header = sesr_seq_pack_header(SESR_SEQ_OP_END, 0, 0);
    cmd.data = 0;
    append(cmd);
}

static int check_mmio(const SesrSeqMmio& mmio)
{
    if (!mmio.write32 || !mmio.read32)
    {
        printf("[ERR][sesr_seq] missing mmio read/write callbacks\n");
        return -1;
    }
    return 0;
}

int sesr_seq_upload(const SesrSeqMmio& mmio, const SesrSeqProgram& program)
{
    if (check_mmio(mmio) != 0)
        return -1;

    const std::vector<SesrSeqCommand>& cmds = program.commands();
    if (program.overflowed() || program.error_code() != 0)
    {
        printf("[ERR][sesr_seq] program record failed, error=%d commands=%zu max=%zu\n",
               program.error_code(), cmds.size(), SESR_SEQ_MAX_COMMANDS);
        return -1;
    }

    if (cmds.size() > SESR_SEQ_MAX_COMMANDS)
    {
        printf("[ERR][sesr_seq] program too large: %zu commands, max=%zu\n",
               cmds.size(), SESR_SEQ_MAX_COMMANDS);
        return -1;
    }

    mmio.write32(mmio.opaque, SESR_SEQ_REG_CTRL, 2);
    mmio.write32(mmio.opaque, SESR_SEQ_REG_CTRL, 0);

    uint32_t load_toggle = 0;
    for (size_t i = 0; i < cmds.size(); i++)
    {
        load_toggle ^= 0x100u;
        mmio.write32(mmio.opaque, SESR_SEQ_REG_WR_INDEX, (uint32_t)i);
        mmio.write32(mmio.opaque, SESR_SEQ_REG_WR_HEADER, cmds[i].header);
        mmio.write32(mmio.opaque, SESR_SEQ_REG_WR_DATA, cmds[i].data);
        mmio.write32(mmio.opaque, SESR_SEQ_REG_CTRL, load_toggle);
    }

    mmio.write32(mmio.opaque, SESR_SEQ_REG_COUNT, (uint32_t)cmds.size());
    mmio.write32(mmio.opaque, SESR_SEQ_REG_CTRL, 0);
    return 0;
}

int sesr_seq_start_and_wait(const SesrSeqMmio& mmio, uint32_t timeout_polls)
{
    if (check_mmio(mmio) != 0)
        return -1;

    mmio.write32(mmio.opaque, SESR_SEQ_REG_CTRL, 1);
    mmio.write32(mmio.opaque, SESR_SEQ_REG_CTRL, 0);

    for (uint32_t i = 0; i < timeout_polls; i++)
    {
        uint32_t status = mmio.read32(mmio.opaque, SESR_SEQ_REG_STATUS);
        if (status & 0x2u)
        {
            if (status & 0x4u)
            {
                uint32_t error = mmio.read32(mmio.opaque, SESR_SEQ_REG_ERROR_READBACK);
                printf("[ERR][sesr_seq] hardware sequencer failed status=0x%08x error=0x%08x\n", status, error);
                return -1;
            }
            return 0;
        }
    }

    printf("[ERR][sesr_seq] timeout waiting for hardware sequencer status=0x%08x pc=%u executed=%u\n",
           mmio.read32(mmio.opaque, SESR_SEQ_REG_STATUS),
           mmio.read32(mmio.opaque, SESR_SEQ_REG_PC_READBACK),
           mmio.read32(mmio.opaque, SESR_SEQ_REG_EXECUTED_READBACK));
    return -1;
}

int sesr_seq_set_default_mmio(const SesrSeqMmio& mmio, uint32_t timeout_polls)
{
    if (check_mmio(mmio) != 0)
        return -1;

    g_default_mmio = mmio;
    g_default_mmio_valid = true;
    g_default_timeout_polls = timeout_polls ? timeout_polls : 100000000u;
    return 0;
}

bool sesr_seq_default_mmio_available()
{
    return g_default_mmio_valid;
}

const SesrSeqMmio& sesr_seq_default_mmio()
{
    return g_default_mmio;
}

uint32_t sesr_seq_default_timeout_polls()
{
    return g_default_timeout_polls;
}

int sesr_seq_record_begin(SesrSeqProgram* program)
{
    if (!program)
    {
        printf("[ERR][sesr_seq] record_begin with null program\n");
        return -1;
    }
    if (g_record_program)
    {
        printf("[ERR][sesr_seq] nested record_begin is not supported\n");
        g_record_error = -1;
        return -1;
    }

    program->clear();
    g_record_program = program;
    g_record_error = 0;
    return 0;
}

int sesr_seq_record_end()
{
    if (!g_record_program)
        return g_record_error;

    if (g_record_program->overflowed())
        g_record_error = -1;
    g_record_program = 0;
    return g_record_error;
}

bool sesr_seq_recording()
{
    return g_record_program != 0;
}

int sesr_seq_record_error()
{
    if (g_record_program && g_record_program->overflowed())
        g_record_error = -1;
    return g_record_error;
}

static int check_record_target()
{
    if (!g_record_program)
        return 0;
    if (g_record_program->overflowed())
    {
        g_record_error = -1;
        return -1;
    }
    return 0;
}

int sesr_seq_record_write_reg(uint32_t reg, uint32_t value)
{
    if (!g_record_program)
        return 0;
    if (reg >= 128)
    {
        printf("[ERR][sesr_seq] invalid recorded register index %u\n", reg);
        g_record_error = -1;
        return -1;
    }
    if (check_record_target() != 0)
        return -1;

    g_record_program->write_reg(reg, value);
    if (g_record_program->overflowed())
    {
        printf("[ERR][sesr_seq] command overflow at reg write, max=%zu\n", SESR_SEQ_MAX_COMMANDS);
        g_record_error = -1;
        return -1;
    }
    return 0;
}

int sesr_seq_record_write_reg32(uint32_t reg, uint32_t value)
{
    return sesr_seq_record_write_reg(reg, (uint32_t)value);
}

int sesr_seq_record_wait(uint32_t done_mask)
{
    if (!g_record_program)
        return 0;
    if ((done_mask & ~0xfu) != 0)
    {
        printf("[ERR][sesr_seq] invalid done mask 0x%08x\n", done_mask);
        g_record_error = -1;
        return -1;
    }
    if (check_record_target() != 0)
        return -1;

    g_record_program->wait_done(done_mask);
    if (g_record_program->overflowed())
    {
        printf("[ERR][sesr_seq] command overflow at wait, max=%zu\n", SESR_SEQ_MAX_COMMANDS);
        g_record_error = -1;
        return -1;
    }
    return 0;
}

int sesr_seq_record_delay(uint32_t cycles)
{
    if (!g_record_program || cycles == 0)
        return 0;
    if (check_record_target() != 0)
        return -1;

    g_record_program->delay(cycles);
    if (g_record_program->overflowed())
    {
        printf("[ERR][sesr_seq] command overflow at delay, max=%zu\n", SESR_SEQ_MAX_COMMANDS);
        g_record_error = -1;
        return -1;
    }
    return 0;
}

} // namespace accel
