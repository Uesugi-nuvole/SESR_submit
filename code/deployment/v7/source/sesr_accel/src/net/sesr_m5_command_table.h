
#ifndef SESR_COMMAND_TABLE_H
#define SESR_COMMAND_TABLE_H

#include "../ncnn/net.h"

namespace demo {

const ncnn::FixedParamTable* sesr_fixed_param_table(int scale);
const char* sesr_fixed_param_table_name(int scale);

} // namespace demo

#endif // SESR_COMMAND_TABLE_H
