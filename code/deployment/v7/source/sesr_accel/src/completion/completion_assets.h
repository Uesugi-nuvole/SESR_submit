#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>
extern const uint32_t completion_asset_words[];
extern const unsigned completion_asset_bytes;
// Destination is native float32 for float assets; packed weights are bytes.
static bool completion_copy_asset(size_t offset, void* dst, size_t bytes, bool packed_bytes) {
    if(offset > completion_asset_bytes || bytes > completion_asset_bytes-offset) return false;
    if(packed_bytes) {
        unsigned char* p = static_cast<unsigned char*>(dst);
        for(size_t j=0;j<bytes;++j) {
            size_t k=offset+j;
            p[j]=static_cast<unsigned char>(completion_asset_words[k/4] >> (8*(k%4)));
        }
    } else {
        if(offset%4 || bytes%4) return false;
        memcpy(dst, completion_asset_words+offset/4, bytes);
    }
    return true;
}
