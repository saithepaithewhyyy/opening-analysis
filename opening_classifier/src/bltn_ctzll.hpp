#pragma once

#ifdef _MSC_VER
    #include <intrin.h>
    inline int ctzll(unsigned long long x) {
        if (x == 0) return -1;  // safety
        unsigned long index;
        _BitScanForward64(&index, x);
        return index;
    }
#else
    inline int ctzll(unsigned long long x) {
        if (x == 0) return -1;  // safety
        return __builtin_ctzll(x);
    }
#endif