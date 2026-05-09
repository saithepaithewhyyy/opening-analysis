#pragma once

#ifdef _MSC_VER
    #include <intrin.h>
    #define U64(u) (u##ui64)
    inline int ctzll(unsigned long long x) {
        if (x == 0) return -1;
        unsigned long index;
        _BitScanForward64(&index, x);
        return index;
    }
    inline int clzll(unsigned long long x) {
        if (x == 0) return -1;
        unsigned long index;
        _BitScanReverse64(&index, x);
        return index;
    }

#else
    #define U64(u) (u##ULL)
    inline int ctzll(unsigned long long x) {
        if (x == 0) return -1;
        return __builtin_ctzll(x);
    }
    inline int clzll(unsigned long long x){
        if (x==0) return -1;
        return __builtin_clzll(x);
    }
#endif