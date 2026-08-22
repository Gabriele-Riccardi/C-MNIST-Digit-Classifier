#include "prng.h"
#include <stdio.h>

/**
 * @brief Default global PRNG state instance.
 * 
 * Used by non-reentrant wrapper functions (e.g., prng_rand(), prng_seed()).
 * Pre-seeded with official PCG default constants and an empty Box-Muller cache.
 */
static prng_state s_prng_state = {
    0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL, NAN
};

/**
 * @brief Reentrant function to seed a specific PRNG instance.
 *
 * Configures the generator stream and state using PCG's standard protocol.
 * Ensures the increment ('inc') is odd for maximum period length, warms up the state,
 * and clears the Box-Muller normal distribution cache.
 */
void prng_seed_r(prng_state* rng, u64 initstate, u64 initseq) {
    rng->state = 0U;
    /* Ensure increment is odd (initseq * 2 + 1) for a full 2^64 period */
    rng->inc = (initseq << 1u) | 1u;
    
    /* Warm up the state mixing algorithm */
    prng_rand_r(rng);
    rng->state += initstate;
    prng_rand_r(rng);

    /* Reset the Gaussian value cache */
    rng->prev_norm = NAN;
}

/**
 * @brief Global wrapper to seed the default PRNG instance (s_prng_state).
 */
void prng_seed(u64 initstate, u64 initseq) {
    prng_seed_r(&s_prng_state, initstate, initseq);
}

/**
 * @brief Reentrant function to generate a pseudo-random 32-bit unsigned integer.
 *
 * Uses PCG-XSH-RR variant: Linear Congruential Generator (LCG) step combined
 * with xorshift and bitwise rotation output permutation.
 */
u32 prng_rand_r(prng_state* rng) {
    u64 oldstate = rng->state;
    /* Advance the internal LCG state */
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    
    /* Permutation step: xorshift followed by variable bit rotation */
    u32 xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    u32 rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

/**
 * @brief Global wrapper to generate a pseudo-random 32-bit integer using s_prng_state.
 */
u32 prng_rand(void) {
    return prng_rand_r(&s_prng_state);
}

/**
 * @brief Uniform integer in [0, bound) from a specific instance, with rejection
 *        sampling to remove modulo bias.
 */
u32 prng_below_r(prng_state* rng, u32 bound) {
    u32 threshold = (u32)(0x100000000ULL % (u64)bound);   /* 2^32 % bound */
    for (;;) {
        u32 r = prng_rand_r(rng);
        if (r >= threshold) return r % bound;
    }
}

/**
 * @brief Uniform integer in [0, bound) with rejection sampling to remove modulo bias.
 */
u32 prng_below(u32 bound) {
    return prng_below_r(&s_prng_state, bound);
}

/**
 * @brief Reentrant function to generate a uniform float in the range [0.0, 1.0].
 */
f32 prng_randf_r(prng_state* rng) {
    return (f32)prng_rand_r(rng) / (f32)UINT32_MAX;
}

/**
 * @brief Global wrapper to generate a uniform float in [0.0, 1.0] using s_prng_state.
 */
f32 prng_randf(void) {
    return prng_randf_r(&s_prng_state);
}

/**
 * @brief Reentrant function to generate normally distributed floats (mean=0, stddev=1).
 *
 * Implements the Box-Muller transform. Caches the second generated value in
 * rng->prev_norm to save CPU cycles on subsequent calls.
 */
f32 prng_rand_norm_r(prng_state* rng) {
    /* Check if a cached value is available from the previous run */
    if (!isnan(rng->prev_norm)) {
        f32 out = rng->prev_norm; 
        rng->prev_norm = NAN;
        return out;
    }
   
    /* Generate u1 in range (0, 1] to prevent log(0) error */
    f32 u1 = 0.0f;
    do {
        u1 = prng_randf_r(rng);
    } while (u1 == 0.0f);

    f32 u2 = prng_randf_r(rng);

    /* Box-Muller transformation equations */
    f32 mag = sqrtf(-2.0f * logf(u1));

    f32 z0 = mag * cosf(2.0f * PI * u2);
    f32 z1 = mag * sinf(2.0f * PI * u2);

    /* Cache the second normal value and return the first */
    rng->prev_norm = z1;
    return z0; 
}

/**
 * @brief Global wrapper to generate a normally distributed float using s_prng_state.
 */
f32 prng_rand_norm(void) {
    return prng_rand_norm_r(&s_prng_state);
}

/* =========================================================================
 * Platform-Specific Hardware Entropy Extraction
 * ========================================================================= 
*/

#if defined(_WIN32)
    #include <windows.h>
    #include <wincrypt.h>
 
    /**
     * @brief Fills the buffer with random entropy on Windows via CryptoAPI.
     */
    void plat_get_entropy(void* data, u32 size) {
        HCRYPTPROV prov;
        if (CryptAcquireContext(&prov, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT)) {
            CryptGenRandom(prov, (DWORD)size, (BYTE*)data);
            CryptReleaseContext(prov, 0);
        }
    }
#elif defined(__linux__)
    #include <sys/random.h>

    /**
     * @brief Fills the buffer with random entropy on Linux via getrandom().
     */
    void plat_get_entropy(void* data, u32 size) {
        ssize_t n = getrandom(data, size, 0);
        (void)n;
    }
#elif defined(__APPLE__)
    #include <sys/random.h>

    /**
     * @brief Fills the buffer with random entropy on macOS via getentropy().
     */
    void plat_get_entropy(void* data, u32 size) {
        int r = getentropy(data, size);
        (void)r;
    }
#else
    #error "plat_get_entropy is not implemented for this platform"
#endif