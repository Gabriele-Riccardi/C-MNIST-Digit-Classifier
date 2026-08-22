#ifndef PCG_PRNG_H
#define PCG_PRNG_H

#include <stdint.h>
#include <math.h>

#include <stdint.h>
#include <math.h>
#include <stdio.h> 

/* Type aliases for clarity and conciseness */
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;

#define PI 3.14159265358979323846f

/*
 * Based on PCG Random Number Generator (https://www.pcg-random.org)
 * Licensed under Apache License 2.0
 */

/**
 * @brief Represents the state of an independent PCG random number generator instance.
 *
 * Storing state in a struct enables reentrancy (_r functions), allowing multiple 
 * independent PRNG streams across different threads or modules.
 */
typedef struct {    
    u64 state;     /**< Current internal 64-bit LCG state. Updated on every draw. */
    u64 inc;       /**< Stream selector / sequence ID. Must be odd for full period. */
    f32 prev_norm; /**< Cache for the Box-Muller transform. Stores 2nd value (NAN if empty). */
} prng_state;

/* Function Prototypes */
void prng_seed_r(prng_state* rng, u64 initstate, u64 initseq);
void prng_seed(u64 initstate, u64 initseq);

u32 prng_below_r(prng_state* rng, u32 bound);
u32 prng_below(u32 bound);

u32 prng_rand_r(prng_state* rng);
u32 prng_rand(void);

f32 prng_randf_r(prng_state* rng);
f32 prng_randf(void);

f32 prng_rand_norm_r(prng_state* rng);
f32 prng_rand_norm(void);

void plat_get_entropy(void* data, u32 size);
#endif // PCG_PRNG_H
