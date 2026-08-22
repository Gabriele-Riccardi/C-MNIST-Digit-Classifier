# prng — PCG32 with normal distribution

A small single-file pseudo-random number generator in C, based on **PCG32**
with a **Box-Muller** transform for Gaussian samples.

The PCG core is based on the work of Melissa O'Neill — <https://www.pcg-random.org> —
distributed under the Apache 2.0 license (NO WARRANTY, see the website).

---

## Features

- **PCG32 (XSH-RR)**: good statistical quality, fast, period 2⁶⁴.
- **Three distributions**:
  - uniform 32-bit integers,
  - uniform reals in `[0, 1]`,
  - Gaussian `N(0, 1)` (mean 0, standard deviation 1).
- **Two variants of every function**:
  - `_r` suffix → *reentrant*, you own the state (thread-safe with one state per thread),
  - no suffix → uses an internal global instance, more convenient.
- **System-entropy seeding** on Windows, Linux, and macOS.

---

## Requirements

- A C99 compiler.
- **libm** (`-lm` on Unix) — needed for `sqrtf`, `logf`, `cosf`, `sinf` used by the normal generator.
- System entropy source:
  - Linux: `getrandom()` (glibc ≥ 2.25),
  - macOS: `getentropy()` (≥ 10.12),
  - Windows: `CryptGenRandom()`.

---

## Building

```sh
# Linux / macOS
gcc -O2 -Wall -Wextra rng.c -o rng -lm

# Windows (MinGW)
gcc -O2 rng.c -o rng.exe
```

> ⚠️ **Do not build with `-ffast-math` or `-Ofast`** — see the *Caveats* section.

---

## Usage

```c
prng_state rng = {0};

// 1. Seed from system entropy
u64 seeds[2] = {0};
plat_get_entropy(seeds, sizeof(seeds));
prng_seed_r(&rng, seeds[0], seeds[1]);

// 2. Draw numbers
u32 a = prng_rand_r(&rng);        // uniform integer  [0, 2^32)
f32 b = prng_randf_r(&rng);       // uniform real     [0, 1]
f32 z = prng_rand_norm_r(&rng);   // Gaussian         N(0, 1)
```

The global variants save you from passing the state around, but must be seeded
once with `prng_seed(...)`:

```c
prng_seed(seeds[0], seeds[1]);
f32 z = prng_rand_norm();
```

---

## API

| Function                                        | Returns | Description                          |
| ----------------------------------------------- | ------- | ------------------------------------ |
| `prng_seed_r(prng_state*, u64 state, u64 seq)`  | `void`  | Seed a generator.                    |
| `prng_seed(u64 state, u64 seq)`                 | `void`  | Seed the global state.               |
| `prng_rand_r(prng_state*)`                      | `u32`   | Uniform integer in `[0, 2^32)`.      |
| `prng_rand(void)`                               | `u32`   | Same, global state.                  |
| `prng_randf_r(prng_state*)`                     | `f32`   | Uniform real in `[0, 1]`.            |
| `prng_randf(void)`                              | `f32`   | Same, global state.                  |
| `prng_rand_norm_r(prng_state*)`                 | `f32`   | Gaussian sample `N(0, 1)`.           |
| `prng_rand_norm(void)`                          | `f32`   | Same, global state.                  |
| `plat_get_entropy(void* data, u32 size)`        | `void`  | Fill `data` with OS entropy.         |

The `seq` parameter of `prng_seed_r` selects the *stream*: two generators with the
same `state` but different `seq` produce independent sequences.

---

## Caveats

- **No `-ffast-math` / `-Ofast`.** `prng_rand_norm_r` uses `NAN` as a sentinel for
  its internal cache (Box-Muller produces two values at a time). Under fast-math the
  compiler assumes NaNs never occur, `isnan()` becomes unreliable, and the function
  may return incorrect values.

- **Seeding fails silently.** `plat_get_entropy` does not report errors: if the OS
  call fails, the seeds stay `0` and you get a *deterministic* sequence without
  noticing. For critical code, change the function to return a status and check it.

- **Not cryptographically secure.** PCG is not a CSPRNG: don't use it for keys,
  tokens, nonces, or anything security-related.

- **Float precision.** `prng_randf_r` has about 24 effective bits (the `float`
  mantissa), not 32. Irrelevant for the Gaussian, but worth knowing.

- **Thread safety.** Use the `_r` variants with one `prng_state` per thread. The
  global variants share a single state and are not thread-safe.

---

## License

The PCG core is by Melissa O'Neill, **Apache 2.0** license. See
<https://www.pcg-random.org>.
