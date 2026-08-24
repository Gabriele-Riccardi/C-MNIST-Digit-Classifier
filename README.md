# C-MNIST Digit Classifier

A handwritten-digit classifier written from scratch in **pure C** — no PyTorch, no TensorFlow, no BLAS, no external dependencies of any kind. Just `libc`, `libm`, and about 2,070 lines of source across a loader, a network, a training loop, a post-training quantiser and a command line — plus another 1,940 lines of tests and benchmarks that check them.

Every part of the pipeline is implemented by hand: the IDX binary parser, the matrix arithmetic, the forward pass, the softmax, the cross-entropy loss, the backpropagation, and the Cholesky factorisation the quantiser runs on. Nothing is delegated to a library.

```
$ scripts/get_dataset.sh          # download and verify the four IDX files
$ make check                      # unit tests, gradient check, quantiser check
$ make && ./mnist train           # fit on 54k, score the 6k held out
$ ./mnist test                    # score the 10k test set, once
$ scripts/quant_seeds.sh 5        # quantise to 8..2 bits and measure what it costs
```

## Every number in this README is reproducible

Backpropagation written by hand fails silently. A wrong sign or a transposed index still trains — the network just learns a bit worse — so an accuracy figure is not evidence that the gradient is correct, only that it is not catastrophically wrong. The same goes for performance claims: "faster" means nothing without a before, an after, and a machine.

So every quantitative claim below has a command next to it.

| Claim | Reproduce it with |
| --- | --- |
| The analytic gradient matches a central difference to 9.2e-08 | `make check` |
| Test accuracy is 98.29% ± 0.07 over five seeds | `scripts/run_seeds.sh 5 -- --val 0` |
| The hyperparameters were chosen on validation, never on test | `scripts/sweep.sh`, `scripts/ablation.sh` |
| The quantiser's compensation is the exact least-squares one, to 2.2e-16 | `make check` (check C4) |
| 2-bit GPTQ holds 97.24% ± 0.20 where round-to-nearest falls to 89.99% | `scripts/quant_seeds.sh 5` |
| At 2 bits, 1.3% of the weights do four times the damage of the other 98.7% | `EXPERIMENTS=layers scripts/quant_seeds.sh 5` |
| Block I/O made the loader ~88x faster | `make bench` |
| PCG instead of `rand()` is worth 1 ms per epoch, not 5x | `make bench` |
| The dataset is the canonical MNIST | `scripts/get_dataset.sh` (SHA-256 verified) |

Where a claim did not survive being measured, the measurement is what is written down. Three of them did not.

## Is the gradient actually right?

`tests/gradcheck.c` compares the gradient backpropagation produces against a central difference of the loss it claims to differentiate:

```
dL/dw  ~=  ( L(w + eps) - L(w - eps) ) / 2 eps
```

for a few hundred randomly chosen parameters across `w1`, `b1`, `w2` and `b2`, on five fixtures: fresh weights and trained weights, at MNIST geometry and at degenerate ones.

```
$ make check
gradcheck  float64  eps=1e-05  threshold=1e-06
  MNIST geometry, trained  (784-128-10, p(label)=0.9988, 50 warm-up steps, 29/128 hidden units active)
    w1  max rel err 9.23e-08  (analytic -1.9290e-07 vs numeric -1.9290e-07)   200 checked,   0 at a kink   ok
    b1  max rel err 1.12e-09  (analytic -9.0423e-06 vs numeric -9.0423e-06)    64 checked,   0 at a kink   ok
    w2  max rel err 1.57e-09  (analytic  2.3960e-06 vs numeric  2.3960e-06)   200 checked,   0 at a kink   ok
    b2  max rel err 7.04e-10  (analytic  6.0818e-06 vs numeric  6.0818e-06)    10 checked,   0 at a kink   ok
```

Worst case across every fixture: **9.2e-08**, against a threshold of 1e-06 and the conventional bar of 1e-05.

Three things had to be true for that number to mean anything.

**The check has to test the code that trains.** Gradients are computed into an explicit `gradients` struct by `net_backward`, and applied separately by `net_sgd_step`. The training loop calls both. Folding the update into the backward pass — which the earlier version of this project did — is faster, but then the gradient never exists as a value and the only way to check it is to re-derive it in the test, which checks the test.

**Float32 cannot demonstrate 1e-5.** A central difference pays for the step twice: truncation growing with `h`, and cancellation growing with `eps_machine / h` as the two loss evaluations lose their leading digits to each other. In float32 there is no `h` that makes both small enough. The library is therefore compiled twice from the same sources — `float` to train, `double` to check — behind a typedef in `src/real.h`, and `make check` runs both. The float32 build lands at **1.87e-02**; the identical code in float64 lands at **9.23e-08**. Five orders of magnitude, same gradient. That difference is the arithmetic, and a check that ignores it is measuring its own noise.

**The loss had to be computed differently.** `-log(softmax(z)[label])` loses its low-order bits once the network is confident: `p` approaches 1 and the loss becomes a small number obtained by cancelling two numbers near 1. That error is invisible during training — the loss is only printed — but it is the noise floor of any finite-difference check, and it held the trained-network fixtures at 2.8e-05, ten times outside the conventional bar. Computing the loss as `logsumexp(z) - z[label]` with the maximum cancelled analytically fixed it, and removed the need to clamp `p` away from zero along with the kink that clamp put in the loss surface.

Beyond the gradient, `tests/test_units.c` covers the IDX parser (wrong magic, truncated data, mismatched pairs, out-of-range labels), the split, serialisation, augmentation, softmax overflow, and the PRNG's bounds, and `tests/quantcheck.c` does the same job for the quantiser — see [Post-training quantisation](#post-training-quantisation). `make sanitize` reruns everything under AddressSanitizer and UndefinedBehaviorSanitizer.

## Protocol

The test set is opened by exactly one command, `./mnist test`, and nothing else in the repository reads it — the quantiser included: `build/quant_sweep` loads the training pair and has no code path that names the `t10k` files, and every quantised test number in this README comes from `./mnist test` afterwards. `./mnist train` splits 60,000 training images into **54,000 train / 6,000 validation** and reports validation accuracy after every epoch.

That separation is the point. Learning rate, hidden width and augmentation rate were chosen by `scripts/sweep.sh`, which reads validation accuracy and cannot see the test files; `scripts/run_seeds.sh` then scores the test set once per seed with the weights that run produced. A hyperparameter tuned against test accuracy turns the test set into a second training set, and the number it reports stops estimating anything.

The split is drawn from its own PRNG stream seeded independently of `--seed`, so changing the training seed reshuffles training without moving the held-out set — otherwise two runs' validation numbers would not be comparable. `tests/test_units.c` asserts this.

## Results

Five seeds, identical protocol, the test set scored once per seed.

| Training data | Validation | Test | Test range |
| --- | --- | --- | --- |
| 54,000 (6,000 held out) | 98.25% ± 0.09 | 98.18% ± 0.04 | 98.13 – 98.23 |
| 60,000 (retrained once the hyperparameters were fixed) | — | **98.29% ± 0.07** | 98.20 – 98.36 |

Mean ± sample standard deviation over seeds 1–5. `scripts/run_seeds.sh 5` and `scripts/run_seeds.sh 5 -- --val 0`.

For reference, the previous defaults — the same network with ±1 px augmentation on half the samples — scored 98.13% ± 0.17 on the 54k split and 98.18% ± 0.14 on 60k, over the same five seeds.

An earlier version of this README reported **98.38%**, from one run at one fixed seed. That run was real and it still reproduces — but it is above the highest of the twenty runs recorded here, the augmented ones included. A fixed seed makes a number reproducible; it says nothing about how stable that number is, and quoting it to two decimals implied a precision the measurement never had. 98.29% ± 0.07 is a claim. 98.38% was a draw.

### Against the literature

From the results table published with the MNIST database:

| Model | Test error | Source |
| --- | --- | --- |
| Linear classifier (1-layer NN) | 12.0% | LeCun et al. 1998 |
| K-nearest neighbours, Euclidean | 5.0% | LeCun et al. 1998 |
| 2-layer NN, 300 hidden units, MSE | 4.7% | LeCun et al. 1998 |
| 2-layer NN, 1000 hidden units, MSE | 4.5% | LeCun et al. 1998 |
| 3-layer NN, 300+100 hidden units | 3.05% | LeCun et al. 1998 |
| **this project, 128 HU, cross-entropy** | **1.71% ± 0.07** | |
| 2-layer NN, 800 HU, cross-entropy | 1.6% | Simard et al., ICDAR 2003 |
| 2-layer NN, 800 HU, cross-entropy, affine distortions | 1.1% | Simard et al., ICDAR 2003 |
| LeNet-5 (convolutional) | 0.8% | LeCun et al. 1998 |

That is the band a two-layer MLP trained with cross-entropy belongs in. The distance from LeCun's 1998 multilayer perceptrons is mostly the loss function — those minimised mean squared error; the remaining tenth of a point to Simard's 1.6% is width, 128 units against 800. Landing at 4.7% would have meant something was quietly wrong. Landing at 0.8% would have meant the test set had leaked.

### How the hyperparameters were chosen

`scripts/sweep.sh`, 10 epochs, seed 1, validation split only — the test files are not opened by anything in it.

| Learning rate | Validation | | Hidden units | Validation | | ±1 px augmentation | Validation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.003 | 97.82% | | 64 | 97.37% | | off | 98.03% |
| **0.01** | **98.03%** | | **128** | **98.03%** | | half the samples | 98.03% |
| 0.03 | 96.95% | | 256 | 98.38% | | every sample | 97.87% |
| 0.1 | 78.38% | | | | | | |

A sweep this size rules things out; it does not crown a winner. A learning rate of 0.1 diverges and 64 hidden units is clearly worse, but 0.003 against 0.01 is two tenths of a point from a single seed — well inside the ±0.09 seed-to-seed spread, so the sweep cannot separate them and does not pretend to.

Two of these differences looked big enough to be worth settling properly. `scripts/ablation.sh` reruns both arms over the same five seeds and reports the paired difference, which cancels most of the seed noise. Both were run on validation only.

**Augmentation, which used to be on by default, does not help.**

| ±1 px translation | Validation over 5 seeds |
| --- | --- |
| off | **98.25% ± 0.09** |
| on half the samples | 98.09% ± 0.13 |

Paired difference +0.16 points in favour of turning it off; sd 0.16, standard error 0.07, t = 2.3 on 4 degrees of freedom. Not decisive on its own, but pointing the wrong way for something that was in the defaults, and free to fix — so `--aug` now defaults to 0.

The likely reason is in the dataset's own construction. MNIST digits are already size-normalised and centred, so translating them manufactures inputs the test distribution does not contain, and a fully connected network — which has no translation invariance to exploit — spends capacity on them. What buys accuracy at this architecture is the kind of elastic and affine distortion behind Simard's 1.1%, not a one-pixel nudge.

**256 hidden units are genuinely better, and cost twice as much.**

| Hidden units | Validation over 5 seeds | Time per epoch |
| --- | --- | --- |
| 128 | 98.25% ± 0.09 | 5.35 s |
| 256 | **98.41% ± 0.06** | ~10 s |

Paired difference +0.16 points; sd 0.10, standard error 0.04, t = 3.6 on 4 degrees of freedom. That one is real.

The default stays at 128 anyway, and the reason is the trade rather than the number: doubling the width doubles the parameters and the epoch time to buy a sixth of a point. Turning augmentation off was free, so the measurement decided it; widening the network is not, so the measurement informs a decision instead of making it. `--hidden 256` is one flag away for anyone who wants the accuracy more than the runtime.

Neither ablation was decided on a test number. The test set has been scored for the shipped defaults on both splits, and for the previous default before it was retired — never to choose between arms.

## Post-training quantisation

The trained network is 101,770 float32 parameters, 407 KB. This section makes them smaller *after* training — no retraining, no fine-tuning — and measures what that costs.

The method is Optimal Brain Quantisation and GPTQ (Frantar, Ashkboos, Hoefler, Alistarh, [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)), applied to a 784-128-10 MLP. **This is not "GPTQ in C".** It is per-layer post-training quantisation with an output-reconstruction objective, on a network four orders of magnitude smaller than the ones that paper is about. Which of its conclusions transfer, and which are artefacts of working at this scale, is the last part of this section.

### The objective is the output, not the weights

A quantiser has to choose which `W_hat` to keep, and the obvious criterion — make `W_hat` as close to `W` as possible — is the wrong one. What the next layer sees is not the weights, it is `W X`. Two weight matrices equally far from the original in parameter space can produce very different layer outputs: an error on a column whose pixel is dark in every image costs nothing, and the same error on a column carrying signal costs a great deal. So the objective is

```
minimise  || W X  -  W_hat X ||_F^2     over a calibration set X,      not  || W - W_hat ||
```

The Hessian of that objective with respect to a row of `W` is `2 X X^T` — the same matrix for every row of the layer, so it is built once per layer and shared by all 128 of them.

### The algorithm

```
H = X X^T                                  accumulated in double
H += lambda * mean(diag(H)) * I            damping, lambda = 0.01
U = cholesky(H^-1)                         upper triangular, U^T U = H^-1

for each column j:
    for each row r:
        q       = quantize(W[r][j])        b-bit grid, per row
        err     = (W[r][j] - q) / U[j][j]
        W[r][j] = q
        W[r][j+1:] -= err * U[j][j+1:]     compensate with what is not yet quantised
```

The last line is the whole method: the error introduced when a column is rounded is pushed into the columns that have not been quantised yet, which still have the freedom to absorb it. Delete that line and what remains is round-to-nearest — the baseline every table below compares against.

It is also the line that is easy to get subtly wrong, and the reason it is right is not obvious. Fix a row and write the objective in terms of `d = W_hat - W`: it is `d H d^T`. Once columns `0..j-1` are frozen, the free block `F = {j..in-1}` sits at the minimiser of that quadratic, so constraining column `j` to a rounded value and re-minimising is the Optimal Brain Surgeon step, `c = -(e / [A^-1]_jj) * (A^-1)_{:,j}` with `A = H_FF`. GPTQ's contribution is that this needs no per-column submatrix inverse: with `M = H^-1 = U^T U` and `U` partitioned into the frozen and free blocks, the block-inverse identity collapses to

```
(H_FF)^-1 = M_FF - M_FQ M_QQ^-1 M_QF = U_FF^T U_FF
```

and since `j` is the first index of `F` and `U_FF` is upper triangular, `[A^-1]_jk / [A^-1]_jj = U[j][k] / U[j][j]`. One Cholesky of one inverse, computed once per layer, serves every column of every row.

At 784 x 784 the factorisation, the inversion and the refactorisation together are about a third of a second of hand-written `O(n^3)` in `src/linalg.c`, with no BLAS. Quantising the whole network takes **0.34 s** with GPTQ and **0.07 s** with round-to-nearest. That is the reason this project is feasible in C and "GPTQ on an LLM in C" is not.

**Precision.** `H`, its inverse and the Cholesky factor are `double` in both builds, including the float32 one, for the reason `src/real.h` gives for the gradient check: accumulating `X X^T` over thousands of samples and inverting a 784 x 784 matrix does not survive seven decimal digits. The weights stay `real`; the sweep runs on a double working copy, because the correction accumulates across 784 columns and that accumulation is exactly what float32 would lose.

**The grid** is uniform, asymmetric and per row — `scale = (max - min) / (2^b - 1)`, `zero = min` — computed once from the row's *original* weights. It has to be computed before the sweep starts: the sweep moves the weights around as it compensates, and a grid that chased them would be fitting itself to its own error. Grouped grids, one scale per block of columns, are the obvious next refinement and are not implemented.

**Biases are not quantised** — 138 of 101,770 parameters, and no part of the objective describes them.

### Is the compensation actually optimal?

Same problem as the gradient check, same answer. A quantiser with a sign error still produces a network, still classifies digits, and still loses a plausible-looking amount of accuracy as the bit width drops. An accuracy table cannot tell "the compensation is optimal" apart from "the compensation points roughly the right way".

`tests/quantcheck.c` is eight checks, built twice like the gradient check, and both builds gate CI.

```
$ make check                           # C1-C8, hermetic, no dataset required
$ ./build/quantcheck64 --data dataset  # and C1/C2 again on the real MNIST Hessian
```

| | Check | Measured |
| --- | --- | --- |
| **C1** | `U^T U` reconstructs the matrix it factorised — random SPD 8x8 and 64x64, a synthetic 784x784 with a dead border, and the real MNIST 784x784 | **6.4e-16** relative, worst case |
| **C2** | `H H^-1 - I` | **1.4e-13** max, worst case (784x784, real MNIST) |
| **C3** | a grid that rounds nothing leaves `W` bit-identical and the layer error *exactly* 0.0 | exact |
| **C4** | after `j` columns are fixed, the incremental update equals the exact least-squares solution for the rest | **2.2e-16** |
| **C5** | GPTQ never loses to round-to-nearest on the objective it minimises | 0 violations in 130 comparisons |
| **C6** | at 16 bits, the largest weight movement relative to the row's range, and predictions unchanged | **3.2e-05**, 0 of 512 changed |
| **C7** | same seed, same calibration set, same bits → byte-identical weight file | identical |
| **C8** | calibration indices lie inside the 54k training split and are disjoint from the 6k validation split | `abort()` on violation |

**C4 is the one that matters.** It is the analogue of the gradient check. Stop the sweep after `j` columns — those are now fixed at their quantised values. Then solve, independently and with no reference to the update rule, the normal equations `H_FF d_F = -H_FQ d_Q` for what the remaining columns *should* be. The two agree to **2.2e-16**: machine precision, at every prefix, on every row, at 2, 3 and 4 bits and at two damping levels. That is the difference between having derived the compensation and having guessed it.

Two checks shaped the code rather than merely confirming it. C3 demands bit-identity, which forced the sweep to skip the propagation when the error is exactly zero — `w -= 0.0 * u` flips the sign of a weight that happens to be `-0.0`, and "very nearly identical" is not what C3 asks for. And C1's fixture is built from images with MNIST's dead border on purpose: **144 of the 784 columns** of the first layer's Hessian are exactly zero, because those pixels are dark in every one of 2,048 calibration images. `X X^T` is singular before damping, and a fixture without a dead border would never exercise the ridge term.

### What it costs

`scripts/quant_seeds.sh 5` — five seeds, each trained from scratch, quantised, and scored once on the test set by `./mnist test`. Calibration is 512 images drawn from the 54k training split. The held-out column is the same objective measured on 2,048 validation images the calibration never saw. Layer errors are `w1 / w2`, per calibration sample, so rows with different `N` compare directly.

| Bits | Method | Layer error, calibration | Layer error, held out | Test accuracy | Bytes | vs float32 |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | — (baseline) | — | — | **98.18% ± 0.04** | 407,080 | 1.00x |
| 8 | RTN | 0.0096 / 0.017 | 0.0096 / 0.017 | 98.19% ± 0.04 | 103,288 | 3.94x |
| 8 | GPTQ | 0.00060 / 0.0051 | 0.0016 / 0.0060 | 98.18% ± 0.03 | 103,288 | 3.94x |
| 6 | RTN | 0.164 / 0.286 | 0.164 / 0.285 | 98.17% ± 0.03 | 77,880 | 5.23x |
| 6 | GPTQ | 0.0099 / 0.081 | 0.027 / 0.099 | 98.20% ± 0.06 | 77,880 | 5.23x |
| 4 | RTN | 2.87 / 4.50 | 2.89 / 4.45 | 98.13% ± 0.08 | 52,472 | 7.76x |
| 4 | GPTQ | 0.175 / 1.50 | 0.476 / 1.78 | 98.15% ± 0.06 | 52,472 | 7.76x |
| 3 | RTN | 14.0 / 25.1 | 14.1 / 25.0 | 97.65% ± 0.11 | 39,768 | 10.24x |
| 3 | GPTQ | 0.801 / 6.96 | 2.19 / 8.33 | **98.02% ± 0.05** | 39,768 | 10.24x |
| 2 | RTN | 138 / 135 | 138 / 135 | 89.99% ± 2.80 | 27,064 | 15.04x |
| 2 | GPTQ | 4.41 / 38.7 | 12.0 / 44.8 | **97.24% ± 0.20** | 27,064 | 15.04x |

Mean ± sample standard deviation over seeds 1–5. The float32 baseline row is 98.18% ± 0.04 — the same number, to both digits, as the 54k-split row in [Results](#results) above, which `scripts/run_seeds.sh` measured separately. That agreement is not a result, but it is the check that says these five runs are the same protocol and not a differently-configured one.

Bytes are what the quantised parameters would occupy *packed* — `out*in*b/8` for the codes, a float32 scale and zero point per row, biases in float32. The weight files this project writes are still float32 with the quantised values stored dequantised, so that column is the size the grid implies, not the size of anything on disk.

Three things in that table are worth separating.

**GPTQ wins on its own objective everywhere, by a lot.** At every bit width and in both layers, its calibration error is several times lower than round-to-nearest's — and so is its held-out error, so this is a better-conditioned solution and not an artefact of fitting 512 images. Check C5 asserts the ordering on every run, and across the protocol it held **130 times out of 130** — every comparison in which both methods actually quantised the layer.

**That advantage does not become accuracy until 3 bits.** At 8, 6 and 4 bits both methods land inside the baseline's ±0.04 seed spread, even where GPTQ's layer error is sixteen times smaller. Below some threshold the quantisation error is already well under the noise the network carries anyway, and reducing it further buys nothing measurable. Minimising `||WX - W_hat X||` on 512 images is not the same objective as classifying digits, and this is the gap between them: the internal metric improves monotonically while the external one stays flat. Reporting the two separately is the only way to see that, which is why `quant_report` exists.

**At 3 bits and below it becomes the whole story.** GPTQ gives **+0.37 points** at 3 bits (98.02 against 97.65) and **+7.25 points** at 2 bits (97.24 against 89.99). At 2 bits the standard deviation is the other half of the finding: RTN's is **± 2.80**, seventy times the baseline's, so round-to-nearest at 2 bits is not merely worse but unstable — which seed you trained with decides how bad it is. GPTQ's is ± 0.20.

### The layer that matters

`w2` is 10 x 128 — **1,280 weights, 1.3% of the total**. `w1` is the other 98.7%. Quantising each alone says where the damage actually comes from.

| Bits | Method | Quantised | Test accuracy | Change vs baseline | Bytes |
| --- | --- | --- | --- | --- | --- |
| 4 | RTN | w1 only | 98.17% ± 0.04 | −0.02 | 56,872 |
| 4 | RTN | w2 only | 98.13% ± 0.06 | −0.05 | 402,680 |
| 4 | RTN | both | 98.13% ± 0.08 | −0.06 | 52,472 |
| 4 | GPTQ | w1 only | 98.19% ± 0.07 | +0.00 | 56,872 |
| 4 | GPTQ | w2 only | 98.17% ± 0.06 | −0.02 | 402,680 |
| 4 | GPTQ | both | 98.15% ± 0.06 | −0.04 | 52,472 |
| 2 | RTN | w1 only | 94.86% ± 0.75 | −3.33 | 31,784 |
| 2 | RTN | w2 only | 96.35% ± 0.28 | −1.83 | 402,360 |
| 2 | RTN | both | 89.99% ± 2.80 | −8.20 | 27,064 |
| 2 | GPTQ | w1 only | 98.03% ± 0.02 | −0.15 | 31,784 |
| 2 | GPTQ | w2 only | 97.53% ± 0.13 | −0.66 | 402,360 |
| 2 | GPTQ | both | **97.24% ± 0.20** | −0.95 | 27,064 |

At 4 bits nothing is happening in either layer. At 2 bits the asymmetry is stark, and it is sharpest for the method that works:

**With GPTQ at 2 bits, the 1.3% of the weights in `w2` do four times the damage of the other 98.7%** — 0.66 points against 0.15. Per weight that is a factor of roughly 350. Round-to-nearest shows the same ordering less extremely: 1.83 points from `w2` against 3.33 from a matrix seventy-eight times larger, about 43x per weight.

And `w2` is not where the compression is. Quantising it to 2 bits and leaving `w1` in float32 saves 4,720 bytes — **1.2%** of the model — for 0.66 points. Quantising `w1` alone saves 92%, for 0.15. Keeping the small layer at full precision is very nearly free and, on these numbers, clearly worth it: `w1` at 2 bits with `w2` untouched is 98.03% ± 0.02 at 12.8x compression, better than quantising both and barely distinguishable from the float32 baseline.

That is, in miniature, why a production mixed-precision recipe is never "quantise everything to *b* bits". DeepSeek V4 Flash quantises its routed experts to about 2 bits while leaving attention, the projections and the router at higher precision, and the argument has the same shape as this table: spend bits where the parameter count is small and the sensitivity is high, save them where the parameter count is large and the redundancy is real. The layer that is 1.3% of the storage is not where the savings are, so paying full precision for it costs almost nothing.

### How much calibration data

`X X^T` for the first layer is 784 x 784, so it cannot have full rank until the calibration set holds more than 784 samples. Below that the damping term is carrying the rank deficiency, and the question is whether the result is an objective fitted to the calibration set rather than to the layer.

| Bits | Method | N | Layer error w1, calibration | Layer error w1, held out | Ratio | Test accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | RTN | 128 | 2.90 | 2.89 | 1.00 | 98.13% ± 0.08 |
| 4 | RTN | 512 | 2.87 | 2.89 | 1.01 | 98.13% ± 0.08 |
| 4 | RTN | 2048 | 2.87 | 2.89 | 1.01 | 98.13% ± 0.08 |
| 4 | GPTQ | 128 | 0.051 | 0.837 | **16.55** | 98.13% ± 0.07 |
| 4 | GPTQ | 512 | 0.175 | 0.476 | 2.73 | 98.15% ± 0.06 |
| 4 | GPTQ | 2048 | 0.268 | 0.339 | 1.26 | 98.14% ± 0.06 |
| 3 | RTN | 128 | 14.3 | 14.1 | 0.99 | 97.65% ± 0.11 |
| 3 | RTN | 512 | 14.0 | 14.1 | 1.01 | 97.65% ± 0.11 |
| 3 | RTN | 2048 | 14.0 | 14.1 | 1.00 | 97.65% ± 0.11 |
| 3 | GPTQ | 128 | 0.233 | 3.82 | **16.39** | 97.96% ± 0.04 |
| 3 | GPTQ | 512 | 0.801 | 2.19 | 2.74 | 98.02% ± 0.05 |
| 3 | GPTQ | 2048 | 1.24 | 1.57 | 1.26 | 98.05% ± 0.11 |

Round-to-nearest never looks at `X`, and the table confirms it: for a given seed, the held-out errors across all three calibration sizes are *bit-identical* (3.025875e+00, three times over, for seed 1), and so is the test accuracy. That is a useful control rather than a triviality — anything leaking calibration data into the round-to-nearest path would move those numbers, and they do not move.

For GPTQ the ratio column is the finding, and it barely depends on the bit width. At **N = 128** the objective on the calibration set is **16.5x** better than the same objective off it: with 128 samples and 784 columns, `X X^T` has rank at most 128, and the compensation is solving a problem that mostly does not exist. At **N = 512**, still below 784, the ratio is 2.7. At **N = 2048**, past the point where the Hessian can be full rank, it is 1.26 — the calibration error has become an honest estimate of the layer error.

What is striking is how little of that reaches accuracy: at 4 bits it never moves at all, and at 3 bits the gap between 128 and 2048 samples is +0.09 points, comparable to the ±0.11 seed spread. The damping is what absorbs it — a ridge of 1% of `mean(diag(H))` keeps the compensation conservative enough that a badly estimated Hessian degrades gracefully instead of producing confident nonsense. Damping is usually introduced as a numerical convenience for making the Cholesky succeed; this says it is also what makes a small calibration set safe.

### Two bits

The prediction, written into the plan before any of this was implemented, was: *8 bits indistinguishable; 6 and 4 the band where GPTQ justifies itself; 3 bits GPTQ degraded but usable, RTN clearly worse; 2 bits probably a collapse, even with GPTQ.*

Two of those four were wrong, and how they were wrong is the interesting part.

**The 6-and-4-bit band did not separate the methods.** GPTQ's layer error there is up to sixteen times lower and it buys nothing: both methods sit inside the baseline's ±0.04 seed spread. The band where the method justifies itself is 3 bits and below, not 4 and above.

**And 2 bits did not collapse — for GPTQ.** It lands at **97.24% ± 0.20**, 0.95 points below the float32 baseline, at 15x compression. RTN does collapse, to 89.99% ± 2.80, exactly as predicted; the gap between them is the largest single effect in this section. The prediction was right about the cliff and wrong about who falls off it.

The honest reading is not that 2-bit quantisation works. It is that **this network has room to be wrong in.** 101,770 parameters fitted to 54,000 images, driven to a training loss below 0.001 — it memorises its training set with capacity left over, and the compensation step has up to 783 not-yet-quantised columns to spread each rounding error across while it does so. MNIST at 98% is a problem with a great deal of slack in it.

Which is precisely why these numbers do not transfer.

A 100k-parameter MLP and a several-hundred-billion-parameter transformer fail under quantisation for different reasons and in different places. This network has no attention, no layer norm, no residual stream, and no outlier activation channels — the phenomenon that made naive LLM quantisation fail in the first place, and what a large part of that literature is actually about. It has two layers rather than a hundred, so error does not compound through depth. Its calibration set is drawn from the same distribution it was trained on, rather than a few hundred sequences standing in for the internet. And a 2-bit result over ten output classes says nothing about a 2-bit result over a 128k-token vocabulary.

What does transfer is the shape of the argument, and it is worth being precise about which parts:

- **The objective should be output reconstruction, not weight proximity.** That is a statement about layers, and it holds at any scale.
- **The per-layer error and the end-to-end accuracy are different quantities**, and a method can dominate on the first while making no measurable difference to the second. The 4-bit rows are a small, clean demonstration.
- **Which layers you spend bits on matters more than the average bit width.** The per-layer ablation is the miniature version of why DeepSeek V4 Flash quantises routed experts to 2 bits and leaves attention, the projections and the routing alone.

What does not transfer is any single number in this section. A large model's redundancy is not this network's redundancy, and a small MLP is close to the worst case for quantisation precisely because every weight in it carries so much signal. If anything, the surprise is that 2-bit GPTQ works *this* well on a network this small — and the reason to say so out loud is that it is the opposite of what was predicted.

### Reproducing it

```
scripts/quant_seeds.sh 5                    # everything above: 5 seeds, 155 measured configurations
EXPERIMENTS=main scripts/quant_seeds.sh 5   # just the bits x method grid
make check                                  # C1-C8, including the C4 least-squares equivalence
./build/quantcheck64 --data dataset         # C1 and C2 on the real MNIST Hessian
```

`build/quant_sweep --help` lists the grid options. It reports validation accuracy and layer errors and never opens the test files; `scripts/quant_seeds.sh` is what scores the test set, once per configuration, with `./mnist test`.

**Two stated limitations, one of them measured.**

Both layers' calibration activations are collected in one pass from the *unquantised* network. For a deep model GPTQ re-collects each layer's inputs from the already-quantised layers below, so errors do not compound. With two layers the sequential variant is two invocations of the same driver, so it is cheap to check rather than assert — quantise `w1`, then feed the result back in and quantise `w2` against activations re-collected from it:

```
./build/quant_sweep --weights network.dat  --out runs/seq --tag a \
    --bits 2 --method gptq --layers w1
./build/quant_sweep --weights runs/seq/a.gptq.b2.w1.dat --out runs/seq --tag b \
    --bits 2 --method gptq --layers w2
```

At 2 bits, over the same five seeds: **97.28% ± 0.17 sequential against 97.24% ± 0.20 one-pass**, a paired difference of +0.04 points with a standard error of 0.07 (t = 0.6 on 4 df). The one-pass shortcut is a real approximation and, at this depth, it costs nothing measurable. At a hundred layers it would be a different answer.

The second limitation is not measured: the quantisation grid is one scale per row, not per group of columns. Grouped grids are the standard next refinement and would most likely move the 2-bit numbers.

## Performance

Measured on a MacBook Air (Apple M2, 8 cores, 8 GB, macOS 26.5.2), Apple clang 21, `-O2`. `make bench`.

### Loading the training set — 60,000 images, 47 MB

| Version | Time | Speedup |
| --- | --- | --- |
| one `fread` per pixel, one `malloc` per image (original) | 0.981 s | 1.00x |
| one `fread` per image, one `malloc` per image | 0.018 s | 54x |
| one read for the whole file, one contiguous allocation (current) | 0.011 s | **90x** |

47 million one-byte `fread` calls is the whole of the first number. Each one crosses into the C library to copy a single byte out of a buffer it already holds. Reading the file in one call and converting afterwards removes all of it; allocating the pixels as one block rather than 60,000 of them removes the rest.

Best of seven, after one discarded pass to warm the page cache — without it the first timed read also pays for pulling 47 MB off disk, and the benchmark measures the storage rather than the parsing, which is the part that changed. Repeated runs put the final ratio between 82x and 91x, so **~88x** is the number worth quoting rather than the best single figure.

This is a load-time saving, not a per-epoch one — the dataset is read once.

### rand() against PCG

| | ns per draw |
| --- | --- |
| `rand()` | 6.71 |
| `prng_rand()` (PCG-XSH-RR) | 2.52 |

2.7x per draw, which is worth about **1 millisecond per epoch**: the training loop draws roughly four numbers per sample, and an epoch takes 5.35 seconds. Speed was never the reason to replace `rand()`. The reason is that its algorithm and its sequence are implementation-defined, so a fixed seed reproduces a run on one machine and one libc and nowhere else. PCG makes `--seed` mean something.

### Training

Median **5.35 s per epoch** over 100 epochs at 784-128-10 on 54,000 samples (min 5.05 s). The machine is fanless, so it clocks down under sustained load and the mean drifts above the median; the median over a full protocol run is the honest figure to quote, not the first epoch of a cold run.

That number is **20% worse than it needs to be**, and on purpose.

Gradients are materialised into a `gradients` struct and applied by a separate `net_sgd_step`, instead of being folded into the backward pass the way the previous version did. The weight arrays get written and read again rather than updated in place, and the cost of that is measurable — building the old implementation out of git history and timing both on the same 60,000 samples, same augmentation setting, best of six epochs each:

| Update path | Seconds per epoch, 60k samples |
| --- | --- |
| fused into the backward pass (previous) | 4.90 |
| explicit gradients, separate SGD step (current) | 5.86 |

```
git show 6cb0e1b:main.c > /tmp/old.c        # the fused version
```

A fifth of the epoch time is what a checkable gradient costs here, because a gradient that is never a value cannot be compared against anything. It is the one place in this project where a measured cost was accepted deliberately, and it seemed worth writing the number down rather than describing it as free.


## Architecture

```
input          hidden           output
 784    -->     128      -->      10
(28x28)       (ReLU)          (softmax)
```

| Component | Choice |
| --- | --- |
| Hidden layer | 128 units, ReLU |
| Output layer | 10 units, softmax |
| Loss | Cross-entropy, evaluated as `logsumexp(z) - z[label]` |
| Optimiser | Stochastic gradient descent, per-sample updates |
| Initialisation | Xavier/Glorot uniform, scaled per layer fan-in |
| RNG | PCG-XSH-RR, seeded explicitly |
| Augmentation | ±1 pixel translation, available via `--aug`, off by default (measured, see above) |
| Defaults | 20 epochs, learning rate 0.01, reshuffled every epoch |

Weights are flat `float` arrays rather than arrays of pointers, so each layer's parameters occupy one contiguous allocation and index arithmetic stays cache-friendly. The whole dataset is one allocation too.

## Requirements

- A C compiler (`gcc` or `clang`) and `make`
- `curl` or `wget`, to fetch the dataset

## Dataset

```
scripts/get_dataset.sh          # into ./dataset
```

The script downloads the four IDX archives, decompresses them, and verifies each against a known SHA-256 digest, refusing anything that does not match. It tries several mirrors in turn: the original `yann.lecun.com` URLs have been unreliable for years, and a truncated download is a worse failure than a missing one because it still parses.

```
dataset/
  train-images.idx3-ubyte     60,000 training images    47,040,016 bytes
  train-labels.idx1-ubyte     60,000 training labels         60,008 bytes
  t10k-images.idx3-ubyte      10,000 test images          7,840,016 bytes
  t10k-labels.idx1-ubyte      10,000 test labels             10,008 bytes
```

## Build and run

```
make                # ./mnist
make check          # unit tests, both gradient checks, both quantiser checks
make sanitize       # the same tests under ASan + UBSan
make bench          # loader and RNG benchmark
make quant          # build/quant_sweep, the post-training quantisation driver
```

```
./mnist train                       # 20 epochs, validation accuracy each epoch
./mnist train --hidden 256          # +0.16 points, twice the epoch time
./mnist test                        # the final measurement
./mnist --help
```

`train` writes `network.dat`; `test` reads it. `--resume` continues from an existing file, which is off by default so that a stale `network.dat` cannot silently change a result.

## How it works

1. **IDX parsing.** The format stores integers big-endian, so every header field is byte-swapped on read (`reverse_int`), and the magic numbers — 2051 for images, 2049 for labels — are checked before anything is allocated. The pixel block is read in a single `fread` into one contiguous allocation and converted to normalised floats afterwards.

2. **Forward pass.** Each hidden unit takes a dot product over the 784 inputs plus a bias and applies ReLU; the output layer produces ten logits.

3. **Loss.** Cross-entropy from the logits, as `logsumexp(z) - z[label]`, with the maximum subtracted for overflow and cancelled analytically for precision.

4. **Backward pass.** The gradient of cross-entropy with respect to the softmax logits collapses to `p - onehot(label)`, which makes the output delta a single subtraction. That delta propagates back through `w2` to give the hidden delta, masked by the ReLU derivative. For a single sample the bias gradient of a layer is exactly its delta, so `db1`/`db2` double as the delta buffers. Rows of `dw1` behind a switched-off unit are zeroed rather than skipped, so every entry is defined after the call — the gradient check reads them.

5. **Update.** `net_sgd_step` applies `theta -= lr * grad`. It is a separate function on purpose; see above.

6. **Persistence.** `net_save` writes a magic number, the three layer sizes, then the parameters — always as float32, whatever precision the build uses, so the float and double builds share a file format. `net_load` refuses a geometry that does not match the network in memory, and refuses a file that is not a network file.

7. **Memory.** Every allocation has a matching free and every failure path unwinds what preceded it. `make sanitize` is the check on that claim.

## Project layout

```
src/real.h        precision switch: float32 to train, float64 to gradient-check
src/idx.[ch]      IDX parser, dataset storage, train/validation split
src/net.[ch]      network, forward, loss, gradients, SGD step, serialisation
src/train.[ch]    augmentation, epoch loop, evaluation
src/linalg.[ch]   Cholesky, triangular solves, SPD inverse -- double, no BLAS
src/quant.[ch]    Hessian, quantisation grid, the GPTQ sweep and the RTN baseline
src/main.c        command line: train and test
tests/gradcheck.c  numerical gradient check
tests/quantcheck.c quantiser correctness suite, C1-C8
tests/test_units.c unit tests, no dataset required
tools/bench_load.c  loader and RNG benchmark
tools/quant_sweep.c quantisation sweep driver (never opens the test set)
scripts/          dataset fetch, hyperparameter sweep, paired ablation, multi-seed protocols
basic_prng/       PCG-XSH-RR generator
.github/workflows/ci.yml
```

## Tests

```
make check       # unit tests, both gradient checks, both quantiser checks -- no dataset required
make sanitize    # the same, under AddressSanitizer and UndefinedBehaviorSanitizer
```

Every fixture is synthesised, so the test suite is hermetic and runs in seconds. CI builds with gcc and clang on Linux and clang on macOS at `-Wall -Wextra -Wpedantic -Werror`, runs the tests and the sanitizers, and separately fetches the dataset and trains an epoch — which is also the test that `scripts/get_dataset.sh` still works, so a mirror going dark fails in CI rather than in someone's clone.


## AI full disclosure

This software is developed with assistance from GPT 5.5, 5.6, Claude Fable and with humans leading the ideas, testing, and debugging.

## Roadmap

- [x] Shuffle the training set between epochs
- [x] Numerical gradient check against a central difference
- [x] Train/validation split, with validation accuracy reported per epoch
- [x] Results over several seeds, with a standard deviation
- [x] Dataset fetched and checksum-verified by script
- [x] Split into modules; delta buffers sized at runtime rather than fixed on the stack
- [ ] Mini-batch gradient descent — now a small change, since the gradients already exist as a value to accumulate into
- [ ] Keep the best-validation weights rather than the last epoch's; worth about 0.05 points, measured
- [ ] Elastic and affine distortions. A ±1 pixel translation measurably does not help; Simard's 1.1% says the stronger distortions are where the accuracy actually is
- [x] Post-training quantisation with an output-reconstruction objective (OBQ/GPTQ), 8 to 2 bits, verified against a brute-force least-squares solve
- [ ] A confusion matrix, to see which digits the errors are concentrated in
- [ ] Grouped quantisation grids — one scale per block of columns rather than one per row
- [ ] Re-collect the second layer's calibration activations from the already-quantised first layer, as GPTQ does for deep models

## References

- Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, *Gradient-Based Learning Applied to Document Recognition*, Proceedings of the IEEE, 86(11), 1998 — the source of the MNIST dataset and of the baseline error rates quoted above.
- P. Y. Simard, D. Steinkraus, J. C. Platt, *Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis*, ICDAR 2003 — the 1.6% and 1.1% two-layer results.
- M. E. O'Neill, *PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation*, 2014 — the generator in `basic_prng/`.
- E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh, *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, arXiv:2210.17323, 2022 — the algorithm in `src/quant.c`.
- E. Frantar, D. Alistarh, *Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning*, NeurIPS 2022 — the Optimal Brain Quantisation step GPTQ builds on.
- B. Hassibi, D. G. Stork, *Second Order Derivatives for Network Pruning: Optimal Brain Surgeon*, NeurIPS 1992 — the constrained-quadratic step underneath both.

## License

MIT
