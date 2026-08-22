# C-MNIST Digit Classifier

A handwritten-digit classifier written from scratch in **pure C** — no PyTorch, no TensorFlow, no BLAS, no external dependencies of any kind. Just `libc`, `libm`, and about 1,170 lines of source across a loader, a network, a training loop and a command line — plus another 950 lines of tests and benchmarks that check them.

Every part of the pipeline is implemented by hand: the IDX binary parser, the matrix arithmetic, the forward pass, the softmax, the cross-entropy loss, and the backpropagation. Nothing is delegated to a library.

```
$ scripts/get_dataset.sh          # download and verify the four IDX files
$ make check                      # unit tests + numerical gradient check
$ make && ./mnist train           # fit on 54k, score the 6k held out
$ ./mnist test                    # score the 10k test set, once
```

## Every number in this README is reproducible

Backpropagation written by hand fails silently. A wrong sign or a transposed index still trains — the network just learns a bit worse — so an accuracy figure is not evidence that the gradient is correct, only that it is not catastrophically wrong. The same goes for performance claims: "faster" means nothing without a before, an after, and a machine.

So every quantitative claim below has a command next to it.

| Claim | Reproduce it with |
| --- | --- |
| The analytic gradient matches a central difference to 9.2e-08 | `make check` |
| Test accuracy is 98.29% ± 0.07 over five seeds | `scripts/run_seeds.sh 5 -- --val 0` |
| The hyperparameters were chosen on validation, never on test | `scripts/sweep.sh`, `scripts/ablation.sh` |
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

Beyond the gradient, `tests/test_units.c` covers the IDX parser (wrong magic, truncated data, mismatched pairs, out-of-range labels), the split, serialisation, augmentation, softmax overflow, and the PRNG's bounds. `make sanitize` reruns everything under AddressSanitizer and UndefinedBehaviorSanitizer.

## Protocol

The test set is opened by exactly one command, `./mnist test`, and nothing else in the repository reads it. `./mnist train` splits 60,000 training images into **54,000 train / 6,000 validation** and reports validation accuracy after every epoch.

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
make check          # unit tests and both gradient checks
make sanitize       # the same tests under ASan + UBSan
make bench          # loader and RNG benchmark
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
src/main.c        command line: train and test
tests/gradcheck.c numerical gradient check
tests/test_units.c unit tests, no dataset required
tools/bench_load.c loader and RNG benchmark
scripts/          dataset fetch, hyperparameter sweep, paired ablation, multi-seed protocol
basic_prng/       PCG-XSH-RR generator
.github/workflows/ci.yml
```

## Tests

```
make check       # unit tests and both gradient checks, no dataset required
make sanitize    # the same, under AddressSanitizer and UndefinedBehaviorSanitizer
```

Every fixture is synthesised, so the test suite is hermetic and runs in seconds. CI builds with gcc and clang on Linux and clang on macOS at `-Wall -Wextra -Wpedantic -Werror`, runs the tests and the sanitizers, and separately fetches the dataset and trains an epoch — which is also the test that `scripts/get_dataset.sh` still works, so a mirror going dark fails in CI rather than in someone's clone.

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
- [ ] A confusion matrix, to see which digits the errors are concentrated in

## References

- Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, *Gradient-Based Learning Applied to Document Recognition*, Proceedings of the IEEE, 86(11), 1998 — the source of the MNIST dataset and of the baseline error rates quoted above.
- P. Y. Simard, D. Steinkraus, J. C. Platt, *Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis*, ICDAR 2003 — the 1.6% and 1.1% two-layer results.
- M. E. O'Neill, *PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation*, 2014 — the generator in `basic_prng/`.

## License

MIT
