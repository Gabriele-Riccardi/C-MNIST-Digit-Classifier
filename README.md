# C-MNIST Digit Classifier

A handwritten-digit classifier written from scratch in **pure C** — no PyTorch, no TensorFlow, no BLAS, no external dependencies of any kind. Just `libc`, `libm`, and about 600 lines of code.

Every part of the pipeline is implemented by hand: the IDX binary parser, the matrix arithmetic, the forward pass, the softmax, the cross-entropy loss, and the backpropagation. Nothing is delegated to a library.

```
$ make && ./mnist
Images and labels loaded successfully.
Number of training images: 60000
Number of testing images: 10000
No existing network found. Starting from scratch.
Epoch 1 in 4.94 s, Loss: 0.255523
Epoch 2 in 4.98 s, Loss: 0.126921
Epoch 3 in 4.75 s, Loss: 0.097708
...
Epoch 19 in 4.91 s, Loss: 0.041599
Epoch 20 in 5.04 s, Loss: 0.041329
Network saved to network.dat.
Accuracy on test set: 98.38%
```

## Why

Modern ML frameworks make it easy to train a network without ever understanding what happens between `model.fit()` and the printed accuracy. This project exists to close that gap: implementing backpropagation by hand forces you to derive the chain rule for every layer, and getting it wrong is silent — the network still runs, it just never learns.

- **Zero dependencies.** A C compiler and the standard library. No `pip install`, no virtualenv, no CUDA toolkit.
- **Fully transparent.** Every weight update is visible in the source. There is no autograd doing the calculus for you.
- **Numerically careful.** Softmax subtracts the row maximum before exponentiating to avoid overflow; the loss clamps probabilities away from zero before taking the logarithm.
- **Persistent.** Trained weights are serialised to a compact binary file and reloaded on the next run, with a shape check that refuses mismatched architectures.

## Architecture

A fully-connected feedforward network:

```
input          hidden           output
 784    -->     128      -->      10
(28x28)       (ReLU)          (softmax)
```

| Component      | Choice                                              |
| -------------- | --------------------------------------------------- |
| Hidden layer   | 128 units, ReLU activation                          |
| Output layer   | 10 units, softmax                                   |
| Loss           | Categorical cross-entropy                           |
| Optimiser      | Stochastic gradient descent, per-sample updates     |
| Initialisation | Xavier/Glorot uniform, scaled per layer fan-in      |
| RNG            | PCG-XSH-RR, fixed seed for reproducible runs        |
| Augmentation   | Random +/-1 pixel translation, applied to ~50% of samples |
| Default run    | 20 epochs, learning rate 0.01, shuffled each epoch  |

Weights are stored as flat `float` arrays rather than arrays of pointers, so each layer's parameters occupy one contiguous allocation and index arithmetic stays cache-friendly.

## Requirements

- A C compiler (`gcc` or `clang`)
- The MNIST dataset in the original IDX format

The dataset is not included in this repository. Download the four files and place them in a `dataset/` directory:

```
dataset/
  train-images.idx3-ubyte     60,000 training images
  train-labels.idx1-ubyte     60,000 training labels
  t10k-images.idx3-ubyte      10,000 test images
  t10k-labels.idx1-ubyte      10,000 test labels
```

## Build

```
make            # produces ./mnist
make clean
```

Or directly:

```
gcc -Wall -Wextra -O2 -Ibasic_prng -o mnist main.c basic_prng/prng.c -lm
```

## Usage

```
./mnist
```

On the first run the network trains from random initialisation and writes its weights to `network.dat`. On subsequent runs those weights are loaded and training resumes from where it left off. Delete `network.dat` to start over.

## How it works

1. **IDX parsing.** The MNIST format stores integers big-endian. Since x86 and ARM are little-endian, every header field is byte-swapped on read (`reverse_int`). The magic numbers — 2051 for images, 2049 for labels — are verified before anything is allocated, so a wrong file fails loudly instead of producing garbage. Pixels are converted from `unsigned char` to `float` and normalised into [0, 1] during the load itself.

2. **Forward pass.** Each hidden unit computes a dot product over the 784 inputs plus a bias, then applies ReLU. The output layer produces ten raw logits, which softmax converts into a probability distribution.

3. **Backward pass.** The gradient of cross-entropy with respect to the softmax logits collapses to `output - target`, which makes the output-layer delta a single subtraction. That delta is propagated back through `w2` to obtain the hidden-layer delta, masked by the ReLU derivative. **The hidden delta is computed before `w2` is updated** — using the already-updated weights is a subtle bug that leaves the network training slowly rather than failing outright.

4. **Persistence.** `save_network` writes the three layer sizes followed by the raw weight and bias arrays. `load_network` reads the header first and refuses to continue if the stored geometry does not match the network in memory.

5. **Memory.** Every allocation has a matching free, and every failed allocation unwinds the ones that preceded it. The image loader frees each row it already allocated before returning `NULL`, so a failure partway through 60,000 allocations does not leak the first 59,999.

## Results

| Metric              | Value       |
| ------------------- | ----------- |
| Test-set accuracy   | 98.38%      |
| Test-set size       | 10,000      |
| Training set size   | 60,000      |
| Final training loss | 0.0413      |
| Epochs              | 20          |
| Time per epoch      | ~4.9 s      |

## Project layout

```
main.c        IDX loader, network, forward/backward pass, training loop, entry point
main.h        struct definitions, function declarations, IDX format notes
basic_prng/   PCG-XSH-RR generator (uniform, unbiased bounded, Gaussian)
Makefile      build and clean targets
dataset/      MNIST IDX files (not tracked; see Requirements)
network.dat   serialised weights (generated at runtime)
```
## Roadmap

- [ ] Mini-batch gradient descent instead of per-sample updates
- [ ] Report validation accuracy per epoch to visualise the learning curve
- [ ] Allocate the backpropagation delta buffers dynamically to support arbitrary layer widths
## License

MIT
