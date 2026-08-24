# C-MNIST Digit Classifier
#
#   make            build ./mnist
#   make check      build and run every test (unit tests, gradient check, quantiser check)
#   make bench      build and run the loader / RNG benchmark
#   make quant      build the quantisation sweep driver
#   make sanitize   run the tests under ASan + UBSan
#   make clean

CC       ?= cc
STD      := -std=c11
WARN     := -Wall -Wextra -Wpedantic -Wshadow -Wvla -Wstrict-prototypes
OPT      ?= -O2
INCLUDES := -Isrc -Ibasic_prng
CFLAGS   := $(STD) $(WARN) $(OPT) $(INCLUDES) -MMD -MP $(EXTRA_CFLAGS)
LDFLAGS  += $(EXTRA_LDFLAGS)
LDLIBS   := -lm

BUILD    := build
LIB_SRCS := src/idx.c src/net.c src/train.c src/linalg.c src/quant.c basic_prng/prng.c

# The library is compiled twice: once in the float32 the network trains in, and
# once in float64 for the gradient check (see src/real.h for why). The
# quantiser check is built both ways too: its linear algebra is double in either
# build, so the two binaries have to agree, and a discrepancy would mean some
# part of it had quietly picked up the network's precision.
F_OBJS := $(patsubst %.c,$(BUILD)/f/%.o,$(LIB_SRCS))
D_OBJS := $(patsubst %.c,$(BUILD)/d/%.o,$(LIB_SRCS))

TESTS := $(BUILD)/gradcheck64 $(BUILD)/gradcheck $(BUILD)/test_units \
         $(BUILD)/quantcheck64 $(BUILD)/quantcheck

.PHONY: all check bench quant sanitize clean

all: mnist

mnist: $(BUILD)/f/src/main.o $(F_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/gradcheck: $(BUILD)/f/tests/gradcheck.o $(F_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/gradcheck64: $(BUILD)/d/tests/gradcheck.o $(D_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/test_units: $(BUILD)/f/tests/test_units.o $(F_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/quantcheck: $(BUILD)/f/tests/quantcheck.o $(F_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/quantcheck64: $(BUILD)/d/tests/quantcheck.o $(D_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/quant_sweep: $(BUILD)/f/tools/quant_sweep.o $(F_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/bench_load: $(BUILD)/f/tools/bench_load.o $(F_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/f/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD)/d/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DNET_REAL_DOUBLE -c $< -o $@

check: $(TESTS)
	@set -e; for t in $(TESTS); do \
	    echo "=== $$t ==="; \
	    ./$$t; \
	    echo; \
	done; \
	echo "all tests passed"

# Same tests, instrumented. Runs slower; catches the out-of-bounds write that a
# green test suite would otherwise sail past.
# Built into its own directory so instrumented objects can never be linked into
# the ordinary binary.
sanitize:
	@$(MAKE) --no-print-directory check \
	    BUILD=$(BUILD)/asan \
	    OPT="-O1 -g -fno-omit-frame-pointer" \
	    EXTRA_CFLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all" \
	    EXTRA_LDFLAGS="-fsanitize=address,undefined"

bench: $(BUILD)/bench_load
	./$(BUILD)/bench_load

# Built, not run: a sweep needs a trained network and takes minutes.
# scripts/quant_seeds.sh drives it.
quant: $(BUILD)/quant_sweep
	@echo "built $(BUILD)/quant_sweep -- see scripts/quant_seeds.sh"

clean:
	rm -rf $(BUILD) mnist mnist.exe quantcheck_*.tmp

-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)
