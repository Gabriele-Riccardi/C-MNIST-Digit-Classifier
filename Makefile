CC      := gcc
CFLAGS  := -Wall -Wextra -O2 -Ibasic_prng
LDFLAGS := -lm
TARGET  := mnist
SRCS    := main.c basic_prng/prng.c

all: $(TARGET)

$(TARGET): $(SRCS) main.h basic_prng/prng.h
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET).exe

.PHONY: all clean
