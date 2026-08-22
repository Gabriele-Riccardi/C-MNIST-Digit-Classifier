CC      := gcc
CFLAGS  := -Wall -Wextra -O2
LDFLAGS := -lm
TARGET  := mnist

all: $(TARGET)

$(TARGET): main.c main.h
	$(CC) $(CFLAGS) -o $(TARGET) main.c $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET).exe

.PHONY: all clean
