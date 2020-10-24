CC=cc
OPT=-O3
CFLAGS=-Wall -Wextra -Wpedantic -std=gnu89 $(OPT)
OBJECTS=$(patsubst src/%.c,obj/%.o,$(wildcard src/*.c))

libgauss.so: $(OBJECTS)
	$(CC) -shared -o libgauss.so $(OBJECTS)

$(OBJECTS): src/*.c
	$(CC) -c -o $@ $(CFLAGS) $<

.PHONY: clean
clean:
	rm -f obj/*
	rm -f libgauss.so
