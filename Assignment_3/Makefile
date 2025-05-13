CC = mpicxx
CFLAGS = -Wall -std=c++17 -O3 -g
BINS = quicksort

all: $(BINS)

quicksort: quicksort.h pivot.h main.cpp
	$(CC) $(CFLAGS) -o $@ main.cpp

clean:
	$(RM) $(BINS)

