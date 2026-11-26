CXX ?= c++
CXXFLAGS ?= -O3 -Wall -Wextra -Wpedantic
REQUIRED = -std=c++17 -ffp-contract=off
SOURCES = native/rowmix.cpp native/kernels.cpp
HEADERS = native/rowmix.h native/kernels.h
.PHONY: all test clean benchmark
all: build/rowmix-run build/librowmix.so

build:
	mkdir -p build

build/rowmix-run: $(SOURCES) native/runner.cpp $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(REQUIRED) $(SOURCES) native/runner.cpp -o $@

build/librowmix.so: $(SOURCES) $(HEADERS) | build
	$(CXX) $(CXXFLAGS) $(REQUIRED) -shared -fPIC $(SOURCES) -o $@

test: all
	OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v

benchmark: all
	OPENBLAS_NUM_THREADS=1 python -m benchmarks.run

clean:
	rm -rf build
