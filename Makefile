CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -pthread
PYTHON ?= python3
ifeq ($(shell uname -s),Darwin)
SHARED = -dynamiclib
else
SHARED = -shared
endif

.PHONY: all test clean
all: build/libexpunge.so
build/libexpunge.so: native/graph.cpp $(wildcard third_party/hnswlib/*.h)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -fPIC $(SHARED) -Ithird_party $< -o $@
test: all
	$(PYTHON) -m unittest discover -s tests -v
clean:
	rm -rf build
