MAKEFLAGS+=-r -j

UNAME=$(shell uname)

NVCC?=nvcc

BUILD=build
ASM_DIR=$(BUILD)/asm

# compile .c, .cpp, .cu files
SOURCES=$(filter-out src/test.cpp,$(wildcard src/*.c))
SOURCES+=$(filter-out src/test.cpp,$(wildcard src/*.cc))
SOURCES+=$(filter-out src/test.cpp,$(wildcard src/*.cpp))
SOURCES+=$(filter-out src/test.cpp,$(wildcard src/*.cu))
SOURCES+=$(wildcard vendor/*.c)
SOURCES+=$(wildcard vendor/*.cc)
SOURCES+=$(wildcard vendor/*.cpp)
SOURCES+=$(wildcard vendor/*.cu)

# Define test sources separately
TEST_SOURCES=src/test.cpp
TEST_SOURCES+=$(filter-out src/main.cpp,$(SOURCES))

OBJECTS=$(SOURCES:%=$(BUILD)/%.o)
TEST_OBJECTS=$(TEST_SOURCES:%=$(BUILD)/%.o)
ASM_FILES=$(patsubst %.cpp,$(ASM_DIR)/%.s,$(filter %.cpp,$(SOURCES)))
TEST_ASM_FILES=$(patsubst %.cpp,$(ASM_DIR)/%.s,$(filter %.cpp,$(TEST_SOURCES)))

BINARY=$(BUILD)/main
TEST_BINARY=$(BUILD)/test

CFLAGS=-g -Wall -Wpointer-arith -Werror -O3 -ffast-math -Ivendor -std=c++20
LDFLAGS=-lm

CFLAGS+=-fopenmp -mf16c -mavx2 -mfma
LDFLAGS+=-fopenmp
LDFLAGS+=-lcudart

ifneq (,$(wildcard /usr/local/cuda))
  LDFLAGS+=-L/usr/local/cuda/lib64
  CFLAGS+=-I/usr/local/cuda/include
endif
# NVHPC SDK lays out CUDA under $NVHPC_ROOT/cuda/{include,lib64} (and
# $NVHPC_ROOT/cuda/<ver>/...). Pick this up when /usr/local/cuda is absent —
# happens on clusters where CUDA ships via nvhpc rather than a
# stand-alone install.
ifneq (,$(NVHPC_ROOT))
  ifneq (,$(wildcard $(NVHPC_ROOT)/cuda/include/cuda_runtime_api.h))
    CFLAGS  += -I$(NVHPC_ROOT)/cuda/include
    LDFLAGS += -L$(NVHPC_ROOT)/cuda/lib64
  endif
endif

CUFLAGS+=-O2 -lineinfo -Ivendor
CUFLAGS+=-allow-unsupported-compiler # for recent CUDA versions

# MPI: extract include and link flags from mpicxx's --showme. We add them to
# CFLAGS/CUFLAGS/LDFLAGS rather than switching CXX to mpicxx so the existing
# benchmark.sbatch (which sets CXX=g++) keeps working.
comma := ,
MPI_INC_DIRS := $(shell mpicxx --showme:incdirs 2>/dev/null)
MPI_LIB_DIRS := $(shell mpicxx --showme:libdirs 2>/dev/null)
MPI_LIB_NAMES := $(shell mpicxx --showme:libs 2>/dev/null)
MPI_CFLAGS := $(addprefix -I,$(MPI_INC_DIRS))
MPI_LDFLAGS := $(addprefix -L,$(MPI_LIB_DIRS)) \
               $(foreach d,$(MPI_LIB_DIRS),-Wl$(comma)-rpath$(comma)$(d)) \
               $(addprefix -l,$(MPI_LIB_NAMES))

CFLAGS  += $(MPI_CFLAGS)
CUFLAGS += $(MPI_CFLAGS)
LDFLAGS += $(MPI_LDFLAGS)

ifeq ($(CUARCH),)
  CUFLAGS+=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 --threads 2
else
  CUFLAGS+=-arch=$(CUARCH)
endif

all: $(BINARY) asm

test: $(TEST_BINARY) test-asm

# Target to build just assembly files
asm: $(ASM_FILES)

test-asm: $(TEST_ASM_FILES)

format:
	clang-format -i src/*

$(BINARY): $(OBJECTS)
	$(CXX) $^ $(LDFLAGS) -o $@

$(TEST_BINARY): $(TEST_OBJECTS)
	$(CXX) $^ $(LDFLAGS) -o $@

# Rule to generate assembly for cpp files
$(ASM_DIR)/%.s: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -S -masm=intel -o $@

$(BUILD)/%.c.o: %.c
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.cc.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $< $(CUFLAGS) -c -MMD -MP -o $@

-include $(OBJECTS:.o=.d)
-include $(TEST_OBJECTS:.o=.d)

clean:
	rm -rf $(BUILD)

.PHONY: all clean format test asm test-asm