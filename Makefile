PTX = make_list.ptx
SASS = make_list.sass
CUBIN = make_list.cubin
TARGET = make_list_gpu.out make_list_cpu.out

WARNINGS = -Wall -Wextra -Wunused-variable -Wsign-compare
OPT_FLAGS = -O3
# OPT_FLAGS = -O0 -g -DDEBUG

# cuda_profile = yes

CUDA_HOME=/home/app/cuda/cuda-7.0
# CUDA_HOME=/usr/local/cuda

BOOST_ROOT=/home/app/boost/1.58

NVCC=$(CUDA_HOME)/bin/nvcc
NVCCFLAGS= $(OPT_FLAGS) -std=c++11 -arch=sm_35 -Xcompiler "$(WARNINGS) $(OPT_FLAGS)" -ccbin=g++
INCLUDE = -I$(CUDA_HOME)/include -I$(CUDA_HOME)/samples/common/inc -I$(BOOST_ROOT)/include
ifeq ($(cuda_profile), yes)
NVCCFLAGS += -lineinfo -Xptxas -v
endif

LIBRARY = -L$(BOOST_ROOT)/lib -lboost_system -lboost_program_options

ICC = icpc

all: $(TARGET) $(PTX) $(SASS) $(CUBIN)

$(PTX): make_list.cu
$(SASS): $(CUBIN)
$(CUBIN): make_list.cu

.SUFFIXES:
.SUFFIXES: .cu .cubin
.cu.cubin:
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -cubin $< $(LIBRARY) -o $@

.SUFFIXES: .cubin .sass
.cubin.sass:
	$(CUDA_HOME)/bin/cuobjdump -sass $< | c++filt > $@

.SUFFIXES: .cu .ptx
.cu.ptx:
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -ptx $< $(LIBRARY) -o $@

make_list_gpu.out: make_list.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) $< $(LIBRARY) -o $@

make_list_cpu.out: make_list.cpp
	$(ICC) $(WARNINGS) $(OPT_FLAGS) -xHOST -std=c++11 -ipo $< -o $@

clean:
	rm -f $(TARGET) $(PTX) $(SASS) $(CUBIN) *~ *.core
