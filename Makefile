PTX = make_list.ptx
SASS = make_list.sass
CUBIN = make_list.cubin
TARGET = make_list_gpu_ref.out make_list_gpu_roc.out make_list_gpu_smem.out make_list_gpu_smem_coars.out make_list_gpu_smem_cell.out make_list_cpu_ref.out make_list_cpu_loop_fused.out make_list_cpu_simd.out

WARNINGS = -Wall -Wextra -Wunused-variable -Wsign-compare
OPT_FLAGS = -O3
# OPT_FLAGS = -O0 -g -DDEBUG

cuda_profile = yes

# CUDA_HOME=/home/app/cuda/cuda-7.0
CUDA_HOME=/usr/local/cuda

NVCC=$(CUDA_HOME)/bin/nvcc
NVCCFLAGS= $(OPT_FLAGS) -std=c++11 -arch=sm_35 -Xcompiler "$(WARNINGS) $(OPT_FLAGS)" -ccbin=g++
INCLUDE = -I$(CUDA_HOME)/include -I$(CUDA_HOME)/samples/common/inc
ifeq ($(cuda_profile), yes)
NVCCFLAGS += -lineinfo -Xptxas -v
endif

# LIBRARY = -L$(BOOST_ROOT)/lib -lboost_system -lboost_program_options

ICC = icpc

all: $(TARGET)
sass: $(SASS)
ptx: $(PTX)

$(PTX): make_list.cu
$(SASS): $(CUBIN)
$(CUBIN): make_list.cu

.SUFFIXES:
.SUFFIXES: .cu .cubin
.cu.cubin:
	$(NVCC) $(NVCCFLAGS) -DUSE_SMEM $(INCLUDE) -cubin $< $(LIBRARY) -o $@

.SUFFIXES: .cubin .sass
.cubin.sass:
	$(CUDA_HOME)/bin/cuobjdump -sass $< | c++filt > $@

.SUFFIXES: .cu .ptx
.cu.ptx:
	$(NVCC) $(NVCCFLAGS) -DUSE_SMEM $(INCLUDE) -ptx $< $(LIBRARY) -o $@

make_list_gpu_ref.out: make_list.cu
	$(NVCC) $(NVCCFLAGS) -DREFERENCE $(INCLUDE) $< $(LIBRARY) -o $@

make_list_gpu_roc.out: make_list.cu
	$(NVCC) $(NVCCFLAGS) -DUSE_ROC $(INCLUDE) $< $(LIBRARY) -o $@

make_list_gpu_smem.out: make_list.cu
	$(NVCC) $(NVCCFLAGS) -DUSE_SMEM $(INCLUDE) $< $(LIBRARY) -o $@

make_list_gpu_smem_coars.out: make_list.cu
	$(NVCC) $(NVCCFLAGS) -DUSE_SMEM_COARS $(INCLUDE) $< $(LIBRARY) -o $@

make_list_gpu_smem_cell.out: make_list.cu
	$(NVCC) $(NVCCFLAGS) -DUSE_SMEM_CELL $(INCLUDE) $< $(LIBRARY) -o $@

make_list_cpu_ref.out: make_list.cpp
	$(ICC) $(WARNINGS) $(OPT_FLAGS) -DREFERENCE -xHOST -std=c++11 -ipo $< -o $@

make_list_cpu_loop_fused.out: make_list.cpp
	$(ICC) $(WARNINGS) $(OPT_FLAGS) -DLOOP_FUSION -xHOST -std=c++11 -ipo $< -o $@

make_list_cpu_simd.out: make_list.cpp
	$(ICC) $(WARNINGS) $(OPT_FLAGS) -DSIMD -DUSE_VEC4 -xHOST -std=c++11 -ipo $< -o $@

clean:
	rm -f $(TARGET) $(PTX) $(SASS) $(CUBIN) *~ *.core

test: make_list_gpu_ref.out make_list_gpu_roc.out make_list_gpu_smem.out make_list_cpu.out
	./make_list_gpu_ref.out
	./make_list_gpu_roc.out
	./make_list_gpu_smem.out
	./make_list_cpu.out
