#pragma once

#include <cassert>

#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "kernel_impl.cuh"

namespace params_d {
__constant__ int32_t grid_numb[3];
__constant__ Dtype inv_grid_leng[3];
}

__host__ __device__ __forceinline__
int32_t gen_hash(const int32_t* idx,
                 const int32_t* grid_numb) {
  return idx[0] + (idx[1] + idx[2] * grid_numb[1]) * grid_numb[0];
}

template <typename Vec>
__global__ void gen_grid_id(const Vec* q,
                            int32_t* grid_id_of_ptcl,
                            const int32_t particle_number) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    int32_t idx[] = { static_cast<int32_t>(qi.x * params_d::inv_grid_leng[0]),
                      static_cast<int32_t>(qi.y * params_d::inv_grid_leng[1]),
                      static_cast<int32_t>(qi.z * params_d::inv_grid_leng[2]) };
    if (idx[0] == params_d::grid_numb[0]) idx[0]--;
    if (idx[1] == params_d::grid_numb[1]) idx[1]--;
    if (idx[2] == params_d::grid_numb[2]) idx[2]--;
    grid_id_of_ptcl[tid] = gen_hash(idx, params_d::grid_numb);
  }
}

template <typename Vec, typename Dtype>
class NeighListGPU {
  bool valid_ = false;
  int32_t grid_numb_[3], all_grid_ = -1;
  Dtype grid_leng_[3], inv_grid_leng_[3];
  Dtype search_length_ = 0.0, search_length2_ = 0.0;

  cuda_ptr<int32_t> grid_id_of_ptcl_, neigh_grid_id_, grid_pointer_, number_in_grid_;
  cuda_ptr<int32_t> ptcl_id_in_grid_;
  cuda_ptr<int32_t> neigh_list_, number_of_partners_;

  thrust::device_ptr<Vec> buffer_;
  thrust::device_ptr<int> uni_vect_, out_key_;

  enum : int32_t {
    MAX_PARTNERS = 400,
    SORT_FREQ = 50,
  };

  int32_t GenHash(const Vec& q) const {
    int32_t idx[] = {
      static_cast<int32_t>(q.x * inv_grid_leng_[0]),
      static_cast<int32_t>(q.y * inv_grid_leng_[1]),
      static_cast<int32_t>(q.z * inv_grid_leng_[2])
    };
    ApplyPBC(idx);
    return gen_hash(idx, grid_numb_);
  }

  void ApplyPBC(int32_t* idx) const {
    for (int i = 0; i < 3; i++) {
      if (idx[i] < 0) idx[i] += grid_numb_[i];
      if (idx[i] >= grid_numb_[i]) idx[i] -= grid_numb_[i];
    }
  }

  void Allocate(const int32_t particle_number) {
    grid_id_of_ptcl_.allocate(particle_number);
    neigh_grid_id_.allocate(27 * all_grid_);
    grid_pointer_.allocate(all_grid_ + 1);
    number_in_grid_.allocate(all_grid_);
    ptcl_id_in_grid_.allocate(particle_number);
    neigh_list_.allocate(MAX_PARTNERS * particle_number);
    number_of_partners_.allocate(particle_number);
    buffer_   = thrust::device_new<Vec>(particle_number);
    uni_vect_ = thrust::device_new<int>(particle_number);
    out_key_  = thrust::device_new<int>(particle_number);
  }

  void Deallocate() {
    thrust::device_delete(buffer_);
    thrust::device_delete(uni_vect_);
    thrust::device_delete(out_key_);
  }

  void MakeNeighGridId() {
    int32_t icell_id = 0;
    for (int32_t iz = 0; iz < grid_numb_[2]; iz++)
      for (int32_t iy = 0; iy < grid_numb_[1]; iy++)
        for (int32_t ix = 0; ix < grid_numb_[0]; ix++) {
          int32_t jcell_id = 0;
          for (int32_t jz = -1; jz < 2; jz++)
            for (int32_t jy = -1; jy < 2; jy++)
              for (int32_t jx = -1; jx < 2; jx++) {
                int32_t idx[] = { ix + jx, iy + jy, iz + jz };
                ApplyPBC(idx);
                neigh_grid_id_[27 * icell_id + jcell_id] = gen_hash(idx, grid_numb_);
                jcell_id++;
              }
          icell_id++;
        }
    neigh_grid_id_.host2dev();
  }

  template <typename T>
  void CopyGather(thrust::device_ptr<T>& __restrict src,
                  thrust::device_ptr<T>& __restrict buf,
                  const thrust::device_ptr<int32_t>& __restrict key,
                  const int size) {
    thrust::copy(src, src + size, buf);
    thrust::gather(key, key + size, buf, src);
  }

  void CountNumberInEachGrid(const int32_t particle_number) {
    const auto new_end = thrust::reduce_by_key(grid_id_of_ptcl_.thrust_ptr,
                                               grid_id_of_ptcl_.thrust_ptr + particle_number,
                                               uni_vect_,
                                               out_key_,
                                               number_in_grid_.thrust_ptr);
    const int val_elem = new_end.second - number_in_grid_.thrust_ptr;
#ifdef DEBUG
    if (val_elem != all_grid_) {
      std::cerr << "val_elem = " << val_elem << std::endl;
      std::cerr << "all_grid_ = " << all_grid_ << std::endl;
      std::exit(1);
    }
#endif
    const int nmax_in_cell = thrust::reduce(number_in_grid_.thrust_ptr,
                                            number_in_grid_.thrust_ptr + val_elem,
                                            0,
                                            thrust::maximum<int>());
    thrust::inclusive_scan(number_in_grid_.thrust_ptr,
                           number_in_grid_.thrust_ptr + all_grid_,
                           grid_pointer_.thrust_ptr + 1);
  }

  void SortPtclData(cuda_ptr<Vec>& q,
                    cuda_ptr<Vec>& p,
                    const int32_t particle_number) {
    thrust::sequence(ptcl_id_in_grid_.thrust_ptr,
                     ptcl_id_in_grid_.thrust_ptr + particle_number);
    thrust::sort_by_key(grid_id_of_ptcl_.thrust_ptr,
                        grid_id_of_ptcl_.thrust_ptr + particle_number,
                        ptcl_id_in_grid_.thrust_ptr);
    CopyGather(q.thrust_ptr,
               buffer_,
               ptcl_id_in_grid_.thrust_ptr,
               particle_number);
    CopyGather(p.thrust_ptr,
               buffer_,
               ptcl_id_in_grid_.thrust_ptr,
               particle_number);
    // thrust::sequence(ptcl_id_in_grid_.thrust_ptr,
    //                  ptcl_id_in_grid_.thrust_ptr + particle_number);
  }

  void CheckGridIdOfPtcl(const int32_t particle_number,
                         cuda_ptr<Vec>& q) {
    grid_id_of_ptcl_.dev2host();
    for (int i = 0; i < particle_number; i++) {
      const auto ref = GenHash(q[i]);
      if (grid_id_of_ptcl_[i] != ref) {
        std::cerr << "grid_id_of_ptcl_ is not correct.\n";
        std::exit(1);
      }
    }
    std::cerr << "grid_id_of_ptcl_ is correct.\n";
  }

  void CheckSorted(cuda_ptr<Vec>& q,
                   const int32_t particle_number) {
    q.dev2host();
    grid_id_of_ptcl_.dev2host();
    for (int32_t i = 0; i < particle_number; i++) {
      const auto hash = GenHash(q[i]);
      if (hash != grid_id_of_ptcl_[i]) {
        std::cerr << "particle data is not correctly sorted.\n";
        std::exit(1);
      }
    }
    std::cerr << "particle data is sorted.\n";
  }

  void CheckGridPointer(cuda_ptr<Vec>& q) {
    q.dev2host();
    grid_pointer_.dev2host();
    for (int32_t grid = 0; grid < all_grid_; grid++) {
      const auto beg = grid_pointer_[grid    ];
      const auto end = grid_pointer_[grid + 1];
      for (int32_t i = beg; i < end; i++) {
        const auto hash = GenHash(q[i]);
        if (hash != grid) {
          std::cerr << "grid_pointer is not correctly sorted.\n";
          std::exit(1);
        }
      }
    }
    std::cerr << "grid_pointer is correct.\n";
  }

public:
  NeighListGPU(const Dtype search_length,
               const Dtype Lx,
               const Dtype Ly,
               const Dtype Lz) {
    grid_numb_[0] = static_cast<int32_t>(Lx / search_length);
    grid_numb_[1] = static_cast<int32_t>(Ly / search_length);
    grid_numb_[2] = static_cast<int32_t>(Lz / search_length);
    all_grid_ = grid_numb_[0] * grid_numb_[1] * grid_numb_[2];

    grid_leng_[0] = Lx / grid_numb_[0];
    grid_leng_[1] = Ly / grid_numb_[1];
    grid_leng_[2] = Lz / grid_numb_[2];

    inv_grid_leng_[0] = 1.0 / grid_leng_[0];
    inv_grid_leng_[1] = 1.0 / grid_leng_[1];
    inv_grid_leng_[2] = 1.0 / grid_leng_[2];

    search_length_  = search_length;
    search_length2_ = search_length * search_length;
  }
  ~NeighListGPU() {
    Deallocate();
  }

  // disable copy
  const NeighListGPU<Vec, Dtype>& operator = (const NeighListGPU<Vec, Dtype>& obj) = delete;
  NeighListGPU<Vec, Dtype>(const NeighListGPU<Vec, Dtype>& obj) = delete;

  // disable move
  NeighListGPU<Vec, Dtype>& operator = (NeighListGPU<Vec, Dtype>&& obj) = delete;
  NeighListGPU<Vec, Dtype>(NeighListGPU<Vec, Dtype>&& obj) = delete;

  void Initialize(const int32_t particle_number) {
    Allocate(particle_number);

    neigh_list_.set_val(-1);
    number_of_partners_.set_val(0);
    grid_pointer_.set_val(0);
    thrust::fill(uni_vect_, uni_vect_ + particle_number, 1);

    checkCudaErrors(cudaMemcpyToSymbol(params_d::grid_numb,
                                       grid_numb_,
                                       3 * sizeof(int32_t)));
    checkCudaErrors(cudaMemcpyToSymbol(params_d::inv_grid_leng,
                                       inv_grid_leng_,
                                       3 * sizeof(Dtype)));

    MakeNeighGridId();
  }

  void MakeNeighList(cuda_ptr<Vec>& q,
                     cuda_ptr<Vec>& p,
                     const int32_t particle_number,
                     const int32_t tblock_size) {
    if (valid_) return;

    const int32_t grid_size = particle_number / tblock_size + 1;
    gen_grid_id<Vec><<<grid_size, tblock_size>>>(q, grid_id_of_ptcl_, particle_number);
    SortPtclData(q, p, particle_number);
    CountNumberInEachGrid(particle_number);

#ifdef DEBUG
    CheckGridIdOfPtcl(particle_number, q);
    CheckSorted(q, particle_number);
    CheckGridPointer(q);
#endif

    make_neighlist_naive<Vec><<<grid_size, tblock_size>>>(q,
                                                          grid_id_of_ptcl_,
                                                          neigh_grid_id_,
                                                          grid_pointer_,
                                                          neigh_list_,
                                                          number_of_partners_,
                                                          search_length2_,
                                                          particle_number);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  const cuda_ptr<int32_t>& neigh_list() const {
    return neigh_list_;
  }

  cuda_ptr<int32_t>& neigh_list() {
    return neigh_list_;
  }

  const cuda_ptr<int32_t>& number_of_partners() const {
    return number_of_partners_;
  }

  cuda_ptr<int32_t>& number_of_partners() {
    return number_of_partners_;
  }

  int32_t number_of_pairs() const {
    return thrust::reduce(number_of_partners_.thrust_ptr,
                          number_of_partners_.thrust_ptr + number_of_partners_.size);
  }
};
