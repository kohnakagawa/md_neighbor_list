#pragma once

#include <cassert>

#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "device_util.cuh"
#include "kernel_impl.cuh"

namespace params_d {
__constant__ int32_t cell_numb[3];
__constant__ Dtype inv_cell_leng[3];
}

__host__ __device__ __forceinline__
int32_t gen_hash(const int32_t* idx,
                 const int32_t* cell_numb) {
  return idx[0] + (idx[1] + idx[2] * cell_numb[1]) * cell_numb[0];
}

template <typename Vec>
__global__ void gen_cell_id(const Vec* q,
                            int32_t* cell_id_of_ptcl,
                            const int32_t particle_number) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    int32_t idx[] = { static_cast<int32_t>(qi.x * params_d::inv_cell_leng[0]),
                      static_cast<int32_t>(qi.y * params_d::inv_cell_leng[1]),
                      static_cast<int32_t>(qi.z * params_d::inv_cell_leng[2]) };
    if (idx[0] == params_d::cell_numb[0]) idx[0]--;
    if (idx[1] == params_d::cell_numb[1]) idx[1]--;
    if (idx[2] == params_d::cell_numb[2]) idx[2]--;
    cell_id_of_ptcl[tid] = gen_hash(idx, params_d::cell_numb);
  }
}

template <typename Vec, typename Dtype>
class NeighListGPU {
  bool valid_ = false;
  int32_t cell_numb_[3], all_cell_ = -1;
  Dtype cell_leng_[3], inv_cell_leng_[3];
  Dtype search_length_ = 0.0, search_length2_ = 0.0;

  int32_t nmax_in_cell_ = 0;
  int32_t tot_max_partner_number_ = 0;
  cuda_ptr<int32_t> cell_id_of_ptcl_, neigh_cell_id_, cell_pointer_, number_in_cell_;
  cuda_ptr<int32_t> ptcl_id_in_cell_;
  cuda_ptr<int32_t> neigh_list_, number_of_partners_;

  thrust::device_ptr<Vec> buffer_;
  thrust::device_ptr<int> uni_vect_, out_key_;

  thrust::device_ptr<int32_t> neigh_list_buf_;
  thrust::device_ptr<int32_t> num_of_ptcl_in_neigh_cell_;
  thrust::device_ptr<int32_t> ptcl_id_of_neigh_cell_;

  enum : int32_t {
    MAX_PARTNERS = 400,
    SORT_FREQ = 50,
    WARP_SIZE = 32,
    NMAX_IN_CELL = 70,
  };

  int32_t GenHash(const Vec& q) const {
    int32_t idx[] = {
      static_cast<int32_t>(q.x * inv_cell_leng_[0]),
      static_cast<int32_t>(q.y * inv_cell_leng_[1]),
      static_cast<int32_t>(q.z * inv_cell_leng_[2])
    };
    ApplyPBC(idx);
    return gen_hash(idx, cell_numb_);
  }

  void ApplyPBC(int32_t* idx) const {
    for (int i = 0; i < 3; i++) {
      if (idx[i] < 0) idx[i] += cell_numb_[i];
      if (idx[i] >= cell_numb_[i]) idx[i] -= cell_numb_[i];
    }
  }

  void Allocate(const int32_t particle_number) {
    tot_max_partner_number_ = MAX_PARTNERS * particle_number;

    cell_id_of_ptcl_.allocate(particle_number);
    neigh_cell_id_.allocate(27 * all_cell_);
    cell_pointer_.allocate(all_cell_ + 1);
    number_in_cell_.allocate(all_cell_);
    ptcl_id_in_cell_.allocate(particle_number);
    neigh_list_.allocate(MAX_PARTNERS * particle_number);
    number_of_partners_.allocate(particle_number);
    buffer_         = thrust::device_new<Vec>(particle_number);
    uni_vect_       = thrust::device_new<int>(particle_number);
    out_key_        = thrust::device_new<int>(particle_number);

    neigh_list_buf_ = thrust::device_new<int32_t>(particle_number * MAX_PARTNERS);
    num_of_ptcl_in_neigh_cell_ = thrust::device_new<int32_t>(all_cell_);
    ptcl_id_of_neigh_cell_ = thrust::device_new<int32_t>(all_cell_ * 27 * NMAX_IN_CELL);
  }

  void Deallocate() {
    thrust::device_delete(buffer_);
    thrust::device_delete(uni_vect_);
    thrust::device_delete(out_key_);

    thrust::device_delete(neigh_list_buf_);
    thrust::device_delete(num_of_ptcl_in_neigh_cell_);
    thrust::device_delete(ptcl_id_of_neigh_cell_);
  }

  void MakeNeighCellId() {
    int32_t icell_id = 0;
    for (int32_t iz = 0; iz < cell_numb_[2]; iz++)
      for (int32_t iy = 0; iy < cell_numb_[1]; iy++)
        for (int32_t ix = 0; ix < cell_numb_[0]; ix++) {
          int32_t jcell_id = 0;
          for (int32_t jz = -1; jz < 2; jz++)
            for (int32_t jy = -1; jy < 2; jy++)
              for (int32_t jx = -1; jx < 2; jx++) {
                int32_t idx[] = { ix + jx, iy + jy, iz + jz };
                ApplyPBC(idx);
                neigh_cell_id_[27 * icell_id + jcell_id] = gen_hash(idx, cell_numb_);
                jcell_id++;
              }
          icell_id++;
        }
    neigh_cell_id_.host2dev();
  }

  template <typename T>
  void CopyGather(thrust::device_ptr<T>& __restrict src,
                  thrust::device_ptr<T>& __restrict buf,
                  const thrust::device_ptr<int32_t>& __restrict key,
                  const int size) {
    thrust::copy(src, src + size, buf);
    thrust::gather(key, key + size, buf, src);
  }

  void CountNumberInEachCell(const int32_t particle_number) {
    const auto new_end = thrust::reduce_by_key(cell_id_of_ptcl_.thrust_ptr,
                                               cell_id_of_ptcl_.thrust_ptr + particle_number,
                                               uni_vect_,
                                               out_key_,
                                               number_in_cell_.thrust_ptr);
#ifdef DEBUG
    const int val_elem = new_end.second - number_in_cell_.thrust_ptr;
    if (val_elem != all_cell_) {
      std::cerr << "val_elem = " << val_elem << std::endl;
      std::cerr << "all_cell_ = " << all_cell_ << std::endl;
      std::exit(1);
    }
#endif
    nmax_in_cell_ = thrust::reduce(number_in_cell_.thrust_ptr,
                                   number_in_cell_.thrust_ptr + all_cell_,
                                   0,
                                   thrust::maximum<int>());
    thrust::inclusive_scan(number_in_cell_.thrust_ptr,
                           number_in_cell_.thrust_ptr + all_cell_,
                           cell_pointer_.thrust_ptr + 1);
  }

  void SortPtclData(cuda_ptr<Vec>& q,
                    cuda_ptr<Vec>& p,
                    const int32_t particle_number) {
    thrust::sequence(ptcl_id_in_cell_.thrust_ptr,
                     ptcl_id_in_cell_.thrust_ptr + particle_number);
    thrust::sort_by_key(cell_id_of_ptcl_.thrust_ptr,
                        cell_id_of_ptcl_.thrust_ptr + particle_number,
                        ptcl_id_in_cell_.thrust_ptr);
    CopyGather(q.thrust_ptr,
               buffer_,
               ptcl_id_in_cell_.thrust_ptr,
               particle_number);
    CopyGather(p.thrust_ptr,
               buffer_,
               ptcl_id_in_cell_.thrust_ptr,
               particle_number);
    // thrust::sequence(ptcl_id_in_cell_.thrust_ptr,
    //                  ptcl_id_in_cell_.thrust_ptr + particle_number);
  }

  void CheckCellIdOfPtcl(const int32_t particle_number,
                         cuda_ptr<Vec>& q) {
    cell_id_of_ptcl_.dev2host();
    for (int i = 0; i < particle_number; i++) {
      const auto ref = GenHash(q[i]);
      if (cell_id_of_ptcl_[i] != ref) {
        std::cerr << "cell_id_of_ptcl_ is not correct.\n";
        std::exit(1);
      }
    }
    std::cerr << "cell_id_of_ptcl_ is correct.\n";
  }

  void CheckSorted(cuda_ptr<Vec>& q,
                   const int32_t particle_number) {
    q.dev2host();
    cell_id_of_ptcl_.dev2host();
    for (int32_t i = 0; i < particle_number; i++) {
      const auto hash = GenHash(q[i]);
      if (hash != cell_id_of_ptcl_[i]) {
        std::cerr << "particle data is not correctly sorted.\n";
        std::exit(1);
      }
    }
    std::cerr << "particle data is sorted.\n";
  }

  void CheckCellPointer(cuda_ptr<Vec>& q) {
    q.dev2host();
    cell_pointer_.dev2host();
    for (int32_t cell = 0; cell < all_cell_; cell++) {
      const auto beg = cell_pointer_[cell    ];
      const auto end = cell_pointer_[cell + 1];
      for (int32_t i = beg; i < end; i++) {
        const auto hash = GenHash(q[i]);
        if (hash != cell) {
          std::cerr << "cell_pointer is not correctly sorted.\n";
          std::exit(1);
        }
      }
    }
    std::cerr << "cell_pointer is correct.\n";
  }

public:
  NeighListGPU(const Dtype search_length,
               const Dtype Lx,
               const Dtype Ly,
               const Dtype Lz) {
    cell_numb_[0] = static_cast<int32_t>(Lx / search_length);
    cell_numb_[1] = static_cast<int32_t>(Ly / search_length);
    cell_numb_[2] = static_cast<int32_t>(Lz / search_length);
    all_cell_ = cell_numb_[0] * cell_numb_[1] * cell_numb_[2];

    cell_leng_[0] = Lx / cell_numb_[0];
    cell_leng_[1] = Ly / cell_numb_[1];
    cell_leng_[2] = Lz / cell_numb_[2];

    inv_cell_leng_[0] = 1.0 / cell_leng_[0];
    inv_cell_leng_[1] = 1.0 / cell_leng_[1];
    inv_cell_leng_[2] = 1.0 / cell_leng_[2];

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
    thrust::fill(neigh_list_buf_,
                 neigh_list_buf_ + tot_max_partner_number_,
                 -1);
    number_of_partners_.set_val(0);
    cell_pointer_.set_val(0);
    thrust::fill(uni_vect_, uni_vect_ + particle_number, 1);

    checkCudaErrors(cudaMemcpyToSymbol(params_d::cell_numb,
                                       cell_numb_,
                                       3 * sizeof(int32_t)));
    checkCudaErrors(cudaMemcpyToSymbol(params_d::inv_cell_leng,
                                       inv_cell_leng_,
                                       3 * sizeof(Dtype)));

    MakeNeighCellId();
  }

  void MakeNeighList(cuda_ptr<Vec>& q,
                     cuda_ptr<Vec>& p,
                     const int32_t particle_number,
                     const bool sync = true,
                     int32_t tblock_size = 128,
                     const int32_t smem_hei = 7) {
    if (valid_) return;

    const int32_t cell_size = particle_number / tblock_size + 1;
    gen_cell_id<Vec><<<cell_size, tblock_size>>>(q, cell_id_of_ptcl_, particle_number);
    SortPtclData(q, p, particle_number);
    CountNumberInEachCell(particle_number);

#ifdef DEBUG
    CheckCellIdOfPtcl(particle_number, q);
    CheckSorted(q, particle_number);
    CheckCellPointer(q);
#endif

    static bool is_first = true;
#ifdef REFERENCE
    if (is_first) {
      checkCudaErrors(cudaFuncSetCacheConfig(make_neighlist_naive<Vec, Dtype>,
                                             cudaFuncCachePreferL1));
      is_first = false;
    }
    make_neighlist_naive<Vec><<<cell_size, tblock_size>>>(q,
                                                          cell_id_of_ptcl_,
                                                          neigh_cell_id_,
                                                          cell_pointer_,
                                                          neigh_list_,
                                                          number_of_partners_,
                                                          search_length2_,
                                                          particle_number);
#elif defined USE_ROC
    if (is_first) {
      checkCudaErrors(cudaFuncSetCacheConfig(make_neighlist_roc<Vec, Dtype>,
                                             cudaFuncCachePreferL1));
      is_first = false;
    }
    make_neighlist_roc<Vec><<<cell_size, tblock_size>>>(q,
                                                        cell_id_of_ptcl_,
                                                        neigh_cell_id_,
                                                        cell_pointer_,
                                                        neigh_list_,
                                                        number_of_partners_,
                                                        search_length2_,
                                                        particle_number);
#elif defined USE_SMEM
    const int32_t num_smem_block = (tblock_size - 1) / SMEM_BLOCK_NUM + 1;
    const int32_t smem_size = smem_hei * num_smem_block * SMEM_BLOCK_NUM * sizeof(int32_t);
    if (is_first) {
      checkCudaErrors(cudaFuncSetCacheConfig(make_neighlist_smem<Vec, Dtype>,
                                             cudaFuncCachePreferShared));
      is_first = false;
    }
    make_neighlist_smem<Vec><<<cell_size, tblock_size, smem_size>>>(q,
                                                                    cell_id_of_ptcl_,
                                                                    neigh_cell_id_,
                                                                    cell_pointer_,
                                                                    neigh_list_,
                                                                    number_of_partners_,
                                                                    smem_hei,
                                                                    search_length2_,
                                                                    particle_number);
#elif defined USE_SMEM_COARS
    const int32_t num_smem_block = (tblock_size - 1) / SMEM_BLOCK_NUM + 1;
    const int32_t smem_size = smem_hei * num_smem_block * SMEM_BLOCK_NUM * sizeof(int32_t);
    if (is_first) {
      checkCudaErrors(cudaFuncSetCacheConfig(make_neighlist_smem_coars<Vec, Dtype>,
                                             cudaFuncCachePreferShared));
      is_first = false;
    }
    make_neighlist_smem_coars<Vec><<<cell_size, tblock_size, smem_size>>>(q,
                                                                          cell_id_of_ptcl_,
                                                                          neigh_cell_id_,
                                                                          cell_pointer_,
                                                                          neigh_list_,
                                                                          number_of_partners_,
                                                                          smem_hei,
                                                                          search_length2_,
                                                                          particle_number);
#elif defined USE_SMEM_CELL
    tblock_size = ((nmax_in_cell_ - 1) / WARP_SIZE + 1) * WARP_SIZE;
    const int32_t num_smem_block = (tblock_size - 1) / SMEM_BLOCK_NUM + 1;
    const int32_t smem_size = smem_hei * num_smem_block * SMEM_BLOCK_NUM * sizeof(int32_t);
    if (is_first) {
      checkCudaErrors(cudaFuncSetCacheConfig(make_neighlist_smem_cell<Vec, Dtype>,
                                             cudaFuncCachePreferShared));
      is_first = false;
    }
    make_neighlist_smem_cell<Vec><<<all_cell_, tblock_size, smem_size>>>(q,
                                                                         neigh_cell_id_,
                                                                         cell_pointer_,
                                                                         neigh_list_,
                                                                         number_of_partners_,
                                                                         smem_hei,
                                                                         search_length2_,
                                                                         particle_number);
#elif defined USE_SMEM_ONCE
    const int32_t num_smem_block = (tblock_size - 1) / SMEM_BLOCK_NUM + 1;
    const int32_t smem_size = smem_hei * num_smem_block * SMEM_BLOCK_NUM * sizeof(int32_t);
    if (is_first) {
      checkCudaErrors(cudaFuncSetCacheConfig(make_neighlist_smem_once<Vec, Dtype>,
                                             cudaFuncCachePreferShared));
      is_first = false;
    }
    make_neighlist_smem_once<Vec><<<cell_size, tblock_size, smem_size>>>(q,
                                                                         cell_id_of_ptcl_,
                                                                         neigh_cell_id_,
                                                                         cell_pointer_,
                                                                         neigh_list_,
                                                                         number_of_partners_,
                                                                         smem_hei,
                                                                         search_length2_,
                                                                         particle_number);
#elif defined USE_MATRIX_TRANSPOSE
    if (is_first) {
      checkCudaErrors(cudaFuncSetCacheConfig(make_neighlist_warp_unroll<Vec, Dtype>,
                                             cudaFuncCachePreferL1));
      is_first = false;
    }

    const auto grid_size = (particle_number - 1) / (tblock_size / 32) + 1;
    make_neighlist_warp_unroll<Vec><<<grid_size, tblock_size>>>(q,
                                                                cell_id_of_ptcl_,
                                                                neigh_cell_id_,
                                                                cell_pointer_,
                                                                thrust::raw_pointer_cast(neigh_list_buf_),
                                                                number_of_partners_,
                                                                search_length2_,
                                                                MAX_PARTNERS,
                                                                particle_number);
    transpose_neighlist(thrust::raw_pointer_cast(neigh_list_buf_),
                        neigh_list_,
                        particle_number,
                        MAX_PARTNERS);
#elif defined USE_MATRIX_TRANSPOSE_LOOP_FUSED
    if (is_first) {
      checkCudaErrors(cudaFuncSetCacheConfig(make_neighlist_warp_unroll_loop_fused<Vec, Dtype, 27 * NMAX_IN_CELL>,
                                             cudaFuncCachePreferL1));
      is_first = false;
    }
    auto grid_size = (all_cell_ - 1) / (tblock_size / 32) + 1;
    make_ptcl_id_of_neigh_cell<27 * NMAX_IN_CELL><<<grid_size, tblock_size>>>(cell_id_of_ptcl_,
                                                                              neigh_cell_id_,
                                                                              cell_pointer_,
                                                                              thrust::raw_pointer_cast(num_of_ptcl_in_neigh_cell_),
                                                                              thrust::raw_pointer_cast(ptcl_id_of_neigh_cell_),
                                                                              all_cell_);
    grid_size = (particle_number - 1) / (tblock_size / 32) + 1;
    make_neighlist_warp_unroll_loop_fused<Vec,
                                          Dtype,
                                          27 * NMAX_IN_CELL><<<grid_size, tblock_size>>>(q,
                                                                                         cell_id_of_ptcl_,
                                                                                         thrust::raw_pointer_cast(num_of_ptcl_in_neigh_cell_),
                                                                                         thrust::raw_pointer_cast(ptcl_id_of_neigh_cell_),
                                                                                         thrust::raw_pointer_cast(neigh_list_buf_),
                                                                                         number_of_partners_,
                                                                                         search_length2_,
                                                                                         MAX_PARTNERS,
                                                                                         particle_number);
    transpose_neighlist(thrust::raw_pointer_cast(neigh_list_buf_),
                        neigh_list_,
                        particle_number,
                        MAX_PARTNERS);
#elif defined USE_MATRIX_TRANSPOSE_LOOP_FUSED_REV
    if (is_first) {
      checkCudaErrors(cudaFuncSetCacheConfig(make_neighlist_warp_unroll_loop_fused<Vec, Dtype, 27 * NMAX_IN_CELL>,
                                             cudaFuncCachePreferL1));
      is_first = false;
    }
    auto grid_size = (all_cell_ - 1) / (tblock_size / 32) + 1;
    make_ptcl_id_of_neigh_cell<27 * NMAX_IN_CELL><<<grid_size, tblock_size>>>(cell_id_of_ptcl_,
                                                                              neigh_cell_id_,
                                                                              cell_pointer_,
                                                                              thrust::raw_pointer_cast(num_of_ptcl_in_neigh_cell_),
                                                                              thrust::raw_pointer_cast(ptcl_id_of_neigh_cell_),
                                                                              all_cell_);
    grid_size = (all_cell_ - 1) / (tblock_size / 32) + 1;
    make_neighlist_warp_unroll_loop_fused_rev<Vec,
                                              Dtype,
                                              27 * NMAX_IN_CELL><<<grid_size, tblock_size>>>(q,
                                                                                             thrust::raw_pointer_cast(num_of_ptcl_in_neigh_cell_),
                                                                                             thrust::raw_pointer_cast(ptcl_id_of_neigh_cell_),
                                                                                             cell_pointer_,
                                                                                             thrust::raw_pointer_cast(neigh_list_buf_),
                                                                                             number_of_partners_,
                                                                                             search_length2_,
                                                                                             all_cell_,
                                                                                             MAX_PARTNERS,
                                                                                             particle_number);
    transpose_neighlist(thrust::raw_pointer_cast(neigh_list_buf_),
                        neigh_list_,
                        particle_number,
                        MAX_PARTNERS);
#endif

    if (sync) checkCudaErrors(cudaDeviceSynchronize());
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
