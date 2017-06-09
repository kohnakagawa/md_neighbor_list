#pragma once

#include <cassert>

#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "device_util.cuh"
#include "kernel_impl.cuh"

constexpr int X = 0, Y = 1, Z = 2, DIM = 3;

namespace params_d {
  __constant__ int32_t mesh_size[DIM];
  __constant__ Dtype inv_ms[DIM];
}

__host__ __device__ __forceinline__
int32_t gen_hash(const int32_t* idx,
                 const int32_t* mesh_size) {
  return idx[0] + (idx[1] + idx[2] * mesh_size[1]) * mesh_size[0];
}

template <typename Vec>
__global__ void make_mesh(const Vec* q,
                          int32_t* particle_position,
                          const int32_t particle_number) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    int32_t idx[] = { static_cast<int32_t>(qi.x * params_d::inv_ms[0]),
                      static_cast<int32_t>(qi.y * params_d::inv_ms[1]),
                      static_cast<int32_t>(qi.z * params_d::inv_ms[2]) };
    if (idx[0] == params_d::mesh_size[0]) idx[0]--;
    if (idx[1] == params_d::mesh_size[1]) idx[1]--;
    if (idx[2] == params_d::mesh_size[2]) idx[2]--;
    particle_position[tid] = gen_hash(idx, params_d::mesh_size);
  }
}

template <typename Vec, typename Dtype>
class NeighListGPU {
  int32_t mesh_size_[3], number_of_mesh_ = -1;
  Dtype ms_[3], inv_ms_[3];
  Dtype search_length_ = 0.0, search_length2_ = 0.0;

  int32_t nmax_in_mesh_ = 0;
  int32_t tot_max_partner_number_ = 0;
  cuda_ptr<int32_t> particle_position_;

  cuda_ptr<int32_t> neigh_mesh_id_;

  cuda_ptr<int32_t> mesh_index_;
  cuda_ptr<int32_t> number_in_mesh_;

  cuda_ptr<int32_t> ptcl_id_in_mesh_;
  cuda_ptr<int32_t> transposed_list_, number_of_partners_;

  thrust::device_ptr<Vec> buffer_;
  thrust::device_ptr<int> uni_vect_, out_key_;

  thrust::device_ptr<int32_t> particle_position_buf_;
  thrust::device_ptr<int32_t> transposed_list_buf_;
  thrust::device_ptr<int32_t> num_of_ptcl_in_neigh_mesh_;
  thrust::device_ptr<int32_t> ptcl_id_of_neigh_mesh_;

  enum : int32_t {
    MAX_PARTNERS = 200, // for density = 1.0
    // MAX_PARTNERS = 100, // for density = 0.5
    SORT_FREQ = 50,
    WARP_SIZE = 32,
    NMAX_IN_MESH = 70,
  };

  int32_t GenHash(const Vec& q) const {
    int32_t idx[] = {
      static_cast<int32_t>(q.x * inv_ms_[0]),
      static_cast<int32_t>(q.y * inv_ms_[1]),
      static_cast<int32_t>(q.z * inv_ms_[2])
    };
    ApplyPBC(idx);
    return gen_hash(idx, mesh_size_);
  }

  void ApplyPBC(int32_t* idx) const {
    for (int i = 0; i < 3; i++) {
      if (idx[i] < 0) idx[i] += mesh_size_[i];
      if (idx[i] >= mesh_size_[i]) idx[i] -= mesh_size_[i];
    }
  }

  void Allocate(const int32_t particle_number) {
    tot_max_partner_number_ = MAX_PARTNERS * particle_number;

    particle_position_.allocate(particle_number);
    neigh_mesh_id_.allocate(27 * number_of_mesh_);
    mesh_index_.allocate(number_of_mesh_ + 1);
    number_in_mesh_.allocate(number_of_mesh_);
    ptcl_id_in_mesh_.allocate(particle_number);
    transposed_list_.allocate(MAX_PARTNERS * particle_number);
    number_of_partners_.allocate(particle_number);
    buffer_         = thrust::device_new<Vec>(particle_number);
    uni_vect_       = thrust::device_new<int>(particle_number);
    out_key_        = thrust::device_new<int>(particle_number);

    particle_position_buf_ = thrust::device_new<int32_t>(particle_number);
    transposed_list_buf_ = thrust::device_new<int32_t>(particle_number * MAX_PARTNERS);
    num_of_ptcl_in_neigh_mesh_ = thrust::device_new<int32_t>(number_of_mesh_);
    ptcl_id_of_neigh_mesh_ = thrust::device_new<int32_t>(number_of_mesh_ * 27 * NMAX_IN_MESH);
  }

  void Deallocate() {
    thrust::device_delete(buffer_);
    thrust::device_delete(uni_vect_);
    thrust::device_delete(out_key_);

    thrust::device_delete(particle_position_buf_);
    thrust::device_delete(transposed_list_buf_);
    thrust::device_delete(num_of_ptcl_in_neigh_mesh_);
    thrust::device_delete(ptcl_id_of_neigh_mesh_);
  }

  void MakeNeighMeshId() {
    int32_t imesh_id = 0;
    for (int32_t iz = 0; iz < mesh_size_[2]; iz++)
      for (int32_t iy = 0; iy < mesh_size_[1]; iy++)
        for (int32_t ix = 0; ix < mesh_size_[0]; ix++) {
          int32_t jmesh_id = 0;
          for (int32_t jz = -1; jz < 2; jz++)
            for (int32_t jy = -1; jy < 2; jy++)
              for (int32_t jx = -1; jx < 2; jx++) {
                int32_t idx[] = { ix + jx, iy + jy, iz + jz };
                ApplyPBC(idx);
                neigh_mesh_id_[27 * imesh_id + jmesh_id] = gen_hash(idx, mesh_size_);
                jmesh_id++;
              }
          imesh_id++;
        }
    neigh_mesh_id_.host2dev();
  }

  template <typename T>
  void CopyGather(thrust::device_ptr<T>& __restrict src,
                  thrust::device_ptr<T>& __restrict buf,
                  const thrust::device_ptr<int32_t>& __restrict key,
                  const int size) {
    thrust::copy(src, src + size, buf);
    thrust::gather(key, key + size, buf, src);
  }

  void CountNumberInEachMesh(const int32_t particle_number) {
    const auto new_end = thrust::reduce_by_key(particle_position_.thrust_ptr,
                                               particle_position_.thrust_ptr + particle_number,
                                               uni_vect_,
                                               out_key_,
                                               number_in_mesh_.thrust_ptr);
#ifdef DEBUG
    const int val_elem = new_end.second - number_in_mesh_.thrust_ptr;
    if (val_elem != number_of_mesh_) {
      std::cerr << "val_elem = " << val_elem << std::endl;
      std::cerr << "number_of_mesh_ = " << number_of_mesh_ << std::endl;
      std::exit(1);
    }
#else
    static_cast<void>(new_end);
#endif
    nmax_in_mesh_ = thrust::reduce(number_in_mesh_.thrust_ptr,
                                   number_in_mesh_.thrust_ptr + number_of_mesh_,
                                   0,
                                   thrust::maximum<int>());
    thrust::inclusive_scan(number_in_mesh_.thrust_ptr,
                           number_in_mesh_.thrust_ptr + number_of_mesh_,
                           mesh_index_.thrust_ptr + 1);

    thrust::copy_n(particle_position_buf_,
                   particle_number,
                   particle_position_.thrust_ptr);
  }

  void MakeMesh(cuda_ptr<Vec>& q,
                const int particle_number,
                const int tblock_size = 128) {
    const int32_t mesh_size = particle_number / tblock_size + 1;
    make_mesh<Vec><<<mesh_size, tblock_size>>>(q, particle_position_, particle_number);
    checkCudaErrors(cudaDeviceSynchronize());
  }

  void MakePtclIdInMesh(const int32_t particle_number) {
    thrust::sequence(ptcl_id_in_mesh_.thrust_ptr,
                     ptcl_id_in_mesh_.thrust_ptr + particle_number);
    thrust::copy_n(particle_position_.thrust_ptr,
                   particle_number,
                   particle_position_buf_);
    thrust::sort_by_key(particle_position_.thrust_ptr,
                        particle_position_.thrust_ptr + particle_number,
                        ptcl_id_in_mesh_.thrust_ptr);
  }

  void CheckParticlePosition(cuda_ptr<Vec>& q,
                             const int32_t particle_number) {
    q.dev2host();
    particle_position_.dev2host();

    for (int i = 0; i < particle_number; i++) {
      if (GenHash(q[i]) != particle_position_[i]) {
        std::cerr << "particle data is not correctly sorted.\n";
        std::exit(1);
      }
    }
    std::cerr << "particle data is sorted.\n";
  }

  void CheckMeshPointer(cuda_ptr<Vec>& q) {
    q.dev2host();
    particle_position_.dev2host();
    ptcl_id_in_mesh_.dev2host();
    mesh_index_.dev2host();

    for (int32_t mesh = 0; mesh < number_of_mesh_; mesh++) {
      const auto beg = mesh_index_[mesh    ];
      const auto end = mesh_index_[mesh + 1];
      for (int32_t i = beg; i < end; i++) {
        const auto hash = particle_position_[ptcl_id_in_mesh_[i]];
        if (hash != mesh) {
          std::cerr << "mesh_pointer is not correctly sorted.\n";
          std::exit(1);
        }
      }
    }
    std::cerr << "mesh_pointer is correct.\n";
  }

public:
  NeighListGPU(const Dtype search_length,
               const Dtype Lx,
               const Dtype Ly,
               const Dtype Lz) {
    mesh_size_[0] = static_cast<int32_t>(Lx / search_length);
    mesh_size_[1] = static_cast<int32_t>(Ly / search_length);
    mesh_size_[2] = static_cast<int32_t>(Lz / search_length);
    number_of_mesh_ = mesh_size_[0] * mesh_size_[1] * mesh_size_[2];

    ms_[0] = Lx / mesh_size_[0];
    ms_[1] = Ly / mesh_size_[1];
    ms_[2] = Lz / mesh_size_[2];

    inv_ms_[0] = 1.0 / ms_[0];
    inv_ms_[1] = 1.0 / ms_[1];
    inv_ms_[2] = 1.0 / ms_[2];

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

    transposed_list_.set_val(-1);
    thrust::fill(transposed_list_buf_,
                 transposed_list_buf_ + tot_max_partner_number_,
                 -1);
    number_of_partners_.set_val(0);
    mesh_index_.set_val(0);
    thrust::fill(uni_vect_, uni_vect_ + particle_number, 1);

    checkCudaErrors(cudaMemcpyToSymbol(params_d::mesh_size,
                                       mesh_size_,
                                       3 * sizeof(int32_t)));
    checkCudaErrors(cudaMemcpyToSymbol(params_d::inv_ms,
                                       inv_ms_,
                                       3 * sizeof(Dtype)));

    MakeNeighMeshId();
  }

  void MakeNeighList(cuda_ptr<Vec>& q,
                     const int32_t particle_number,
                     const bool sync = true,
                     int32_t tblock_size = 128,
                     const int32_t smem_hei = 7) {
    MakeMesh(q, particle_number, tblock_size);
    MakePtclIdInMesh(particle_number);
    CountNumberInEachMesh(particle_number);

#ifdef DEBUG
    CheckParticlePosition(q, particle_number);
    CheckMeshPointer(q);
#endif

    static bool is_first = true;
#ifdef REFERENCE
    static_cast<void>(smem_hei);
    if (is_first) {
      checkCudaErrors(cudaFuncSetCacheConfig(make_neighlist_naive<Vec, Dtype>,
                                             cudaFuncCachePreferL1));
      is_first = false;
    }
    const int grid_size = (particle_number - 1) / tblock_size + 1;
    make_neighlist_naive<Vec><<<grid_size, tblock_size>>>(q,
                                                          particle_position_,
                                                          neigh_mesh_id_,
                                                          mesh_index_,
                                                          ptcl_id_in_mesh_,
                                                          transposed_list_,
                                                          number_of_partners_,
                                                          search_length2_,
                                                          particle_number);
#elif USE_ROC
    static_cast<void>(smem_hei);
    if (is_first) {
      checkCudaErrors(cudaFuncSetCacheConfig(make_neighlist_roc<Vec, Dtype>,
                                             cudaFuncCachePreferL1));
      is_first = false;
    }
    const int grid_size = (particle_number - 1) / tblock_size + 1;
    make_neighlist_roc<Vec><<<grid_size, tblock_size>>>(q,
                                                        particle_position_,
                                                        neigh_mesh_id_,
                                                        mesh_index_,
                                                        ptcl_id_in_mesh_,
                                                        transposed_list_,
                                                        number_of_partners_,
                                                        search_length2_,
                                                        particle_number);
#elif USE_SMEM
    const int32_t num_smem_block = (tblock_size - 1) / SMEM_BLOCK_NUM + 1;
    const int32_t smem_size = smem_hei * num_smem_block * SMEM_BLOCK_NUM * sizeof(int32_t);
    if (is_first) {
      checkCudaErrors(cudaFuncSetCacheConfig(make_neighlist_smem<Vec, Dtype>,
                                             cudaFuncCachePreferShared));
      is_first = false;
    }
    const int grid_size = (particle_number - 1) / tblock_size + 1;
    make_neighlist_smem<Vec><<<grid_size, tblock_size, smem_size>>>(q,
                                                                    particle_position_,
                                                                    neigh_mesh_id_,
                                                                    mesh_index_,
                                                                    ptcl_id_in_mesh_,
                                                                    transposed_list_,
                                                                    number_of_partners_,
                                                                    smem_hei,
                                                                    search_length2_,
                                                                    particle_number);
#elif USE_SMEM_MESH
    tblock_size = ((nmax_in_mesh_ - 1) / WARP_SIZE + 1) * WARP_SIZE;
    const int32_t smem_size = nmax_in_mesh_ * sizeof(Vec);
    if (is_first) {
      checkCudaErrors(cudaFuncSetCacheConfig(make_neighlist_smem_mesh<Vec, Dtype>,
                                             cudaFuncCachePreferShared));
      is_first = false;
    }
    make_neighlist_smem_mesh<Vec><<<number_of_mesh_, tblock_size, smem_size>>>(q,
                                                                               neigh_mesh_id_,
                                                                               mesh_index_,
                                                                               ptcl_id_in_mesh_,
                                                                               transposed_list_,
                                                                               number_of_partners_,
                                                                               smem_hei,
                                                                               search_length2_,
                                                                               particle_number);
#elif USE_MATRIX_TRANSPOSE
    static_cast<void>(smem_hei);
    if (is_first) {
      checkCudaErrors(cudaFuncSetCacheConfig(make_neighlist_warp_unroll<Vec, Dtype>,
                                             cudaFuncCachePreferL1));
      is_first = false;
    }

    const auto grid_size = (particle_number - 1) / (tblock_size / 32) + 1;
    make_neighlist_warp_unroll<Vec><<<grid_size, tblock_size>>>(q,
                                                                particle_position_,
                                                                neigh_mesh_id_,
                                                                mesh_index_,
                                                                ptcl_id_in_mesh_,
                                                                thrust::raw_pointer_cast(transposed_list_buf_),
                                                                number_of_partners_,
                                                                search_length2_,
                                                                MAX_PARTNERS,
                                                                particle_number);
    transpose_neighlist(thrust::raw_pointer_cast(transposed_list_buf_),
                        transposed_list_,
                        particle_number,
                        MAX_PARTNERS);
#elif USE_MATRIX_TRANSPOSE_LOOP_FUSED
    static_cast<void>(smem_hei);
    if (is_first) {
      checkCudaErrors(cudaFuncSetCacheConfig(make_neighlist_warp_unroll_loop_fused<Vec, Dtype, 27 * NMAX_IN_MESH>,
                                             cudaFuncCachePreferL1));
      is_first = false;
    }
    auto grid_size = (number_of_mesh_ - 1) / (tblock_size / 32) + 1;
    make_ptcl_id_of_neigh_mesh<27 * NMAX_IN_MESH><<<grid_size, tblock_size>>>(particle_position_,
                                                                              neigh_mesh_id_,
                                                                              mesh_index_,
                                                                              ptcl_id_in_mesh_,
                                                                              thrust::raw_pointer_cast(num_of_ptcl_in_neigh_mesh_),
                                                                              thrust::raw_pointer_cast(ptcl_id_of_neigh_mesh_),
                                                                              number_of_mesh_);
    grid_size = (particle_number - 1) / (tblock_size / 32) + 1;
    make_neighlist_warp_unroll_loop_fused<Vec,
                                          Dtype,
                                          27 * NMAX_IN_MESH><<<grid_size, tblock_size>>>(q,
                                                                                         particle_position_,
                                                                                         thrust::raw_pointer_cast(num_of_ptcl_in_neigh_mesh_),
                                                                                         thrust::raw_pointer_cast(ptcl_id_of_neigh_mesh_),
                                                                                         thrust::raw_pointer_cast(transposed_list_buf_),
                                                                                         number_of_partners_,
                                                                                         search_length2_,
                                                                                         MAX_PARTNERS,
                                                                                         particle_number);
    transpose_neighlist(thrust::raw_pointer_cast(transposed_list_buf_),
                        transposed_list_,
                        particle_number,
                        MAX_PARTNERS);
#endif

    if (sync) checkCudaErrors(cudaDeviceSynchronize());
  }

  const cuda_ptr<int32_t>& neigh_list() const {
    return transposed_list_;
  }

  cuda_ptr<int32_t>& neigh_list() {
    return transposed_list_;
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
