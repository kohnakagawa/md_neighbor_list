#pragma once

template <typename Vec, typename Dtype>
__global__ void make_neighlist_naive(const Vec* q,
                                     const int32_t* particle_position,
                                     const int32_t* neigh_mesh_id,
                                     const int32_t* mesh_index,
                                     const int32_t* ptcl_id_in_mesh,
                                     int32_t* transposed_list,
                                     int32_t* number_of_partners,
                                     const Dtype search_length2,
                                     const int32_t particle_number) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= particle_number) return;

  const auto qi = q[tid];
  const auto i_mesh_id = particle_position[tid];
  int32_t n_neigh = 0;
  for (int32_t cid = 0; cid < 27; cid++) {
    const auto j_mesh_id = neigh_mesh_id[27 * i_mesh_id + cid];
    const auto beg_id = mesh_index[j_mesh_id    ];
    const auto end_id = mesh_index[j_mesh_id + 1];
    for (int32_t k = beg_id; k < end_id; k++) {
      const auto j = ptcl_id_in_mesh[k];
      const auto drx = qi.x - q[j].x;
      const auto dry = qi.y - q[j].y;
      const auto drz = qi.z - q[j].z;
      const auto dr2 = drx * drx + dry * dry + drz * drz;
      if (dr2 > search_length2 || j == tid) continue;
      transposed_list[particle_number * n_neigh + tid] = j;
      n_neigh++;
    }
  }
  number_of_partners[tid] = n_neigh;
}

template <typename Vec, typename Dtype>
__global__ void make_neighlist_roc(const Vec* __restrict__ q,
                                   const int32_t* __restrict__ particle_position,
                                   const int32_t* __restrict__ neigh_mesh_id,
                                   const int32_t* __restrict__ mesh_index,
                                   const int32_t* __restrict__ ptcl_id_in_mesh,
                                   int32_t* __restrict__ transposed_list,
                                   int32_t* __restrict__ number_of_partners,
                                   const Dtype search_length2,
                                   const int32_t particle_number) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= particle_number) return;

  const auto qi = q[tid];
  const auto i_mesh_id = particle_position[tid];
  int32_t n_neigh = 0;
  for (int32_t cid = 0; cid < 27; cid++) {
    const auto j_mesh_id = neigh_mesh_id[27 * i_mesh_id + cid];
    const auto beg_id = mesh_index[j_mesh_id    ];
    const auto end_id = mesh_index[j_mesh_id + 1];
    for (int32_t k = beg_id; k < end_id; k++) {
      const auto j = ptcl_id_in_mesh[k];
      const auto drx = qi.x - q[j].x;
      const auto dry = qi.y - q[j].y;
      const auto drz = qi.z - q[j].z;
      const auto dr2 = drx * drx + dry * dry + drz * drz;
      if (dr2 > search_length2 || j == tid) continue;
      transposed_list[particle_number * n_neigh + tid] = j;
      n_neigh++;
    }
  }
  number_of_partners[tid] = n_neigh;
}

__device__ __forceinline__
void memcpy_to_gmem(const int32_t* list_buffer,
                    int32_t& n_neigh,
                    int32_t* transposed_list,
                    const int32_t num_out,
                    const int32_t tid,
                    const int32_t particle_number,
                    const int32_t loc_list_beg) {
  int32_t loc_list_idx = loc_list_beg;
  int32_t transposed_list_idx = n_neigh * particle_number + tid;
  for (int k = 0; k < num_out; k++) {
    transposed_list[transposed_list_idx] = list_buffer[loc_list_idx];
    loc_list_idx += SMEM_BLOCK_NUM;
    transposed_list_idx += particle_number;
  }
  n_neigh += num_out;
}

template <typename Vec, typename Dtype>
__global__ void make_neighlist_smem(const Vec* __restrict__ q,
                                    const int32_t* __restrict__ particle_position,
                                    const int32_t* __restrict__ neigh_mesh_id,
                                    const int32_t* __restrict__ mesh_index,
                                    const int32_t* __restrict__ ptcl_id_in_mesh,
                                    int32_t* __restrict__ transposed_list,
                                    int32_t* __restrict__ number_of_partners,
                                    const int32_t smem_loc_hei,
                                    const Dtype search_length2,
                                    const int32_t particle_number) {
  if (smem_loc_hei >= SMEM_MAX_HEI) {
    printf("smem_loc_hei is too large!\n");
    printf("smem_loc_hei = %d SMEM_MAX_HEI = %d", smem_loc_hei, SMEM_MAX_HEI);
    return;
  }

  extern __shared__ int32_t list_buffer[];
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= particle_number) return;

  const auto qi = q[tid];
  const auto i_mesh_id = particle_position[tid];
  const auto tile_offset = smem_loc_hei * SMEM_BLOCK_NUM;

  const int32_t loc_list_beg = smem_tile_beg(tile_offset) + lane_id();
  int32_t loc_list_idx = loc_list_beg;

  int32_t n_neigh = 0, n_loc_list = 0;
  for (int32_t cid = 0; cid < 27; cid++) {
    const auto j_mesh_id = neigh_mesh_id[27 * i_mesh_id + cid];
    const auto beg_id = mesh_index[j_mesh_id    ];
    const auto end_id = mesh_index[j_mesh_id + 1];
    for (int32_t k = beg_id; k < end_id; k++) {
      const auto j = ptcl_id_in_mesh[k];
      const auto drx = qi.x - q[j].x;
      const auto dry = qi.y - q[j].y;
      const auto drz = qi.z - q[j].z;
      const auto dr2 = drx * drx + dry * dry + drz * drz;
      if (dr2 < search_length2 && j != tid) {
        list_buffer[loc_list_idx] = j;
        n_loc_list++;
        loc_list_idx += SMEM_BLOCK_NUM;
      }

      const auto write_to_gmem = __any(n_loc_list == smem_loc_hei);
      if (write_to_gmem) {
        memcpy_to_gmem(list_buffer,
                       n_neigh,
                       transposed_list,
                       n_loc_list,
                       tid,
                       particle_number,
                       loc_list_beg);
        n_loc_list = 0;
        loc_list_idx = loc_list_beg;
      }
    }
  }

  memcpy_to_gmem(list_buffer,
                 n_neigh,
                 transposed_list,
                 n_loc_list,
                 tid,
                 particle_number,
                 loc_list_beg);
  number_of_partners[tid] = n_neigh;
}

template <typename Vec, typename Dtype>
__global__ void make_neighlist_smem_mesh(const Vec* __restrict__ q,
                                         const int32_t* __restrict__ neigh_mesh_id,
                                         const int32_t* __restrict__ mesh_index,
                                         const int32_t* __restrict__ ptcl_id_in_mesh,
                                         int32_t* __restrict__ transposed_list,
                                         int32_t* __restrict__ number_of_partners,
                                         const int32_t smem_loc_hei,
                                         const Dtype search_length2,
                                         const int32_t particle_number) {
  extern __shared__ Vec pos_buffer[];
  const auto tid_loc   = threadIdx.x;
  const auto i_mesh_id = blockIdx.x;
  const auto tid       = mesh_index[i_mesh_id] + tid_loc;
  const auto i_end_id  = mesh_index[i_mesh_id + 1];

  int i_ptcl_id = -1;
  Vec qi = {0.0};
  if (tid < i_end_id) {
    i_ptcl_id = ptcl_id_in_mesh[tid];
    qi = q[i_ptcl_id];
  }

  int32_t n_neigh = 0;
  for (int32_t cid = 0; cid < 27; cid++) {
    const auto j_mesh_id  = neigh_mesh_id[27 * i_mesh_id + cid];
    const auto j_beg_id   = mesh_index[j_mesh_id    ];
    const auto j_end_id   = mesh_index[j_mesh_id + 1];
    const auto num_loop_j = j_end_id - j_beg_id;

    // copy to smem
    __syncthreads();
    if (tid_loc < num_loop_j) {
      const auto j_ptcl_id = ptcl_id_in_mesh[j_beg_id + tid_loc];
      pos_buffer[tid_loc].x = q[j_ptcl_id].x;
      pos_buffer[tid_loc].y = q[j_ptcl_id].y;
      pos_buffer[tid_loc].z = q[j_ptcl_id].z;
    }
    __syncthreads();

    if (tid < i_end_id) {
      for (int32_t j = 0; j < num_loop_j; j++) {
        const auto j_ptcl_id = ptcl_id_in_mesh[j_beg_id + j];
        if (i_ptcl_id == j_ptcl_id) continue;
        const auto drx = qi.x - pos_buffer[j].x;
        const auto dry = qi.y - pos_buffer[j].y;
        const auto drz = qi.z - pos_buffer[j].z;
        const auto dr2 = drx * drx + dry * dry + drz * drz;
        if (dr2 > search_length2) continue;
        transposed_list[particle_number * n_neigh + i_ptcl_id] = j_ptcl_id;
        n_neigh++;
      }
    }
  }

  if (tid < i_end_id) number_of_partners[i_ptcl_id] = n_neigh;
}

// implement using cublas
void transpose_neighlist(const int32_t* __restrict__ transposed_list_buf,
                         int32_t* __restrict__ transposed_list,
                         const int32_t particle_number,
                         const int32_t max_partners) {
  static cublasHandle_t handle;
  static bool first_call = true;
  if (first_call) {
    cublasSafeCall(cublasCreate(&handle));
    first_call = false;
  }
  const float alpha = 1.0, beta = 0.0;
  cublasSafeCall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                             particle_number, max_partners,
                             &alpha,
                             reinterpret_cast<const float*>(transposed_list_buf),
                             max_partners,
                             &beta,
                             reinterpret_cast<const float*>(transposed_list_buf),
                             max_partners,
                             reinterpret_cast<float*>(transposed_list),
                             particle_number));
}

template <typename Vec, typename Dtype>
__global__ void make_neighlist_warp_unroll(const Vec* __restrict__ q,
                                           const int32_t* __restrict__ particle_position,
                                           const int32_t* __restrict__ neigh_mesh_id,
                                           const int32_t* __restrict__ mesh_index,
                                           const int32_t* __restrict__ ptcl_id_in_mesh,
                                           int32_t* __restrict__ transposed_list_buf,
                                           int32_t* __restrict__ number_of_partners,
                                           const Dtype search_length2,
                                           const int32_t max_partners,
                                           const int32_t particle_number) {
  const auto i_ptcl_id = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  if (i_ptcl_id >= particle_number) return;

  const auto qi        = q[i_ptcl_id];
  const auto i_mesh_id = particle_position[i_ptcl_id];
  const auto lid       = lane_id();
  const uint32_t mask   = (0xffffffff >> (31 - lid));

  int32_t n_neigh      = 0;
  for (int32_t cid = 0; cid < 27; cid++) {
    const auto j_mesh_id    = neigh_mesh_id[27 * i_mesh_id + cid];
    const auto beg_id       = mesh_index[j_mesh_id    ];
    const auto num_loop     = mesh_index[j_mesh_id + 1] - beg_id;

    for (int32_t j = lid; j < num_loop; j += warpSize) {
      const auto j_ptcl_id = ptcl_id_in_mesh[beg_id + j];
      const auto drx = qi.x - q[j_ptcl_id].x;
      const auto dry = qi.y - q[j_ptcl_id].y;
      const auto drz = qi.z - q[j_ptcl_id].z;
      const auto dr2 = drx * drx + dry * dry + drz * drz;
      const int32_t in_range = (dr2 <= search_length2) && (i_ptcl_id != j_ptcl_id);
      const uint32_t flag = __ballot(in_range);
      if (in_range) {
        const int32_t str_dst = __popc(flag & mask) + n_neigh - 1;
        transposed_list_buf[i_ptcl_id * max_partners + str_dst] = j_ptcl_id;
      }
      n_neigh += __popc(flag);
    }
    n_neigh = __shfl(n_neigh, 0);
  }

  if (lid == 0) number_of_partners[i_ptcl_id] = n_neigh;
}

template <int MAX_PTCL_NUM_IN_NMESH>
__global__ void make_ptcl_id_of_neigh_mesh(const int32_t* __restrict__ particle_position,
                                           const int32_t* __restrict__ neigh_mesh_id,
                                           const int32_t* __restrict__ mesh_index,
                                           const int32_t* __restrict__ ptcl_id_in_mesh,
                                           int32_t* __restrict__ num_of_ptcl_in_neigh_mesh,
                                           int32_t* __restrict__ ptcl_id_of_neigh_mesh,
                                           const int32_t tot_mesh_num) {
  const auto i_mesh_id = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  if (i_mesh_id >= tot_mesh_num) return;

  const auto lid = lane_id();
  int32_t* loc_id = &ptcl_id_of_neigh_mesh[i_mesh_id * MAX_PTCL_NUM_IN_NMESH];

  int32_t n_neigh = 0;
  for (int32_t cid = 0; cid < 27; cid++) {
    const auto j_mesh_id    = neigh_mesh_id[27 * i_mesh_id + cid];
    const auto beg_id       = mesh_index[j_mesh_id];
    const auto num_loop     = mesh_index[j_mesh_id + 1] - beg_id;
    const auto num_loop_ini = (num_loop / warpSize) * warpSize;

    int32_t j = 0;
    for (; j < num_loop_ini; j += warpSize) {
      loc_id[n_neigh + lid] = ptcl_id_in_mesh[beg_id + j + lid];
      n_neigh += warpSize;
    }

    const auto remaining_loop = num_loop % warpSize;
    if (lid < remaining_loop) {
      loc_id[n_neigh + lid] = ptcl_id_in_mesh[beg_id + j + lid];
    }

    n_neigh += remaining_loop;
  }

  if (lid == 0) num_of_ptcl_in_neigh_mesh[i_mesh_id] = n_neigh;
}

template <typename Vec, typename Dtype, int MAX_PTCL_NUM_IN_NMESH>
__global__ void make_neighlist_warp_unroll_loop_fused(const Vec* __restrict__ q,
                                                      const int32_t* __restrict__ particle_position,
                                                      const int32_t* __restrict__ num_of_ptcl_in_neigh_mesh,
                                                      const int32_t* __restrict__ ptcl_id_of_neigh_mesh,
                                                      int32_t* __restrict__ transposed_list_buf,
                                                      int32_t* __restrict__ number_of_partners,
                                                      const Dtype search_length2,
                                                      const int32_t max_partners,
                                                      const int32_t particle_number) {
  const auto i_ptcl_id = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  if (i_ptcl_id >= particle_number) return;

  const auto qi                    = q[i_ptcl_id];
  const auto i_mesh_id             = particle_position[i_ptcl_id];
  const auto lid                   = lane_id();
  int32_t n_neigh                  = 0;
  const int32_t* loc_id            = &ptcl_id_of_neigh_mesh[i_mesh_id * MAX_PTCL_NUM_IN_NMESH];
  int32_t* transposed_list_buf_loc = &transposed_list_buf[i_ptcl_id * max_partners];
  const auto num_loop              = num_of_ptcl_in_neigh_mesh[i_mesh_id];
  const uint32_t mask              = (0xffffffff >> (31 - lid));
  for (int32_t j = lid; j < num_loop; j += warpSize) {
    const auto j_ptcl_id = loc_id[j];
    const auto drx = qi.x - q[j_ptcl_id].x;
    const auto dry = qi.y - q[j_ptcl_id].y;
    const auto drz = qi.z - q[j_ptcl_id].z;
    const auto dr2 = drx * drx + dry * dry + drz * drz;
    const int32_t in_range = (dr2 <= search_length2) && (i_ptcl_id != j_ptcl_id);
    const uint32_t flag = __ballot(in_range);
    if (in_range) {
      const int32_t str_dst = __popc(flag & mask) + n_neigh - 1;
      transposed_list_buf_loc[str_dst] = j_ptcl_id;
    }
    n_neigh += __popc(flag);
  }

  if (lid == 0) number_of_partners[i_ptcl_id] = n_neigh;
}

struct Slot {
  int32_t i;
  double3 r;
  int32_t nn;
};

// Y.-H Tang, G.E. Karniadakis / CPC 185(2014) 2809--2822
// # of thread == warpSize * number of mesh
template <typename Vec, typename Dtype, int MAX_PTCL_NUM_IN_NMESH>
__global__ void make_neighlist_warp_unroll_smem(const Vec*     __restrict__ q,
                                                const int32_t* __restrict__ particle_position,
                                                const int32_t* __restrict__ num_of_ptcl_in_neigh_mesh,
                                                const int32_t* __restrict__ ptcl_id_of_neigh_mesh,
                                                const int32_t* __restrict__ mesh_index,
                                                const int32_t* __restrict__ ptcl_id_in_mesh,
                                                int32_t*       __restrict__ transposed_list_buf,
                                                int32_t*       __restrict__ number_of_partners,
                                                const Dtype search_length2,
                                                const int32_t max_partners,
                                                const int32_t particle_number,
                                                const int32_t number_of_mesh) {
  const auto i_mesh_id = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;

  if (i_mesh_id >= number_of_mesh) return;

  extern __shared__ Slot buffer[];
  Slot* slots = &buffer[warpSize * warp_id()];

  const auto lid        = lane_id();
  const auto beg_imesh  = mesh_index[i_mesh_id    ];
  const auto end_imesh  = mesh_index[i_mesh_id + 1];
  const auto nStencil   = num_of_ptcl_in_neigh_mesh[i_mesh_id];
  const int32_t* loc_id = &ptcl_id_of_neigh_mesh[i_mesh_id * MAX_PTCL_NUM_IN_NMESH];

  int32_t i_ptcl_offset = beg_imesh;
  while (i_ptcl_offset < end_imesh) {
    const auto part = ((end_imesh - i_ptcl_offset) > warpSize) ? warpSize : (end_imesh - i_ptcl_offset);
    const auto i_ptcl_id = lid + i_ptcl_offset;
    if (lid < part) {
      slots[lid].i = ptcl_id_in_mesh[i_ptcl_id];
      slots[lid].r.x = q[slots[lid].i].x;
      slots[lid].r.y = q[slots[lid].i].y;
      slots[lid].r.z = q[slots[lid].i].z;
      slots[lid].nn  = 0;
    }

    for (int32_t k = lid; k < nStencil; k += warpSize) {
      const auto j_ptcl_id = loc_id[k];
      const auto qj = q[j_ptcl_id];
      for (int32_t i = 0; i < part; i++) {
        const auto drx       = slots[i].r.x - qj.x;
        const auto dry       = slots[i].r.y - qj.y;
        const auto drz       = slots[i].r.z - qj.z;
        const auto dr2       = drx * drx + dry * dry + drz * drz;
        const int32_t hit    = (dr2 <= search_length2) && (slots[i].i != j_ptcl_id);
        const uint32_t nhit  = __ballot(hit);
        const int32_t nahead = nhit << (warpSize - lid);
        const int32_t pins   = slots[i].nn + __popc(nahead);
        if (hit) {
          transposed_list_buf[slots[i].i * max_partners + pins] = j_ptcl_id;
        }
        if (lid == 0) {
          slots[i].nn += __popc(nhit);
        }
      }
    }

    if (lid < part) {
      number_of_partners[slots[lid].i] = slots[lid].nn;
    }

    i_ptcl_offset += warpSize;
  }
}
