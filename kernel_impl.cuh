#pragma once

template <typename Vec, typename Dtype>
__global__ void make_neighlist_naive(const Vec* q,
                                     const int32_t* cell_id_of_ptcl,
                                     const int32_t* neigh_cell_id,
                                     const int32_t* cell_pointer,
                                     int32_t* neigh_list,
                                     int32_t* number_of_partners,
                                     const Dtype search_length2,
                                     const int32_t particle_number) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto i_cell_id = cell_id_of_ptcl[tid];
    int32_t n_neigh = 0;
    for (int32_t cid = 0; cid < 27; cid++) {
      const auto j_cell_id = neigh_cell_id[27 * i_cell_id + cid];
      const auto beg_id = cell_pointer[j_cell_id    ];
      const auto end_id = cell_pointer[j_cell_id + 1];
      for (int32_t j = beg_id; j < end_id; j++) {
        const auto drx = qi.x - q[j].x;
        const auto dry = qi.y - q[j].y;
        const auto drz = qi.z - q[j].z;
        const auto dr2 = drx * drx + dry * dry + drz * drz;
        if (dr2 > search_length2 || j == tid) continue;
        neigh_list[particle_number * n_neigh + tid] = j;
        n_neigh++;
      }
    }
    number_of_partners[tid] = n_neigh;
  }
}

template <typename Vec, typename Dtype>
__global__ void make_neighlist_roc(const Vec* __restrict__ q,
                                   const int32_t* __restrict__ cell_id_of_ptcl,
                                   const int32_t* __restrict__ neigh_cell_id,
                                   const int32_t* __restrict__ cell_pointer,
                                   int32_t* __restrict__ neigh_list,
                                   int32_t* __restrict__ number_of_partners,
                                   const Dtype search_length2,
                                   const int32_t particle_number) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto i_cell_id = cell_id_of_ptcl[tid];
    int32_t n_neigh = 0;
    for (int32_t cid = 0; cid < 27; cid++) {
      const auto j_cell_id = neigh_cell_id[27 * i_cell_id + cid];
      const auto beg_id = cell_pointer[j_cell_id    ];
      const auto end_id = cell_pointer[j_cell_id + 1];
      for (int32_t j = beg_id; j < end_id; j++) {
        const auto drx = qi.x - q[j].x;
        const auto dry = qi.y - q[j].y;
        const auto drz = qi.z - q[j].z;
        const auto dr2 = drx * drx + dry * dry + drz * drz;
        if (dr2 > search_length2 || j == tid) continue;
        neigh_list[particle_number * n_neigh + tid] = j;
        n_neigh++;
      }
    }
    number_of_partners[tid] = n_neigh;
  }
}

__device__ __forceinline__
void memcpy_to_gmem(const int32_t* list_buffer,
                    int32_t& n_neigh,
                    int32_t* neigh_list,
                    const int32_t num_out,
                    const int32_t tid,
                    const int32_t particle_number,
                    const int32_t loc_list_beg) {
  int32_t loc_list_idx = loc_list_beg;
  int32_t neigh_list_idx = n_neigh * particle_number + tid;
  for (int k = 0; k < num_out; k++) {
    neigh_list[neigh_list_idx] = list_buffer[loc_list_idx];
    loc_list_idx += SMEM_BLOCK_NUM;
    neigh_list_idx += particle_number;
  }
  n_neigh += num_out;
}

template <typename Vec, typename Dtype>
__global__ void make_neighlist_smem(const Vec* __restrict__ q,
                                    const int32_t* __restrict__ cell_id_of_ptcl,
                                    const int32_t* __restrict__ neigh_cell_id,
                                    const int32_t* __restrict__ cell_pointer,
                                    int32_t* __restrict__ neigh_list,
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
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto i_cell_id = cell_id_of_ptcl[tid];
    const auto tile_offset = smem_loc_hei * SMEM_BLOCK_NUM;

    const int32_t loc_list_beg = smem_tile_beg(tile_offset) + lane_id();
    int32_t loc_list_idx = loc_list_beg;

    int32_t n_neigh = 0, n_loc_list = 0;
    for (int32_t cid = 0; cid < 27; cid++) {
      const auto j_cell_id = neigh_cell_id[27 * i_cell_id + cid];
      const auto beg_id = cell_pointer[j_cell_id    ];
      const auto end_id = cell_pointer[j_cell_id + 1];
      for (int32_t j = beg_id; j < end_id; j++) {
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
                         neigh_list,
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
                   neigh_list,
                   n_loc_list,
                   tid,
                   particle_number,
                   loc_list_beg);
    number_of_partners[tid] = n_neigh;
  }
}

__device__ __forceinline__
int32_t loc_list_incr(int32_t loc_list_idx,
                      const int32_t loc_list_beg,
                      const int32_t loc_list_end) {
  loc_list_idx += SMEM_BLOCK_NUM;
  if (loc_list_idx > loc_list_end) loc_list_idx = loc_list_beg;
  return loc_list_idx;
}

__device__ __forceinline__
void memcpy_to_gmem(const int32_t* list_buffer,
                    int32_t& n_neigh,
                    int32_t* neigh_list,
                    int32_t& loc_list_org,
                    const int32_t num_out,
                    const int32_t tid,
                    const int32_t particle_number,
                    const int32_t loc_list_beg,
                    const int32_t loc_list_end) {
  int32_t loc_list_idx = loc_list_org;
  int32_t neigh_list_idx = n_neigh * particle_number + tid;
  for (int k = 0; k < num_out; k++) {
    neigh_list[neigh_list_idx] = list_buffer[loc_list_idx];
    loc_list_idx = loc_list_incr(loc_list_idx, loc_list_beg, loc_list_end);
    neigh_list_idx += particle_number;
  }
  loc_list_org = loc_list_idx;
  n_neigh += num_out;
}

template <typename Vec, typename Dtype>
__global__ void make_neighlist_smem_coars(const Vec* __restrict__ q,
                                          const int32_t* __restrict__ cell_id_of_ptcl,
                                          const int32_t* __restrict__ neigh_cell_id,
                                          const int32_t* __restrict__ cell_pointer,
                                          int32_t* __restrict__ neigh_list,
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
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto i_cell_id = cell_id_of_ptcl[tid];
    const auto tile_offset = smem_loc_hei * SMEM_BLOCK_NUM;

    const int32_t loc_list_beg = smem_tile_beg(tile_offset) + lane_id();
    const int32_t loc_list_end = smem_tile_end(tile_offset) + lane_id();
    int32_t loc_list_idx = loc_list_beg;
    int32_t loc_list_org = loc_list_beg;

    int32_t n_neigh = 0, n_loc_list = 0;
    for (int32_t cid = 0; cid < 27; cid++) {
      const auto j_cell_id = neigh_cell_id[27 * i_cell_id + cid];
      const auto beg_id = cell_pointer[j_cell_id    ];
      const auto end_id = cell_pointer[j_cell_id + 1];
      for (int32_t j = beg_id; j < end_id; j++) {
        const auto drx = qi.x - q[j].x;
        const auto dry = qi.y - q[j].y;
        const auto drz = qi.z - q[j].z;
        const auto dr2 = drx * drx + dry * dry + drz * drz;
        if (dr2 < search_length2 && j != tid) {
          list_buffer[loc_list_idx] = j;
          n_loc_list++;
          loc_list_idx = loc_list_incr(loc_list_idx,
                                       loc_list_beg,
                                       loc_list_end);
        }

        const auto write_to_gmem = __any(n_loc_list == smem_loc_hei);
        if (write_to_gmem) {
          const auto num_out = get_min_in_warp(n_loc_list);
          if (num_out != 0) {
            memcpy_to_gmem(list_buffer,
                           n_neigh,
                           neigh_list,
                           loc_list_org,
                           num_out,
                           tid,
                           particle_number,
                           loc_list_beg,
                           loc_list_end);
            n_loc_list -= num_out;
          } else if (n_loc_list != 0) {
            memcpy_to_gmem(list_buffer,
                           n_neigh,
                           neigh_list,
                           loc_list_org,
                           1,
                           tid,
                           particle_number,
                           loc_list_beg,
                           loc_list_end);
            n_loc_list--;
          }
        }
      }
    }

    memcpy_to_gmem(list_buffer,
                   n_neigh,
                   neigh_list,
                   loc_list_org,
                   n_loc_list,
                   tid,
                   particle_number,
                   loc_list_beg,
                   loc_list_end);
    number_of_partners[tid] = n_neigh;
  }
}

template <typename Vec, typename Dtype>
__global__ void make_neighlist_smem_cell(const Vec* __restrict__ q,
                                         const int32_t* __restrict__ neigh_cell_id,
                                         const int32_t* __restrict__ cell_pointer,
                                         int32_t* __restrict__ neigh_list,
                                         int32_t* __restrict__ number_of_partners,
                                         const int32_t smem_loc_hei,
                                         const Dtype search_length2,
                                         const int32_t particle_number) {
  extern __shared__ Vec pos_buffer[];
  const auto i_cell_id = blockIdx.x;
  const auto tid = cell_pointer[i_cell_id] + threadIdx.x;
  const auto qi = q[tid];
  const auto i_end_id = cell_pointer[i_cell_id + 1];

  int32_t n_neigh = 0;
  for (int32_t cid = 0; cid < 27; cid++) {
    const auto j_cell_id  = neigh_cell_id[27 * i_cell_id + cid];
    const auto j_beg_id   = cell_pointer[j_cell_id];

    // copy to smem
    __syncthreads();
    auto j_ptcl_id = j_beg_id + threadIdx.x;
    pos_buffer[threadIdx.x].x = q[j_ptcl_id].x;
    pos_buffer[threadIdx.x].y = q[j_ptcl_id].y;
    pos_buffer[threadIdx.x].z = q[j_ptcl_id].z;
    __syncthreads();

    if (tid < i_end_id) {
      const auto num_loop_j = cell_pointer[j_cell_id + 1] - j_beg_id;
      for (int32_t j = 0; j < num_loop_j; j++) {
        const auto drx = qi.x - pos_buffer[j].x;
        const auto dry = qi.y - pos_buffer[j].y;
        const auto drz = qi.z - pos_buffer[j].z;
        const auto dr2 = drx * drx + dry * dry + drz * drz;
        j_ptcl_id = j + j_beg_id;
        if (dr2 > search_length2 || j_ptcl_id == tid) continue;
        neigh_list[particle_number * n_neigh + tid] = j_ptcl_id;
        n_neigh++;
      }
    }
  }

  if (tid < i_end_id) number_of_partners[tid] = n_neigh;
}

template <typename Vec, typename Dtype>
__global__ void make_neighlist_smem_once(const Vec* __restrict__ q,
                                         const int32_t* __restrict__ cell_id_of_ptcl,
                                         const int32_t* __restrict__ neigh_cell_id,
                                         const int32_t* __restrict__ cell_pointer,
                                         int32_t* __restrict__ neigh_list,
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
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto i_cell_id = cell_id_of_ptcl[tid];
    const auto tile_offset = smem_loc_hei * SMEM_BLOCK_NUM;

    const int32_t loc_list_beg = smem_tile_beg(tile_offset) + lane_id();
    int32_t loc_list_idx = loc_list_beg;

    int32_t n_neigh = 0, n_loc_list = 0;
    bool use_smem = true;
    for (int32_t cid = 0; cid < 27; cid++) {
      const auto j_cell_id = neigh_cell_id[27 * i_cell_id + cid];
      const auto beg_id = cell_pointer[j_cell_id    ];
      const auto end_id = cell_pointer[j_cell_id + 1];
      for (int32_t j = beg_id; j < end_id; j++) {
        const auto drx = qi.x - q[j].x;
        const auto dry = qi.y - q[j].y;
        const auto drz = qi.z - q[j].z;
        const auto dr2 = drx * drx + dry * dry + drz * drz;
        if (use_smem) {
          if (dr2 < search_length2 && j != tid) {
            list_buffer[loc_list_idx] = j;
            n_loc_list++;
            loc_list_idx += SMEM_BLOCK_NUM;
          }
          if (__any(n_loc_list == smem_loc_hei)) {
            memcpy_to_gmem(list_buffer,
                           n_neigh,
                           neigh_list,
                           n_loc_list,
                           tid,
                           particle_number,
                           loc_list_beg);
            n_loc_list = 0;
            loc_list_idx = loc_list_beg;
          }
          use_smem = false;
        } else {
          if (dr2 < search_length2 && j != tid) {
            neigh_list[particle_number * n_neigh + tid] = j;
            n_neigh++;
          }
        }
      }
    }
    number_of_partners[tid] = n_neigh;
  }
}

// implement using cublas
void transpose_neighlist(const int32_t* __restrict__ neigh_list_buf,
                         int32_t* __restrict__ neigh_list,
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
                             reinterpret_cast<const float*>(neigh_list_buf),
                             max_partners,
                             &beta,
                             reinterpret_cast<const float*>(neigh_list_buf),
                             max_partners,
                             reinterpret_cast<float*>(neigh_list),
                             particle_number));
}

template <typename Vec, typename Dtype>
__global__ void make_neighlist_warp_unroll(const Vec* __restrict__ q,
                                           const int32_t* __restrict__ cell_id_of_ptcl,
                                           const int32_t* __restrict__ neigh_cell_id,
                                           const int32_t* __restrict__ cell_pointer,
                                           int32_t* __restrict__ neigh_list_buf,
                                           int32_t* __restrict__ number_of_partners,
                                           const Dtype search_length2,
                                           const int32_t max_partners,
                                           const int32_t particle_number) {
  const auto i_ptcl_id = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  if (i_ptcl_id < particle_number) {
    const auto qi        = q[i_ptcl_id];
    const auto i_cell_id = cell_id_of_ptcl[i_ptcl_id];
    const auto lid       = lane_id();
    int32_t n_neigh      = 0;

    for (int32_t cid = 0; cid < 27; cid++) {
      const auto j_cell_id    = neigh_cell_id[27 * i_cell_id + cid];
      const auto beg_id       = cell_pointer[j_cell_id];
      const auto num_loop     = cell_pointer[j_cell_id + 1] - beg_id;
      const auto num_loop_ini = (num_loop / warpSize) * warpSize;

      int32_t j = 0;
      for (; j < num_loop_ini; j += warpSize) {
        const auto j_ptcl_id = beg_id + j + lid;
        const auto drx = qi.x - q[j_ptcl_id].x;
        const auto dry = qi.y - q[j_ptcl_id].y;
        const auto drz = qi.z - q[j_ptcl_id].z;
        const auto dr2 = drx * drx + dry * dry + drz * drz;
        const int32_t in_range = (dr2 <= search_length2) && (i_ptcl_id != j_ptcl_id);
        const uint32_t flag = __ballot(in_range);
        if (in_range) {
          const uint32_t mask   = (0xffffffff >> (31 - lid));
          const int32_t str_dst = __popc(flag & mask) + n_neigh - 1;
          neigh_list_buf[i_ptcl_id * max_partners + str_dst] = j_ptcl_id;
        }
        n_neigh += __popc(flag);
      }

      // remaining loop
      if (lid < (num_loop % warpSize)) {
        const auto j_ptcl_id = beg_id + j + lid;
        const auto drx = qi.x - q[j_ptcl_id].x;
        const auto dry = qi.y - q[j_ptcl_id].y;
        const auto drz = qi.z - q[j_ptcl_id].z;
        const auto dr2 = drx * drx + dry * dry + drz * drz;
        const int32_t in_range = (dr2 <= search_length2) && (i_ptcl_id != j_ptcl_id);
        const uint32_t flag = __ballot(in_range);
        if (in_range) {
          const uint32_t mask   = (0xffffffff >> (31 - lid));
          const int32_t str_dst = __popc(flag & mask) + n_neigh - 1;
          neigh_list_buf[i_ptcl_id * max_partners + str_dst] = j_ptcl_id;
        }
        n_neigh += __popc(flag);
      }
      n_neigh = __shfl(n_neigh, 0);
    }

    if (lid == 0) number_of_partners[i_ptcl_id] = n_neigh;
  }
}

template <int MAX_PTCL_NUM_IN_NCELL>
__global__ void make_ptcl_id_of_neigh_cell(const int32_t* __restrict__ cell_id_of_ptcl,
                                           const int32_t* __restrict__ neigh_cell_id,
                                           const int32_t* __restrict__ cell_pointer,
                                           int32_t* __restrict__ num_of_ptcl_in_neigh_cell,
                                           int32_t* __restrict__ ptcl_id_of_neigh_cell,
                                           const int32_t tot_cell_num) {
  const auto i_cell_id = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  if (i_cell_id < tot_cell_num) {
    const auto lid = lane_id();
    int32_t* loc_id = &ptcl_id_of_neigh_cell[i_cell_id * MAX_PTCL_NUM_IN_NCELL];

    int32_t n_neigh = 0;
    for (int32_t cid = 0; cid < 27; cid++) {
      const auto j_cell_id    = neigh_cell_id[27 * i_cell_id + cid];
      const auto beg_id       = cell_pointer[j_cell_id];
      const auto num_loop     = cell_pointer[j_cell_id + 1] - beg_id;
      const auto num_loop_ini = (num_loop / warpSize) * warpSize;

      int32_t j = 0;
      for (; j < num_loop_ini; j += warpSize) {
        loc_id[n_neigh + lid] = beg_id + j + lid;
        n_neigh += warpSize;
      }

      const auto remaining_loop = num_loop % warpSize;
      if (lid < remaining_loop) {
        loc_id[n_neigh + lid] = beg_id + j + lid;
      }
      n_neigh += remaining_loop;
    }

    if (lid == 0) num_of_ptcl_in_neigh_cell[i_cell_id] = n_neigh;
  }
}

template <typename Vec, typename Dtype, int MAX_PTCL_NUM_IN_NCELL>
__global__ void make_neighlist_warp_unroll_loop_fused(const Vec* __restrict__ q,
                                                      const int32_t* __restrict__ cell_id_of_ptcl,
                                                      const int32_t* __restrict__ num_of_ptcl_in_neigh_cell,
                                                      const int32_t* __restrict__ ptcl_id_of_neigh_cell,
                                                      int32_t* __restrict__ neigh_list_buf,
                                                      int32_t* __restrict__ number_of_partners,
                                                      const Dtype search_length2,
                                                      const int32_t max_partners,
                                                      const int32_t particle_number) {
  const auto i_ptcl_id = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  if (i_ptcl_id < particle_number) {
    const auto qi        = q[i_ptcl_id];
    const auto i_cell_id = cell_id_of_ptcl[i_ptcl_id];
    const auto lid       = lane_id();
    int32_t n_neigh      = 0;

    const int32_t* loc_id       = &ptcl_id_of_neigh_cell[i_cell_id * MAX_PTCL_NUM_IN_NCELL];
    int32_t* neigh_list_buf_loc = &neigh_list_buf[i_ptcl_id * max_partners];
    const auto num_loop         = num_of_ptcl_in_neigh_cell[i_cell_id];
    const auto num_loop_ini     = (num_loop / warpSize) * warpSize;
    const uint32_t mask         = (0xffffffff >> (31 - lid));

    int32_t j = 0;
    for (; j < num_loop_ini; j += warpSize) {
      const auto j_ptcl_id = loc_id[j + lid];
      const auto drx = qi.x - q[j_ptcl_id].x;
      const auto dry = qi.y - q[j_ptcl_id].y;
      const auto drz = qi.z - q[j_ptcl_id].z;
      const auto dr2 = drx * drx + dry * dry + drz * drz;
      const int32_t in_range = (dr2 <= search_length2) && (i_ptcl_id != j_ptcl_id);
      const uint32_t flag = __ballot(in_range);
      if (in_range) {
        const int32_t str_dst = __popc(flag & mask) + n_neigh - 1;
        neigh_list_buf_loc[str_dst] = j_ptcl_id;
      }
      n_neigh += __popc(flag);
    }

    // remaining loop
    if (lid < (num_loop % warpSize)) {
      const auto j_ptcl_id = loc_id[j + lid];
      const auto drx = qi.x - q[j_ptcl_id].x;
      const auto dry = qi.y - q[j_ptcl_id].y;
      const auto drz = qi.z - q[j_ptcl_id].z;
      const auto dr2 = drx * drx + dry * dry + drz * drz;
      const int32_t in_range = (dr2 <= search_length2) && (i_ptcl_id != j_ptcl_id);
      const uint32_t flag = __ballot(in_range);
      if (in_range) {
        const int32_t str_dst = __popc(flag & mask) + n_neigh - 1;
        neigh_list_buf_loc[str_dst] = j_ptcl_id;
      }
      n_neigh += __popc(flag);
    }

    if (lid == 0) number_of_partners[i_ptcl_id] = n_neigh;
  }
}

template <typename Vec, typename Dtype, int MAX_PTCL_NUM_IN_NCELL>
__global__ void make_neighlist_warp_unroll_loop_fused_rev(const Vec* __restrict__ q,
                                                          const int32_t* __restrict__ num_of_ptcl_in_neigh_cell,
                                                          const int32_t* __restrict__ ptcl_id_of_neigh_cell,
                                                          const int32_t* __restrict__ cell_pointer,
                                                          int32_t* __restrict__ neigh_list_buf,
                                                          int32_t* __restrict__ number_of_partners,
                                                          const Dtype search_length2,
                                                          const int32_t tot_cell_num,
                                                          const int32_t max_partners,
                                                          const int32_t particle_number) {
  const auto i_cell_id = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
  if (i_cell_id < tot_cell_num) {
    const auto beg_id         = cell_pointer[i_cell_id];
    const auto num_i_loop     = cell_pointer[i_cell_id + 1] - beg_id;
    const auto num_i_loop_ini = (num_i_loop >> 2) << 2;

    int32_t i = 0;
    for (; i < num_i_loop_ini; i += 4) {
      const auto i_ptcl_id0 = beg_id + i;
      const auto i_ptcl_id1 = i_ptcl_id0 + 1;
      const auto i_ptcl_id2 = i_ptcl_id0 + 2;
      const auto i_ptcl_id3 = i_ptcl_id0 + 3;

      const auto qi0       = q[i_ptcl_id0];
      const auto qi1       = q[i_ptcl_id1];
      const auto qi2       = q[i_ptcl_id2];
      const auto qi3       = q[i_ptcl_id3];

      const auto lid       = lane_id();
      int32_t n_neigh0 = 0, n_neigh1 = 0, n_neigh2 = 0, n_neigh3 = 0;

      int32_t* neigh_list_buf_loc0 = &neigh_list_buf[i_ptcl_id0 * max_partners];
      int32_t* neigh_list_buf_loc1 = &neigh_list_buf[i_ptcl_id1 * max_partners];
      int32_t* neigh_list_buf_loc2 = &neigh_list_buf[i_ptcl_id2 * max_partners];
      int32_t* neigh_list_buf_loc3 = &neigh_list_buf[i_ptcl_id3 * max_partners];

      const int32_t* loc_id     = &ptcl_id_of_neigh_cell[i_cell_id * MAX_PTCL_NUM_IN_NCELL];
      const auto num_j_loop     = num_of_ptcl_in_neigh_cell[i_cell_id];
      const auto num_j_loop_ini = (num_j_loop / warpSize) * warpSize;
      const uint32_t mask       = (0xffffffff >> (31 - lid));

      int32_t j = 0;
      for (; j < num_j_loop_ini; j += warpSize) {
        const auto j_ptcl_id = loc_id[j + lid];
        const auto qj = q[j_ptcl_id];

        const auto drx0  = qi0.x - qj.x;
        const auto dry0  = qi0.y - qj.y;
        const auto drz0  = qi0.z - qj.z;
        const auto dr2_0 = drx0 * drx0 + dry0 * dry0 + drz0 * drz0;

        const auto drx1  = qi1.x - qj.x;
        const auto dry1  = qi1.y - qj.y;
        const auto drz1  = qi1.z - qj.z;
        const auto dr2_1 = drx1 * drx1 + dry1 * dry1 + drz1 * drz1;

        const auto drx2  = qi2.x - qj.x;
        const auto dry2  = qi2.y - qj.y;
        const auto drz2  = qi2.z - qj.z;
        const auto dr2_2 = drx2 * drx2 + dry2 * dry2 + drz2 * drz2;

        const auto drx3  = qi3.x - qj.x;
        const auto dry3  = qi3.y - qj.y;
        const auto drz3  = qi3.z - qj.z;
        const auto dr2_3 = drx3 * drx3 + dry3 * dry3 + drz3 * drz3;

        int32_t in_range = (dr2_0 <= search_length2) && (i_ptcl_id0 != j_ptcl_id);
        uint32_t flag = __ballot(in_range);
        if (in_range) {
          const int32_t str_dst = __popc(flag & mask) + n_neigh0 - 1;
          neigh_list_buf_loc0[str_dst] = j_ptcl_id;
        }
        n_neigh0 += __popc(flag);

        in_range = (dr2_1 <= search_length2) && (i_ptcl_id1 != j_ptcl_id);
        flag = __ballot(in_range);
        if (in_range) {
          const int32_t str_dst = __popc(flag & mask) + n_neigh1 - 1;
          neigh_list_buf_loc1[str_dst] = j_ptcl_id;
        }
        n_neigh1 += __popc(flag);

        in_range = (dr2_2 <= search_length2) && (i_ptcl_id2 != j_ptcl_id);
        flag = __ballot(in_range);
        if (in_range) {
          const int32_t str_dst = __popc(flag & mask) + n_neigh2 - 1;
          neigh_list_buf_loc2[str_dst] = j_ptcl_id;
        }
        n_neigh2 += __popc(flag);

        in_range = (dr2_3 <= search_length2) && (i_ptcl_id3 != j_ptcl_id);
        flag = __ballot(in_range);
        if (in_range) {
          const int32_t str_dst = __popc(flag & mask) + n_neigh3 - 1;
          neigh_list_buf_loc3[str_dst] = j_ptcl_id;
        }
        n_neigh3 += __popc(flag);
      }

      // remaining loop
      if (lid < (num_j_loop - num_j_loop_ini)) {
        const auto j_ptcl_id = loc_id[j + lid];
        const auto qj = q[j_ptcl_id];

        const auto drx0  = qi0.x - qj.x;
        const auto dry0  = qi0.y - qj.y;
        const auto drz0  = qi0.z - qj.z;
        const auto dr2_0 = drx0 * drx0 + dry0 * dry0 + drz0 * drz0;

        const auto drx1  = qi1.x - qj.x;
        const auto dry1  = qi1.y - qj.y;
        const auto drz1  = qi1.z - qj.z;
        const auto dr2_1 = drx1 * drx1 + dry1 * dry1 + drz1 * drz1;

        const auto drx2  = qi2.x - qj.x;
        const auto dry2  = qi2.y - qj.y;
        const auto drz2  = qi2.z - qj.z;
        const auto dr2_2 = drx2 * drx2 + dry2 * dry2 + drz2 * drz2;

        const auto drx3  = qi3.x - qj.x;
        const auto dry3  = qi3.y - qj.y;
        const auto drz3  = qi3.z - qj.z;
        const auto dr2_3 = drx3 * drx3 + dry3 * dry3 + drz3 * drz3;

        int32_t in_range = (dr2_0 <= search_length2) && (i_ptcl_id0 != j_ptcl_id);
        uint32_t flag = __ballot(in_range);
        if (in_range) {
          const int32_t str_dst = __popc(flag & mask) + n_neigh0 - 1;
          neigh_list_buf_loc0[str_dst] = j_ptcl_id;
        }
        n_neigh0 += __popc(flag);

        in_range = (dr2_1 <= search_length2) && (i_ptcl_id1 != j_ptcl_id);
        flag = __ballot(in_range);
        if (in_range) {
          const int32_t str_dst = __popc(flag & mask) + n_neigh1 - 1;
          neigh_list_buf_loc1[str_dst] = j_ptcl_id;
        }
        n_neigh1 += __popc(flag);

        in_range = (dr2_2 <= search_length2) && (i_ptcl_id2 != j_ptcl_id);
        flag = __ballot(in_range);
        if (in_range) {
          const int32_t str_dst = __popc(flag & mask) + n_neigh2 - 1;
          neigh_list_buf_loc2[str_dst] = j_ptcl_id;
        }
        n_neigh2 += __popc(flag);

        in_range = (dr2_3 <= search_length2) && (i_ptcl_id3 != j_ptcl_id);
        flag = __ballot(in_range);
        if (in_range) {
          const int32_t str_dst = __popc(flag & mask) + n_neigh3 - 1;
          neigh_list_buf_loc3[str_dst] = j_ptcl_id;
        }
        n_neigh3 += __popc(flag);
      }

      if (lid == 0) {
        number_of_partners[i_ptcl_id0] = n_neigh0;
        number_of_partners[i_ptcl_id1] = n_neigh1;
        number_of_partners[i_ptcl_id2] = n_neigh2;
        number_of_partners[i_ptcl_id3] = n_neigh3;
      }
    }

    for (; i < num_i_loop; i++) {
      const auto i_ptcl_id = beg_id + i;
      const auto qi        = q[i_ptcl_id];
      const auto lid       = lane_id();
      int32_t n_neigh      = 0;

      const int32_t* loc_id       = &ptcl_id_of_neigh_cell[i_cell_id * MAX_PTCL_NUM_IN_NCELL];
      int32_t* neigh_list_buf_loc = &neigh_list_buf[i_ptcl_id * max_partners];
      const auto num_loop         = num_of_ptcl_in_neigh_cell[i_cell_id];
      const auto num_loop_ini     = (num_loop / warpSize) * warpSize;
      const uint32_t mask         = (0xffffffff >> (31 - lid));

      int32_t j = 0;
      for (; j < num_loop_ini; j += warpSize) {
        const auto j_ptcl_id = loc_id[j + lid];
        const auto drx = qi.x - q[j_ptcl_id].x;
        const auto dry = qi.y - q[j_ptcl_id].y;
        const auto drz = qi.z - q[j_ptcl_id].z;
        const auto dr2 = drx * drx + dry * dry + drz * drz;
        const int32_t in_range = (dr2 <= search_length2) && (i_ptcl_id != j_ptcl_id);
        const uint32_t flag = __ballot(in_range);
        if (in_range) {
          const int32_t str_dst = __popc(flag & mask) + n_neigh - 1;
          neigh_list_buf_loc[str_dst] = j_ptcl_id;
        }
        n_neigh += __popc(flag);
      }

      // remaining loop
      if (lid < (num_loop - num_loop_ini)) {
        const auto j_ptcl_id = loc_id[j + lid];
        const auto drx = qi.x - q[j_ptcl_id].x;
        const auto dry = qi.y - q[j_ptcl_id].y;
        const auto drz = qi.z - q[j_ptcl_id].z;
        const auto dr2 = drx * drx + dry * dry + drz * drz;
        const int32_t in_range = (dr2 <= search_length2) && (i_ptcl_id != j_ptcl_id);
        const uint32_t flag = __ballot(in_range);
        if (in_range) {
          const int32_t str_dst = __popc(flag & mask) + n_neigh - 1;
          neigh_list_buf_loc[str_dst] = j_ptcl_id;
        }
        n_neigh += __popc(flag);
      }

      if (lid == 0) number_of_partners[i_ptcl_id] = n_neigh;
    }
  }
}
