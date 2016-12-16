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
  if (smem_loc_hei >= SMEM_MAX_HEI) {
    printf("smem_loc_hei is too large!\n");
    printf("smem_loc_hei = %d SMEM_MAX_HEI = %d", smem_loc_hei, SMEM_MAX_HEI);
    return;
  }

  extern __shared__ int32_t list_buffer[];
  const auto i_cell_id = blockIdx.x;
  const auto tid = cell_pointer[i_cell_id] + threadIdx.x;

  if (tid < cell_pointer[i_cell_id + 1]) {
    const auto qi = q[tid];
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

