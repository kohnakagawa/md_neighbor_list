#pragma once

template <typename Vec, typename Dtype>
__global__ void make_neighlist_naive(const Vec* q,
                                     const int32_t* grid_id_of_ptcl,
                                     const int32_t* neigh_grid_id,
                                     const int32_t* grid_pointer,
                                     int32_t* neigh_list,
                                     int32_t* number_of_partners,
                                     const Dtype search_length2,
                                     const int32_t particle_number) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < particle_number) {
    const auto qi = q[tid];
    const auto i_grid_id = grid_id_of_ptcl[tid];
    int32_t n_neigh = 0;
    for (int32_t gid = 0; gid < 27; gid++) {
      const auto j_grid_id = neigh_grid_id[27 * i_grid_id + gid];
      const auto beg_id = grid_pointer[j_grid_id    ];
      const auto end_id = grid_pointer[j_grid_id + 1];
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
