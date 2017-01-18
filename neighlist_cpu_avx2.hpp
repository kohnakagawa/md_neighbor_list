#pragma once

#include <cassert>
#include <vector>
#include <numeric>

#include "dyalloc2d.hpp"
#include "simd_util.hpp"

template <typename Vec>
class NeighListAVX2 {
  bool valid_ = false;
  int32_t cell_numb_[3], all_cell_ = -1;
  Vec cell_leng_, inv_cell_leng_;
  double search_length_  = 0.0, search_length2_ = 0.0;

  int32_t number_of_pairs_ = 0;

  int32_t *cell_id_of_ptcl_ = nullptr, *neigh_cell_id_ = nullptr;
  int32_t *number_in_cell_ = nullptr;
  int32_t *cell_pointer_ = nullptr, *cell_pointer_buf_ = nullptr;
  int32_t *ptcl_id_in_cell_ = nullptr;
  int32_t *next_dst_ = nullptr;

  int32_t *neigh_list_ = nullptr, *number_of_partners_ = nullptr;
  int32_t *neigh_pointer_ = nullptr, *neigh_pointer_buf_ = nullptr;

  int32_t **key_partner_particles_ = nullptr;

  std::vector<int32_t>* ptcl_id_of_neigh_cell_ = nullptr;

  Vec *data_buf_ = nullptr;

  int32_t shfl_table_[16][8] {0};

  enum : int32_t {
    MAX_PARTNERS = 100,
    SORT_FREQ = 50,
    NUM_NEIGH_CELL = 13,
    NUM_PTCL_IN_NEIGH_CELL = 500,
  };

  enum : int32_t {
    KEY = 0,
    PARTNER = 1
  };

  void GenShflTable() {
    for (int i = 0; i < 16; i++) {
      auto tbl_id = i;
      int cnt = 0;
      for (int j = 0; j < 4; j++) {
        if (tbl_id & 0x1) {
          shfl_table_[i][cnt++] = 2 * j;
          shfl_table_[i][cnt++] = 2 * j + 1;
        }
        tbl_id >>= 1;
      }
    }
  }

  int32_t GenHash(const int32_t* idx) const {
    const auto ret = idx[0] + (idx[1] + idx[2] * cell_numb_[1]) * cell_numb_[0];
#ifdef DEBUG
    assert(ret >= 0);
    assert(ret < all_cell_);
#endif
    return ret;
  }

  int32_t GenHash(const Vec& q) const {
    int32_t idx[] = {
      static_cast<int32_t>(q.x * inv_cell_leng_.x),
      static_cast<int32_t>(q.y * inv_cell_leng_.y),
      static_cast<int32_t>(q.z * inv_cell_leng_.z)
    };
    ApplyPBC(idx);
    return GenHash(idx);
  }

  void ApplyPBC(int32_t* idx) const {
    for (int i = 0; i < 3; i++) {
      if (idx[i] < 0) idx[i] += cell_numb_[i];
      if (idx[i] >= cell_numb_[i]) idx[i] -= cell_numb_[i];
    }
  }

  void Allocate(const int32_t particle_number) {
    cell_id_of_ptcl_       = new int32_t [particle_number];
    neigh_cell_id_         = new int32_t [NUM_NEIGH_CELL * all_cell_];
    number_in_cell_        = new int32_t [all_cell_];
    cell_pointer_          = new int32_t [all_cell_ + 1];
    cell_pointer_buf_      = new int32_t [all_cell_ + 1];
    ptcl_id_in_cell_       = new int32_t [particle_number];
    next_dst_              = new int32_t [particle_number];
    neigh_list_            = new int32_t [MAX_PARTNERS * particle_number];
    number_of_partners_    = new int32_t [particle_number];
    neigh_pointer_         = new int32_t [particle_number + 1];
    neigh_pointer_buf_     = new int32_t [particle_number + 1];
    allocate2D_aligend<int32_t, 32>(MAX_PARTNERS * particle_number, 2, key_partner_particles_);
    ptcl_id_of_neigh_cell_ = new std::vector<int32_t> [all_cell_];
    data_buf_              = new Vec [particle_number];
    for (int32_t i = 0; i < all_cell_; i++) {
      ptcl_id_of_neigh_cell_[i].resize(NUM_PTCL_IN_NEIGH_CELL);
    }
  }

  void Deallocate() {
    delete [] cell_id_of_ptcl_;
    delete [] neigh_cell_id_;
    delete [] number_in_cell_;
    delete [] cell_pointer_;
    delete [] cell_pointer_buf_;
    delete [] ptcl_id_in_cell_;
    delete [] next_dst_;
    delete [] neigh_list_;
    delete [] number_of_partners_;
    delete [] neigh_pointer_;
    delete [] neigh_pointer_buf_;
    deallocate2D_aligend(key_partner_particles_);
    delete [] ptcl_id_of_neigh_cell_;
    delete [] data_buf_;
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
                neigh_cell_id_[NUM_NEIGH_CELL * icell_id + jcell_id] = GenHash(idx);
                jcell_id++;
                if (jcell_id == NUM_NEIGH_CELL) goto OUT;
              }
        OUT:
          icell_id++;
        }
#ifdef DEBUG
    assert(icell_id == all_cell_);
    for (int i = 0; i < all_cell_ * NUM_NEIGH_CELL; i++) {
      assert(neigh_cell_id_[i] >= 0);
      assert(neigh_cell_id_[i] < all_cell_);
    }
#endif
  }

  void MakeCellidOfPtcl(const Vec* q,
                        const int32_t particle_number) {
    std::fill(number_in_cell_,
              number_in_cell_ + all_cell_,
              0);
    for (int32_t i = 0; i < particle_number; i++) {
      const auto hash = GenHash(q[i]);
      number_in_cell_[hash]++;
      cell_id_of_ptcl_[i] = hash;
    }
  }

  void MakeNextDest(const int32_t particle_number) {
    cell_pointer_[0] = cell_pointer_buf_[0] = 0;
    for (int32_t i = 0; i < all_cell_; i++) {
      const auto g_ptr = cell_pointer_[i] + number_in_cell_[i];
      cell_pointer_[i + 1] = g_ptr;
      cell_pointer_buf_[i + 1] = g_ptr;
    }

    for (int32_t i = 0; i < particle_number; i++) {
      const auto hash = cell_id_of_ptcl_[i];
      const auto dst = cell_pointer_buf_[hash];
      next_dst_[i] = dst;
      ptcl_id_in_cell_[dst] = i;
      cell_pointer_buf_[hash]++;
    }

#ifdef DEBUG
    assert(cell_pointer_[all_cell_] == particle_number);
#endif
  }

  template <typename T>
  void Gather(T* __restrict dat,
              T* __restrict buf,
              const int elem,
              int32_t* __restrict dst) {
    for (int32_t i = 0; i < elem; i++) buf[i] = dat[i];
    for (int32_t i = 0; i < elem; i++) dat[dst[i]] = buf[i];
  }

  void SortPtclData(Vec* __restrict q,
                    Vec* __restrict p,
                    const int32_t particle_number) {
    Gather(q, data_buf_, particle_number, next_dst_);
    Gather(p, data_buf_, particle_number, next_dst_);
    std::iota(ptcl_id_in_cell_, ptcl_id_in_cell_ + particle_number, 0);
  }

  void CheckSorted(const Vec* q) const {
    for (int32_t cell = 0; cell < all_cell_; cell++) {
      const auto beg = cell_pointer_[cell    ];
      const auto end = cell_pointer_[cell + 1];
      for (int32_t i = beg; i < end; i++) {
        const auto hash = GenHash(q[i]);
        if (hash != cell) {
          std::cerr << "particle data is not correctly sorted.\n";
          std::exit(1);
        }
      }
    }
  }

  void MakeNeighCellPtclId() {
    for (int32_t icell = 0; icell < all_cell_; icell++) {
      ptcl_id_of_neigh_cell_[icell].clear();
      const auto icell_beg = cell_pointer_[icell    ];
      const auto icell_end = cell_pointer_[icell + 1];
      ptcl_id_of_neigh_cell_[icell].insert(ptcl_id_of_neigh_cell_[icell].end(),
                                           &ptcl_id_in_cell_[icell_beg],
                                           &ptcl_id_in_cell_[icell_end]);
      for (int32_t k = 0; k < NUM_NEIGH_CELL; k++) {
        const auto jcell = neigh_cell_id_[NUM_NEIGH_CELL * icell + k];
        const auto jcell_beg = cell_pointer_[jcell    ];
        const auto jcell_end = cell_pointer_[jcell + 1];
        ptcl_id_of_neigh_cell_[icell].insert(ptcl_id_of_neigh_cell_[icell].end(),
                                             &ptcl_id_in_cell_[jcell_beg],
                                             &ptcl_id_in_cell_[jcell_end]);
      }
    }
  }

  void RegistPair(const int32_t index1,
                  const int32_t index2) {
    int i, j;
    if (index1 < index2) {
      i = index1;
      j = index2;
    } else {
      i = index2;
      j = index1;
    }
    key_partner_particles_[number_of_pairs_][KEY] = i;
    key_partner_particles_[number_of_pairs_][PARTNER] = j;
    number_of_pairs_++;
  }

  void RegistInteractPair(const Vec& qi,
                          const Vec& qj,
                          const int32_t index1,
                          const int32_t index2) {
    const auto dx = qj.x - qi.x;
    const auto dy = qj.y - qi.y;
    const auto dz = qj.z - qi.z;
    const auto r2 = dx * dx + dy * dy + dz * dz;
    if (r2 > search_length2_) return;

    RegistPair(index1, index2);
  }

  void MakePairListSIMD1x4SeqStore(const Vec* q,
                                   const int32_t particle_number) {
    MakeNeighCellPtclId();
    number_of_pairs_ = 0;
    const v4df vsl2 = _mm256_set1_pd(search_length2_);
    for (int32_t icell = 0; icell < all_cell_; icell++) {
      const auto icell_beg = cell_pointer_[icell];
      const auto icell_size = number_in_cell_[icell];
      const int32_t* pid_of_neigh_cell_loc = &ptcl_id_of_neigh_cell_[icell][0];
      const int32_t num_of_neigh_cell = ptcl_id_of_neigh_cell_[icell].size();
      for (int32_t l = 0; l < icell_size; l++) {
        const auto i = l + icell_beg;
        v4df vqix = _mm256_set1_pd(q[i].x);
        v4df vqiy = _mm256_set1_pd(q[i].y);
        v4df vqiz = _mm256_set1_pd(q[i].z);

        const auto num_loop = num_of_neigh_cell - (l + 1);
        for (int32_t k = 0; k < (num_loop / 4) * 4; k += 4) {
          const auto ja = pid_of_neigh_cell_loc[k + l + 1];
          const auto jb = pid_of_neigh_cell_loc[k + l + 2];
          const auto jc = pid_of_neigh_cell_loc[k + l + 3];
          const auto jd = pid_of_neigh_cell_loc[k + l + 4];

          const v4df vqja = _mm256_load_pd(&q[ja].x);
          const v4df vqjb = _mm256_load_pd(&q[jb].x);
          const v4df vqjc = _mm256_load_pd(&q[jc].x);
          const v4df vqjd = _mm256_load_pd(&q[jd].x);

          v4df vqjx, vqjy, vqjz;
          transpose_4x4(vqja, vqjb, vqjc, vqjd,
                        vqjx, vqjy, vqjz);

          v4df dvx = vqjx - vqix;
          v4df dvy = vqjy - vqiy;
          v4df dvz = vqjz - vqiz;

          // norm
          v4df dr2 = dvx * dvx + dvy * dvy + dvz * dvz;

          // dr2 <= search_length2
          v4df dr2_flag = _mm256_cmp_pd(dr2, vsl2, _CMP_LE_OS);

          int32_t hash = _mm256_movemask_pd(dr2_flag);

          if (hash == 0) continue;

          if (hash & 1) RegistPair(i, ja);
          hash >>= 1;
          if (hash & 1) RegistPair(i, jb);
          hash >>= 1;
          if (hash & 1) RegistPair(i, jc);
          hash >>= 1;
          if (hash & 1) RegistPair(i, jd);
        }

        for (int32_t k = (num_loop / 4) * 4; k < num_loop; k++) {
          const auto j = pid_of_neigh_cell_loc[k + l + 1];
          RegistInteractPair(q[i], q[j], i, j);
        }
      }
    }
  }

  void MakePairListSIMD4x1SeqStore(const Vec* q,
                                            const int32_t particle_number) {
    MakeNeighCellPtclId();
    number_of_pairs_ = 0;
    const v4df vsl2 = _mm256_set_pd(search_length2_,
                                    search_length2_,
                                    search_length2_,
                                    search_length2_);
    for (int32_t icell = 0; icell < all_cell_; icell++) {
      const auto icell_beg = cell_pointer_[icell    ];
      const auto icell_end = cell_pointer_[icell + 1];
      const auto icell_size = icell_end - icell_beg;
      const int32_t* pid_of_neigh_cell_loc = &ptcl_id_of_neigh_cell_[icell][0];
      const int32_t num_of_neigh_cell = ptcl_id_of_neigh_cell_[icell].size();
      for (int32_t l = 0; l < (icell_size / 4) * 4 ; l += 4) {
        const auto i_a = l + icell_beg;
        const auto i_b = l + icell_beg + 1;
        const auto i_c = l + icell_beg + 2;
        const auto i_d = l + icell_beg + 3;

        v4df vqia = _mm256_load_pd(&q[i_a].x);
        v4df vqib = _mm256_load_pd(&q[i_b].x);
        v4df vqic = _mm256_load_pd(&q[i_c].x);
        v4df vqid = _mm256_load_pd(&q[i_d].x);

        v4df vqix, vqiy, vqiz;
        transpose_4x4(vqia, vqib, vqic, vqid, vqix, vqiy, vqiz);
        for (int32_t k = l + 4; k < num_of_neigh_cell; k++) {
          const auto j = pid_of_neigh_cell_loc[k];

          v4df vqjx = _mm256_set1_pd(q[j].x);
          v4df vqjy = _mm256_set1_pd(q[j].y);
          v4df vqjz = _mm256_set1_pd(q[j].z);

          v4df dvx = vqjx - vqix;
          v4df dvy = vqjy - vqiy;
          v4df dvz = vqjz - vqiz;

          // norm
          v4df dr2  = dvx * dvx + dvy * dvy + dvz * dvz;

          // dr2 <= search_length2
          v4df dr2_flag = _mm256_cmp_pd(dr2, vsl2, _CMP_LE_OS);

          int32_t hash = _mm256_movemask_pd(dr2_flag);

          if (hash == 0) continue;

          if (hash & 1) RegistPair(i_a, j);
          hash >>= 1;
          if (hash & 1) RegistPair(i_b, j);
          hash >>= 1;
          if (hash & 1) RegistPair(i_c, j);
          hash >>= 1;
          if (hash & 1) RegistPair(i_d, j);
        }

        // remaining pairs
        RegistInteractPair(q[i_a], q[i_b], i_a, i_b);
        RegistInteractPair(q[i_a], q[i_c], i_a, i_c);
        RegistInteractPair(q[i_a], q[i_d], i_a, i_d);
        RegistInteractPair(q[i_b], q[i_c], i_b, i_c);
        RegistInteractPair(q[i_b], q[i_d], i_b, i_d);
        RegistInteractPair(q[i_c], q[i_d], i_c, i_d);
      }

      // remaining i loop
      for (int32_t l = (icell_size / 4) * 4; l < icell_size; l++) {
        const auto i = l + icell_beg;
        v4df vqix = _mm256_set1_pd(q[i].x);
        v4df vqiy = _mm256_set1_pd(q[i].y);
        v4df vqiz = _mm256_set1_pd(q[i].z);

        const auto num_loop = num_of_neigh_cell - (l + 1);
        for (int32_t k = 0; k < (num_loop / 4) * 4; k += 4) {
          const auto ja = pid_of_neigh_cell_loc[k + l + 1];
          const auto jb = pid_of_neigh_cell_loc[k + l + 2];
          const auto jc = pid_of_neigh_cell_loc[k + l + 3];
          const auto jd = pid_of_neigh_cell_loc[k + l + 4];

          const v4df vqja = _mm256_load_pd(&q[ja].x);
          const v4df vqjb = _mm256_load_pd(&q[jb].x);
          const v4df vqjc = _mm256_load_pd(&q[jc].x);
          const v4df vqjd = _mm256_load_pd(&q[jd].x);

          v4df vqjx, vqjy, vqjz;
          transpose_4x4(vqja, vqjb, vqjc, vqjd,
                        vqjx, vqjy, vqjz);

          v4df dvx = vqjx - vqix;
          v4df dvy = vqjy - vqiy;
          v4df dvz = vqjz - vqiz;

          // norm
          v4df dr2 = dvx * dvx + dvy * dvy + dvz * dvz;

          // dr2 <= search_length2
          v4df dr2_flag = _mm256_cmp_pd(dr2, vsl2, _CMP_LE_OS);

          int32_t hash = _mm256_movemask_pd(dr2_flag);

          if (hash == 0) continue;

          if (hash & 1) RegistPair(i, ja);
          hash >>= 1;
          if (hash & 1) RegistPair(i, jb);
          hash >>= 1;
          if (hash & 1) RegistPair(i, jc);
          hash >>= 1;
          if (hash & 1) RegistPair(i, jd);
        }

        for (int32_t k = (num_loop / 4) * 4; k < num_loop; k++) {
          const auto j = pid_of_neigh_cell_loc[k + l + 1];
          RegistInteractPair(q[i], q[j], i, j);
        }
      }
    }
  }

  void MakePairListSIMD1x4(const Vec* q,
                           const int32_t particle_number) {
    MakeNeighCellPtclId();
    number_of_pairs_ = 0;
    const v4df vsl2 = _mm256_set1_pd(search_length2_);
    for (int32_t icell = 0; icell < all_cell_; icell++) {
      const auto icell_beg = cell_pointer_[icell];
      const auto icell_size = number_in_cell_[icell];
      const int32_t* pid_of_neigh_cell_loc = &ptcl_id_of_neigh_cell_[icell][0];
      const int32_t num_of_neigh_cell = ptcl_id_of_neigh_cell_[icell].size();
      for (int32_t l = 0; l < icell_size; l++) {
        const auto i = l + icell_beg;
        v4df vqix = _mm256_set1_pd(q[i].x);
        v4df vqiy = _mm256_set1_pd(q[i].y);
        v4df vqiz = _mm256_set1_pd(q[i].z);
        v4di vi_id = _mm256_set1_epi64x(i);

        const auto num_loop = num_of_neigh_cell - (l + 1);
        for (int32_t k = 0; k < (num_loop / 4) * 4; k += 4) {
          const auto ja = pid_of_neigh_cell_loc[k + l + 1];
          const auto jb = pid_of_neigh_cell_loc[k + l + 2];
          const auto jc = pid_of_neigh_cell_loc[k + l + 3];
          const auto jd = pid_of_neigh_cell_loc[k + l + 4];

          v4df vqja = _mm256_load_pd(&q[ja].x);
          v4df vqjb = _mm256_load_pd(&q[jb].x);
          v4df vqjc = _mm256_load_pd(&q[jc].x);
          v4df vqjd = _mm256_load_pd(&q[jd].x);

          v4df vqjx, vqjy, vqjz;
          transpose_4x4(vqja, vqjb, vqjc, vqjd,
                        vqjx, vqjy, vqjz);

          v4df dvx = vqjx - vqix;
          v4df dvy = vqjy - vqiy;
          v4df dvz = vqjz - vqiz;

          // norm
          v4df dr2 = dvx * dvx + dvy * dvy + dvz * dvz;

          // dr2 <= search_length2
          v4df dr2_flag = _mm256_cmp_pd(dr2, vsl2, _CMP_LE_OS);

          // get shfl hash
          const int32_t hash = _mm256_movemask_pd(dr2_flag);

          if (hash == 0) continue;

          const int incr = _popcnt32(hash);

          // key_id < part_id
          v4di vj_id = _mm256_set_epi64x(jd, jc, jb, ja);
          v8si vkey_id = _mm256_min_epi32(vi_id, vj_id);
          v8si vpart_id = _mm256_max_epi32(vi_id, vj_id);
          vpart_id = _mm256_slli_si256(vpart_id, 4);
          v8si vpart_key_id = _mm256_or_si256(vkey_id, vpart_id);

          // shuffle id and store pair data
          v8si idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(shfl_table_[hash]));
          vpart_key_id = _mm256_permutevar8x32_epi32(vpart_key_id, idx);
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(key_partner_particles_[number_of_pairs_]),
                              vpart_key_id);

          number_of_pairs_ += incr;
        }

        for (int32_t k = (num_loop / 4) * 4; k < num_loop; k++) {
          const auto j = pid_of_neigh_cell_loc[k + l + 1];
          RegistInteractPair(q[i], q[j], i, j);
        }
      }
    }
  }

  void MakePairListSIMD4x1(const Vec* q,
                           const int32_t particle_number) {
    MakeNeighCellPtclId();
    number_of_pairs_ = 0;
    const v4df vsl2 = _mm256_set1_pd(search_length2_);
    for (int32_t icell = 0; icell < all_cell_; icell++) {
      const auto icell_beg = cell_pointer_[icell    ];
      const auto icell_end = cell_pointer_[icell + 1];
      const auto icell_size = icell_end - icell_beg;
      const int32_t* pid_of_neigh_cell_loc = &ptcl_id_of_neigh_cell_[icell][0];
      const int32_t num_of_neigh_cell = ptcl_id_of_neigh_cell_[icell].size();
      for (int32_t l = 0; l < (icell_size / 4) * 4 ; l += 4) {
        const auto i_a = l + icell_beg;
        const auto i_b = l + icell_beg + 1;
        const auto i_c = l + icell_beg + 2;
        const auto i_d = l + icell_beg + 3;

        v4df vqia = _mm256_load_pd(&q[i_a].x);
        v4df vqib = _mm256_load_pd(&q[i_b].x);
        v4df vqic = _mm256_load_pd(&q[i_c].x);
        v4df vqid = _mm256_load_pd(&q[i_d].x);

        v4df vqix, vqiy, vqiz;
        transpose_4x4(vqia, vqib, vqic, vqid, vqix, vqiy, vqiz);

        v4di vi_id = _mm256_set_epi64x(i_d, i_c, i_b, i_a);
        for (int32_t k = l + 4; k < num_of_neigh_cell; k++) {
          const auto j = pid_of_neigh_cell_loc[k];
          v4df vqjx = _mm256_set1_pd(q[j].x);
          v4df vqjy = _mm256_set1_pd(q[j].y);
          v4df vqjz = _mm256_set1_pd(q[j].z);

          v4df dvx = vqjx - vqix;
          v4df dvy = vqjy - vqiy;
          v4df dvz = vqjz - vqiz;

          // norm
          v4df dr2 = dvx * dvx + dvy * dvy + dvz * dvz;

          // dr2 <= search_length2
          v4df dr2_flag = _mm256_cmp_pd(dr2, vsl2, _CMP_LE_OS);

          // get shfl hash
          const int32_t hash = _mm256_movemask_pd(dr2_flag);

          if (hash == 0) continue;

          const int incr = _popcnt32(hash);

          // key_id < part_id
          v4di vj_id = _mm256_set_epi64x(j, j, j, j);
          v8si vkey_id = _mm256_min_epi32(vi_id, vj_id);
          v8si vpart_id = _mm256_max_epi32(vi_id, vj_id);
          vpart_id = _mm256_slli_si256(vpart_id, 4);
          v8si vpart_key_id = _mm256_or_si256(vkey_id, vpart_id);

          // shuffle id and store pair data
          v8si idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(shfl_table_[hash]));
          vpart_key_id = _mm256_permutevar8x32_epi32(vpart_key_id, idx);
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(key_partner_particles_[number_of_pairs_]),
                              vpart_key_id);

          number_of_pairs_ += incr;
        }
        // remaining pairs
        RegistInteractPair(q[i_a], q[i_b], i_a, i_b);
        RegistInteractPair(q[i_a], q[i_c], i_a, i_c);
        RegistInteractPair(q[i_a], q[i_d], i_a, i_d);
        RegistInteractPair(q[i_b], q[i_c], i_b, i_c);
        RegistInteractPair(q[i_b], q[i_d], i_b, i_d);
        RegistInteractPair(q[i_c], q[i_d], i_c, i_d);
      }

      // remaining i loop
      for (int32_t l = (icell_size / 4) * 4; l < icell_size; l++) {
        const auto i = l + icell_beg;
        v4df vqix = _mm256_set1_pd(q[i].x);
        v4df vqiy = _mm256_set1_pd(q[i].y);
        v4df vqiz = _mm256_set1_pd(q[i].z);
        v4di vi_id = _mm256_set1_epi64x(i);

        const auto num_loop = num_of_neigh_cell - (l + 1);
        for (int32_t k = 0; k < (num_loop / 4) * 4; k += 4) {
          const auto ja = pid_of_neigh_cell_loc[k + l + 1];
          const auto jb = pid_of_neigh_cell_loc[k + l + 2];
          const auto jc = pid_of_neigh_cell_loc[k + l + 3];
          const auto jd = pid_of_neigh_cell_loc[k + l + 4];

          v4df vqja = _mm256_load_pd(&q[ja].x);
          v4df vqjb = _mm256_load_pd(&q[jb].x);
          v4df vqjc = _mm256_load_pd(&q[jc].x);
          v4df vqjd = _mm256_load_pd(&q[jd].x);

          v4df vqjx, vqjy, vqjz;
          transpose_4x4(vqja, vqjb, vqjc, vqjd,
                        vqjx, vqjy, vqjz);

          v4df dvx = vqjx - vqix;
          v4df dvy = vqjy - vqiy;
          v4df dvz = vqjz - vqiz;

          // norm
          v4df dr2 = dvx * dvx + dvy * dvy + dvz * dvz;

          // dr2 <= search_length2
          v4df dr2_flag = _mm256_cmp_pd(dr2, vsl2, _CMP_LE_OS);

          // get shfl hash
          const int32_t hash = _mm256_movemask_pd(dr2_flag);

          if (hash == 0) continue;

          const int incr = _popcnt32(hash);

          // key_id < part_id
          v4di vj_id = _mm256_set_epi64x(jd, jc, jb, ja);
          v8si vkey_id = _mm256_min_epi32(vi_id, vj_id);
          v8si vpart_id = _mm256_max_epi32(vi_id, vj_id);
          vpart_id = _mm256_slli_si256(vpart_id, 4);
          v8si vpart_key_id = _mm256_or_si256(vkey_id, vpart_id);

          // shuffle id and store pair data
          v8si idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(shfl_table_[hash]));
          vpart_key_id = _mm256_permutevar8x32_epi32(vpart_key_id, idx);
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(key_partner_particles_[number_of_pairs_]),
                              vpart_key_id);

          number_of_pairs_ += incr;
        }

        for (int32_t k = (num_loop / 4) * 4; k < num_loop; k++) {
          const auto j = pid_of_neigh_cell_loc[k + l + 1];
          RegistInteractPair(q[i], q[j], i, j);
        }
      }
    }
  }

  /* void MakePairListSIMD4x1Swp(const Vec* q,
                         const int32_t particle_number) {
    MakeNeighCellPtclId();
    number_of_pairs_ = 0;
    const v4df vsl2 = _mm256_set_pd(search_length2_,
                                    search_length2_,
                                    search_length2_,
                                    search_length2_);
    for (int32_t icell = 0; icell < all_cell_; icell++) {
      const auto icell_beg = cell_pointer_[icell    ];
      const auto icell_end = cell_pointer_[icell + 1];
      const auto icell_size = icell_end - icell_beg;
      const int32_t* pid_of_neigh_cell_loc = &ptcl_id_of_neigh_cell_[icell][0];
      const int32_t num_of_neigh_cell = ptcl_id_of_neigh_cell_[icell].size();
      for (int32_t l = 0; l < (icell_size / 4) * 4 ; l += 4) {
        const auto i_a = l + icell_beg;
        const v4df vqia = _mm256_load_pd(reinterpret_cast<const double*>(q + i_a));
        const auto i_b = l + icell_beg + 1;
        const v4df vqib = _mm256_load_pd(reinterpret_cast<const double*>(q + i_b));
        const auto i_c = l + icell_beg + 2;
        const v4df vqic = _mm256_load_pd(reinterpret_cast<const double*>(q + i_c));
        const auto i_d = l + icell_beg + 3;
        const v4df vqid = _mm256_load_pd(reinterpret_cast<const double*>(q + i_d));
        v4di vi_id = _mm256_set_epi64x(i_a, i_b, i_c, i_d);

        // transpose 4x4
        v4df tmp0 = _mm256_unpacklo_pd(vqia, vqib);
        v4df tmp1 = _mm256_unpackhi_pd(vqia, vqib);
        v4df tmp2 = _mm256_unpacklo_pd(vqic, vqid);
        v4df tmp3 = _mm256_unpackhi_pd(vqic, vqid);

        v4df vqix_abcd = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
        v4df vqiy_abcd = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
        v4df vqiz_abcd = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);

        // initially distance calculation
        auto j_0 = pid_of_neigh_cell_loc[l + 4];

        v4df vqjx = _mm256_set1_pd(q[j_0].x);
        v4df vqjy = _mm256_set1_pd(q[j_0].y);
        v4df vqjz = _mm256_set1_pd(q[j_0].z);

        v4df dvx = vqjx - vqix_abcd;
        v4df dvy = vqjy - vqiy_abcd;
        v4df dvz = vqjz - vqiz_abcd;

        // norm
        v4df dr2_abcd = dvx * dvx + dvy * dvy + dvz * dvz;

        // dr2 <= search_length2
        v4df dr2_flag = _mm256_cmp_pd(dr2_abcd, vsl2, _CMP_LE_OS);

        // get shfl hash
        int32_t hash_0 = _mm256_movemask_pd(dr2_flag);

        for (int32_t k = l + 5; k < num_of_neigh_cell; k++) {
          if (hash_0 != 0) {
            const int incr = _popcnt32(hash_0);

            v4di vj_id        = _mm256_set_epi64x(j_0, j_0, j_0, j_0);
            v8si vkey_id      = _mm256_min_epi32(vi_id, vj_id);
            v8si vpart_id     = _mm256_max_epi32(vi_id, vj_id);
            vpart_id          = _mm256_slli_si256(vpart_id, 4);
            v8si vpart_key_id = _mm256_or_si256(vkey_id, vpart_id);

            // shuffle id and store pair data
            v8si idx     = _mm256_load_si256(reinterpret_cast<const __m256i*>(shfl_table_[hash_0]));
            vpart_key_id = _mm256_permutevar8x32_epi32(vpart_key_id, idx);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(key_partner_particles_[number_of_pairs_]), vpart_key_id);

            number_of_pairs_ += incr;
          }

          const auto j_1 = pid_of_neigh_cell_loc[k];
          vqjx = _mm256_set1_pd(q[j_1].x);
          vqjy = _mm256_set1_pd(q[j_1].y);
          vqjz = _mm256_set1_pd(q[j_1].z);

          dvx = vqjx - vqix_abcd;
          dvy = vqjy - vqiy_abcd;
          dvz = vqjz - vqiz_abcd;

          // norm
          dr2_abcd = dvx * dvx + dvy * dvy + dvz * dvz;

          // dr2 <= search_length2
          dr2_flag = _mm256_cmp_pd(dr2_abcd, vsl2, _CMP_LE_OS);

          // get shfl hash
          const int32_t hash_1 = _mm256_movemask_pd(dr2_flag);

          // send to next
          j_0 = j_1;
          hash_0 = hash_1;
        }
        if (hash_0 != 0) {
          const int incr = _popcnt32(hash_0);

          v4di vj_id        = _mm256_set_epi64x(j_0, j_0, j_0, j_0);
          v8si vkey_id      = _mm256_min_epi32(vi_id, vj_id);
          v8si vpart_id     = _mm256_max_epi32(vi_id, vj_id);
          vpart_id          = _mm256_slli_si256(vpart_id, 4);
          v8si vpart_key_id = _mm256_or_si256(vkey_id, vpart_id);

          // shuffle id and store pair data
          v8si idx     = _mm256_load_si256(reinterpret_cast<const __m256i*>(shfl_table_[hash_0]));
          vpart_key_id = _mm256_permutevar8x32_epi32(vpart_key_id, idx);
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(key_partner_particles_[number_of_pairs_]), vpart_key_id);

          number_of_pairs_ += incr;
        }

        // remaining pairs
        RegistInteractPair(q[i_a], q[i_a + 1], i_a, i_a + 1);
        RegistInteractPair(q[i_a], q[i_a + 2], i_a, i_a + 2);
        RegistInteractPair(q[i_a], q[i_a + 3], i_a, i_a + 3);
        RegistInteractPair(q[i_b], q[i_b + 1], i_b, i_b + 1);
        RegistInteractPair(q[i_b], q[i_b + 2], i_b, i_b + 2);
        RegistInteractPair(q[i_c], q[i_c + 1], i_c, i_c + 1);
      }

      // remaining i loop
      for (int32_t l = (icell_size / 4) * 4; l < icell_size; l++) {
        const auto i = l + icell_beg;
        const v4df vqi = _mm256_load_pd(reinterpret_cast<const double*>(q + i));
        v4di vi_id = _mm256_set_epi64x(i, i, i, i);
        const auto num_loop = num_of_neigh_cell - (l + 1);
        for (int32_t k = 0; k < (num_loop / 4) * 4; k += 4) {
          const auto ja = pid_of_neigh_cell_loc[k + l + 1];
          const v4df vqja = _mm256_load_pd(reinterpret_cast<const double*>(q + ja));
          v4df dvqa = vqja - vqi;

          const auto jb = pid_of_neigh_cell_loc[k + l + 2];
          const v4df vqjb = _mm256_load_pd(reinterpret_cast<const double*>(q + jb));
          v4df dvqb = vqjb - vqi;

          const auto jc = pid_of_neigh_cell_loc[k + l + 3];
          const v4df vqjc = _mm256_load_pd(reinterpret_cast<const double*>(q + jc));
          v4df dvqc = vqjc - vqi;

          const auto jd = pid_of_neigh_cell_loc[k + l + 4];
          const v4df vqjd = _mm256_load_pd(reinterpret_cast<const double*>(q + jd));
          v4df dvqd = vqjd - vqi;

          // transpose 4x4
          v4df tmp0 = _mm256_unpacklo_pd(dvqa, dvqb);
          v4df tmp1 = _mm256_unpackhi_pd(dvqa, dvqb);
          v4df tmp2 = _mm256_unpacklo_pd(dvqc, dvqd);
          v4df tmp3 = _mm256_unpackhi_pd(dvqc, dvqd);
          dvqa = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
          dvqb = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
          dvqc = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);

          // norm
          v4df dr2_abc = dvqa * dvqa + dvqb * dvqb + dvqc * dvqc;

          // dr2 <= search_length2
          v4df dr2_flag = _mm256_cmp_pd(dr2_abc, vsl2, _CMP_LE_OS);

          // get shfl hash
          const int32_t hash = _mm256_movemask_pd(dr2_flag);

          if (hash == 0) continue;

          const int num = _popcnt32(hash);

          // key_id < part_id
          v4di vj_id = _mm256_set_epi64x(ja, jb, jc, jd);
          v8si vkey_id = _mm256_min_epi32(vi_id, vj_id);
          v8si vpart_id = _mm256_max_epi32(vi_id, vj_id);
          vpart_id = _mm256_slli_si256(vpart_id, 4);
          v8si vpart_key_id = _mm256_or_si256(vkey_id, vpart_id);

          // shuffle id and store pair data
          v8si idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(shfl_table_[hash]));
          vpart_key_id = _mm256_permutevar8x32_epi32(vpart_key_id, idx);
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(key_partner_particles_[number_of_pairs_]), vpart_key_id);

          number_of_pairs_ += num;
        }

        for (int32_t k = (num_loop / 4) * 4; k < num_loop; k++) {
          const auto j = pid_of_neigh_cell_loc[k + l + 1];
          RegistInteractPair(q[i], q[j], i, j);
        }
      }
    }
  }
*/

  void MakeNeighListForEachPtcl(const int32_t particle_number) {
    std::fill(number_of_partners_,
              number_of_partners_ + particle_number,
              0);

    auto k0 = key_partner_particles_[0][KEY];
    auto n0 = number_of_partners_[k0];
    for (int32_t i = 1; i < number_of_pairs_; i++) {
      n0++;
      number_of_partners_[k0] = n0;

      auto k1 = key_partner_particles_[i][KEY];
      auto n1 = number_of_partners_[k1];
      k0 = k1;
      n0 = n1;
    }
    n0++;
    number_of_partners_[k0] = n0;

    neigh_pointer_[0] = neigh_pointer_buf_[0] = 0;
    for (int32_t i = 0; i < particle_number; i++) {
      const auto nei_ptr = neigh_pointer_[i] + number_of_partners_[i];
      neigh_pointer_[i + 1] = nei_ptr;
      neigh_pointer_buf_[i + 1] = nei_ptr;
    }

    auto id_k0 = key_partner_particles_[0][KEY];
    auto id_p0 = key_partner_particles_[0][PARTNER];
    auto next_dst0 = neigh_pointer_buf_[id_k0];
    for (int32_t i = 1; i < number_of_pairs_; i++) {
      // store  incr
      neigh_list_[next_dst0] = id_p0;
      neigh_pointer_buf_[id_k0] = next_dst0 + 1;

      // load next data
      auto id_k1 = key_partner_particles_[i][KEY];
      auto id_p1 = key_partner_particles_[i][PARTNER];
      auto next_dst1 = neigh_pointer_buf_[id_k1];

      id_k0 = id_k1;
      id_p0 = id_p1;
      next_dst0 = next_dst1;
    }
    // store incr
    neigh_list_[next_dst0] = id_p0;
    neigh_pointer_buf_[id_k0] = next_dst0 + 1;

#ifdef DEBUG
    assert(neigh_pointer_[particle_number] == number_of_pairs_);
#endif
  }

public:
  NeighListAVX2(const double search_length,
                const double Lx,
                const double Ly,
                const double Lz) {
    cell_numb_[0] = static_cast<int32_t>(Lx / search_length);
    cell_numb_[1] = static_cast<int32_t>(Ly / search_length);
    cell_numb_[2] = static_cast<int32_t>(Lz / search_length);
    all_cell_ = cell_numb_[0] * cell_numb_[1] * cell_numb_[2];

    cell_leng_.x = Lx / cell_numb_[0];
    cell_leng_.y = Ly / cell_numb_[1];
    cell_leng_.z = Lz / cell_numb_[2];

    search_length_  = search_length;
    search_length2_ = search_length * search_length;
  }
  ~NeighListAVX2() {
    Deallocate();
  }

  // disable copy
  const NeighListAVX2<Vec>& operator = (const NeighListAVX2<Vec>& obj) = delete;
  NeighListAVX2<Vec>(const NeighListAVX2<Vec>& obj) = delete;

  // disable move
  NeighListAVX2<Vec>& operator = (NeighListAVX2<Vec>&& obj) = delete;
  NeighListAVX2<Vec>(NeighListAVX2<Vec>&& obj) = delete;

  void Initialize(const int32_t particle_number) {
    inv_cell_leng_.x = 1.0 / cell_leng_.x;
    inv_cell_leng_.y = 1.0 / cell_leng_.y;
    inv_cell_leng_.z = 1.0 / cell_leng_.z;

    Allocate(particle_number);
    MakeNeighCellId();
    GenShflTable();
  }

  void MakeNeighList(Vec* q,
                     Vec* p,
                     const int32_t particle_number) {
    if (valid_) return;
    MakeCellidOfPtcl(q, particle_number);
    MakeNextDest(particle_number);
    SortPtclData(q, p, particle_number);
#ifdef DEBUG
    CheckSorted(q);
#endif

#ifdef USE1x4
    MakePairListSIMD1x4(q, particle_number);
#elif USE4x1
    MakePairListSIMD4x1(q, particle_number);
#elif SEQ_USE1x4
    MakePairListSIMD1x4SeqStore(q, particle_number);
#elif SEQ_USE4x1
    MakePairListSIMD4x1SeqStore(q, particle_number);
#endif
    MakeNeighListForEachPtcl(particle_number);
  }

  int32_t number_of_pairs() const {
    return number_of_pairs_;
  }

  int32_t* neigh_list() {
    return neigh_list_;
  }

  const int32_t* neigh_list() const {
    return neigh_list_;
  }

  int32_t* neigh_pointer() {
    return neigh_pointer_;
  }

  const int32_t* neigh_pointer() const {
    return neigh_pointer_;
  }

  int32_t* number_of_partners() {
    return number_of_partners_;
  }

  const int32_t* number_of_partners() const {
    return number_of_partners_;
  }
};
