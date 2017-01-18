#pragma once

#include <cassert>
#include <vector>
#include <numeric>
#include <iomanip>

#include "dyalloc2d.hpp"
#include "simd_util.hpp"

template <typename Vec>
class NeighListAVX512 {
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

  int64_t shfl_table_[256][8] {0};

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
    for (int i = 0; i < 256; i++) {
      auto tbl_id = i;
      int cnt = 0;
      for (int j = 0; j < 8; j++) {
        if (tbl_id & 0x1) shfl_table_[i][cnt++] = j;
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

  void MakePairListSIMD1x8(const Vec* q,
                                    const int32_t particle_number) {
    MakeNeighCellPtclId();
    number_of_pairs_ = 0;
    const v8df vsl2 = _mm512_set1_pd(search_length2_);
    for (int32_t icell = 0; icell < all_cell_; icell++) {
      const auto icell_beg = cell_pointer_[icell];
      const auto icell_size = number_in_cell_[icell];
      const int32_t* pid_of_neigh_cell_loc = &ptcl_id_of_neigh_cell_[icell][0];
      const int32_t num_of_neigh_cell = ptcl_id_of_neigh_cell_[icell].size();
      for (int32_t l = 0; l < icell_size; l++) {
        const auto i = l + icell_beg;
        v8df vqix = _mm512_set1_pd(q[i].x);
        v8df vqiy = _mm512_set1_pd(q[i].y);
        v8df vqiz = _mm512_set1_pd(q[i].z);

        v8di vi_id = _mm512_set1_epi64(i);
        const auto num_loop = num_of_neigh_cell - (l + 1);
        for (int32_t k = 0; k < (num_loop / 8) * 8; k += 8) {
          v8di vj_id
            = _mm512_cvtepi32_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i*>(&pid_of_neigh_cell_loc[k + l + 1])));
          v8di vindex = _mm512_slli_epi64(vj_id, 2);

          v8df vqjx = _mm512_i64gather_pd(vindex, &q[0].x, 8);
          v8df vqjy = _mm512_i64gather_pd(vindex, &q[0].y, 8);
          v8df vqjz = _mm512_i64gather_pd(vindex, &q[0].z, 8);

          v8df dvx = vqjx - vqix;
          v8df dvy = vqjy - vqiy;
          v8df dvz = vqjz - vqiz;

          // norm
          v8df dr2 = dvx * dvx + dvy * dvy + dvz * dvz;

          // dr2 <= search_length2
          __mmask8 dr2_flag = _mm512_cmple_pd_mask(dr2, vsl2);

          if (dr2_flag == 0) continue;

          RegistPairSIMD(dr2_flag, vi_id, vj_id);
        }

        for (int32_t k = (num_loop / 8) * 8; k < num_loop; k++) {
          const auto j = pid_of_neigh_cell_loc[k + l + 1];
          RegistInteractPair(q[i], q[j], i, j);
        }
      }
    }
  }

  void RegistPairSIMD(const __mmask8 dr2_flag,
                      const v8di& vi_id,
                      const v8di& vj_id) {
    const int incr = _popcnt32(dr2_flag);

    v8di vkey_id = _mm512_min_epi32(vi_id, vj_id);
    v8di vpart_id = _mm512_max_epi32(vi_id, vj_id);
    vpart_id = _mm512_slli_epi64(vpart_id, 32);
    v8di vpart_key_id = _mm512_or_si512(vkey_id, vpart_id);

    // store key and partner particle ids
    v8di idx = _mm512_load_si512(shfl_table_[dr2_flag]);
    vpart_key_id = _mm512_permutexvar_epi64(idx, vpart_key_id);

    _mm512_storeu_si512(key_partner_particles_[number_of_pairs_],
                        vpart_key_id);

    // count number of pairs
    number_of_pairs_ += incr;
  }

  void RegistRemainingPair(const v8df& vqix,
                           const v8df& vqiy,
                           const v8df& vqiz,
                           v8df& vqjx,
                           v8df& vqjy,
                           v8df& vqjz,
                           v8di& vi_id,
                           v8di& vj_id,
                           const v8df& vsl2,
                           const int32_t mask) {
    vqjx  = _mm512_rot_rshift_b64(vqjx, 1);
    vqjy  = _mm512_rot_rshift_b64(vqjy, 1);
    vqjz  = _mm512_rot_rshift_b64(vqjz, 1);
    vj_id = _mm512_rot_rshift_b64(vj_id, 1);

    v8df dvx = vqjx - vqix;
    v8df dvy = vqjy - vqiy;
    v8df dvz = vqjz - vqiz;

    // norm
    v8df dr2 = dvx * dvx + dvy * dvy + dvz * dvz;

    // dr2 <= search_length2
    __mmask8 dr2_flag = _mm512_cmple_pd_mask(dr2, vsl2) & mask;

    if (dr2_flag == 0) return;

    RegistPairSIMD(dr2_flag, vi_id, vj_id);
  }

  void MakePairListSIMD8x1(const Vec* q,
                                    const int32_t particle_number) {
    MakeNeighCellPtclId();
    number_of_pairs_ = 0;
    const v8df vsl2 = _mm512_set1_pd(search_length2_);
    for (int32_t icell = 0; icell < all_cell_; icell++) {
      const auto icell_beg  = cell_pointer_[icell    ];
      const auto icell_size = cell_pointer_[icell + 1] - icell_beg;
      const int32_t* pid_of_neigh_cell_loc = &ptcl_id_of_neigh_cell_[icell][0];
      const int32_t num_of_neigh_cell = ptcl_id_of_neigh_cell_[icell].size();
      for (int32_t l = 0; l < (icell_size / 8) * 8 ; l += 8) {
        const auto i_a = l + icell_beg    , i_e = l + icell_beg + 4;
        const auto i_b = l + icell_beg + 1, i_f = l + icell_beg + 5;
        const auto i_c = l + icell_beg + 2, i_g = l + icell_beg + 6;
        const auto i_d = l + icell_beg + 3, i_h = l + icell_beg + 7;

        v8di vi_id  = _mm512_set_epi64(i_h, i_g, i_f, i_e,
                                       i_d, i_c, i_b, i_a);
        v8di vindex = _mm512_slli_epi64(vi_id, 2);
        v8df vqix   = _mm512_i64gather_pd(vindex, &q[0].x, 8);
        v8df vqiy   = _mm512_i64gather_pd(vindex, &q[0].y, 8);
        v8df vqiz   = _mm512_i64gather_pd(vindex, &q[0].z, 8);

        for (int32_t k = l + 8; k < num_of_neigh_cell; k++) {
          const auto j = pid_of_neigh_cell_loc[k];
          v8df vqjx = _mm512_set1_pd(q[j].x);
          v8df vqjy = _mm512_set1_pd(q[j].y);
          v8df vqjz = _mm512_set1_pd(q[j].z);

          v8df dvx = vqjx - vqix;
          v8df dvy = vqjy - vqiy;
          v8df dvz = vqjz - vqiz;

          v8di vj_id = _mm512_set1_epi64(j);

          // norm
          v8df dr2 = dvx * dvx + dvy * dvy + dvz * dvz;

          // dr2 <= search_length2
          __mmask8 dr2_flag = _mm512_cmple_pd_mask(dr2, vsl2);

          if (dr2_flag == 0) continue;

          RegistPairSIMD(dr2_flag, vi_id, vj_id);
        }

        // remaining pairs
        v8df vqjx = vqix, vqjy = vqiy, vqjz = vqiz;
        v8di vj_id = vi_id;
        RegistRemainingPair(vqix, vqiy, vqiz, vqjx, vqjy, vqjz, vi_id, vj_id, vsl2, 0xff);
        RegistRemainingPair(vqix, vqiy, vqiz, vqjx, vqjy, vqjz, vi_id, vj_id, vsl2, 0xff);
        RegistRemainingPair(vqix, vqiy, vqiz, vqjx, vqjy, vqjz, vi_id, vj_id, vsl2, 0xff);
        RegistRemainingPair(vqix, vqiy, vqiz, vqjx, vqjy, vqjz, vi_id, vj_id, vsl2, 0x0f);
      }

      // remaining i loop
      for (int32_t l = (icell_size / 8) * 8; l < icell_size; l++) {
        const auto i = l + icell_beg;
        v8df vqix = _mm512_set1_pd(q[i].x);
        v8df vqiy = _mm512_set1_pd(q[i].y);
        v8df vqiz = _mm512_set1_pd(q[i].z);

        v8di vi_id = _mm512_set1_epi64(i);
        const auto num_loop = num_of_neigh_cell - (l + 1);
        for (int32_t k = 0; k < (num_loop / 8) * 8; k += 8) {
          v8di vj_id
            = _mm512_cvtepi32_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i*>(&pid_of_neigh_cell_loc[k + l + 1])));
          v8di vindex = _mm512_slli_epi64(vj_id, 2);

          v8df vqjx = _mm512_i64gather_pd(vindex, &q[0].x, 8);
          v8df vqjy = _mm512_i64gather_pd(vindex, &q[0].y, 8);
          v8df vqjz = _mm512_i64gather_pd(vindex, &q[0].z, 8);

          v8df dvx = vqjx - vqix;
          v8df dvy = vqjy - vqiy;
          v8df dvz = vqjz - vqiz;

          // norm
          v8df dr2 = dvx * dvx + dvy * dvy + dvz * dvz;

          // dr2 <= search_length2
          __mmask8 dr2_flag = _mm512_cmple_pd_mask(dr2, vsl2);

          if (dr2_flag == 0) continue;

          RegistPairSIMD(dr2_flag, vi_id, vj_id);
        }

        for (int32_t k = (num_loop / 8) * 8; k < num_loop; k++) {
          const auto j = pid_of_neigh_cell_loc[k + l + 1];
          RegistInteractPair(q[i], q[j], i, j);
        }
      }
    }
  }

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
  NeighListAVX512(const double search_length,
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
  ~NeighListAVX512() {
    Deallocate();
  }

  // disable copy
  const NeighListAVX512<Vec>& operator = (const NeighListAVX512<Vec>& obj) = delete;
  NeighListAVX512<Vec>(const NeighListAVX512<Vec>& obj) = delete;

  // disable move
  NeighListAVX512<Vec>& operator = (NeighListAVX512<Vec>&& obj) = delete;
  NeighListAVX512<Vec>(NeighListAVX512<Vec>&& obj) = delete;

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

#ifdef USE8x1
    MakePairListSIMD8x1(q, particle_number);
#elif defined USE1x8
    MakePairListSIMD1x8(q, particle_number);
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
