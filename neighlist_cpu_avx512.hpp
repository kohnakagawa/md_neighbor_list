#pragma once

#include <cassert>
#include <vector>
#include <numeric>
#include <iomanip>

#include "dyalloc2d.hpp"
#include "simd_util.hpp"

template <typename Vec>
class NeighListAVX512 {
  enum : int32_t {X = 0, Y, Z};

  int32_t mesh_size_[3] {0}, number_of_mesh_ = -1;
  Vec ms_, ims_;
  double search_length_  = 0.0, search_length2_ = 0.0;

  int32_t number_of_pairs_ = 0;

  int32_t *particle_position_ = nullptr, *neigh_mesh_id_ = nullptr;
  int32_t *mesh_particle_number_ = nullptr;
  int32_t *mesh_index_ = nullptr, *mesh_index2_ = nullptr;
  int32_t *ptcl_id_in_mesh_ = nullptr;
  int32_t *sort_buf_ = nullptr;

  int32_t *sorted_list_ = nullptr, *number_of_partners_ = nullptr;
  int32_t *key_pointer_ = nullptr, *key_pointer2_ = nullptr;

  int32_t **key_partner_particles_ = nullptr;

  std::vector<int32_t>* ptcl_id_of_neigh_mesh_ = nullptr;

  Vec *data_buf_ = nullptr;

  int64_t shfl_table_[256][8] {0};

  enum : int32_t {
    MAX_PARTNERS = 100,
    NUM_NEIGH_MESH = 13,
    NUM_PTCL_IN_NEIGH_MESH = 500,
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
    const auto ret = idx[X] + (idx[Y] + idx[Z] * mesh_size_[Y]) * mesh_size_[X];
#ifdef DEBUG
    assert(ret >= 0);
    assert(ret < number_of_mesh_);
#endif
    return ret;
  }

  int32_t GenHash(const Vec& q) const {
    int32_t idx[] = {
      static_cast<int32_t>(q.x * ims_.x),
      static_cast<int32_t>(q.y * ims_.y),
      static_cast<int32_t>(q.z * ims_.z)
    };
    ApplyPBC(idx);
    return GenHash(idx);
  }

  void ApplyPBC(int32_t* idx) const {
    for (int i = 0; i < 3; i++) {
      if (idx[i] < 0) idx[i] += mesh_size_[i];
      if (idx[i] >= mesh_size_[i]) idx[i] -= mesh_size_[i];
    }
  }

  void Allocate(const int32_t particle_number) {
    particle_position_     = new int32_t [particle_number];
    neigh_mesh_id_         = new int32_t [NUM_NEIGH_MESH * number_of_mesh_];
    mesh_particle_number_  = new int32_t [number_of_mesh_];
    mesh_index_            = new int32_t [number_of_mesh_ + 1];
    mesh_index2_           = new int32_t [number_of_mesh_ + 1];
    ptcl_id_in_mesh_       = new int32_t [particle_number];
    sort_buf_              = new int32_t [particle_number];
    sorted_list_           = new int32_t [MAX_PARTNERS * particle_number];
    number_of_partners_    = new int32_t [particle_number];
    key_pointer_           = new int32_t [particle_number + 1];
    key_pointer2_          = new int32_t [particle_number + 1];
    allocate2D_aligend<int32_t, 32>(MAX_PARTNERS * particle_number, 2, key_partner_particles_);
    ptcl_id_of_neigh_mesh_ = new std::vector<int32_t> [number_of_mesh_];
    data_buf_              = new Vec [particle_number];
    for (int32_t i = 0; i < number_of_mesh_; i++) {
      ptcl_id_of_neigh_mesh_[i].resize(NUM_PTCL_IN_NEIGH_MESH);
    }
  }

  void Deallocate() {
    delete [] particle_position_;
    delete [] neigh_mesh_id_;
    delete [] mesh_particle_number_;
    delete [] mesh_index_;
    delete [] mesh_index2_;
    delete [] ptcl_id_in_mesh_;
    delete [] sort_buf_;
    delete [] sorted_list_;
    delete [] number_of_partners_;
    delete [] key_pointer_;
    delete [] key_pointer2_;
    deallocate2D_aligend(key_partner_particles_);
    delete [] ptcl_id_of_neigh_mesh_;
    delete [] data_buf_;
  }

  void MakeNeighMeshId() {
    int32_t imesh_id = 0;
    for (int32_t iz = 0; iz < mesh_size_[Z]; iz++)
      for (int32_t iy = 0; iy < mesh_size_[Y]; iy++)
        for (int32_t ix = 0; ix < mesh_size_[X]; ix++) {
          int32_t jmesh_id = 0;
          for (int32_t jz = -1; jz < 2; jz++)
            for (int32_t jy = -1; jy < 2; jy++)
              for (int32_t jx = -1; jx < 2; jx++) {
                int32_t idx[] = { ix + jx, iy + jy, iz + jz };
                ApplyPBC(idx);
                neigh_mesh_id_[NUM_NEIGH_MESH * imesh_id + jmesh_id] = GenHash(idx);
                jmesh_id++;
                if (jmesh_id == NUM_NEIGH_MESH) goto OUT;
              }
        OUT:
          imesh_id++;
        }
#ifdef DEBUG
    assert(imesh_id == number_of_mesh_);
    for (int i = 0; i < number_of_mesh_ * NUM_NEIGH_MESH; i++) {
      assert(neigh_mesh_id_[i] >= 0);
      assert(neigh_mesh_id_[i] < number_of_mesh_);
    }
#endif
  }

  void MakeMeshidOfPtcl(const Vec* q,
                        const int32_t particle_number) {
    std::fill(mesh_particle_number_,
              mesh_particle_number_ + number_of_mesh_,
              0);
    for (int32_t i = 0; i < particle_number; i++) {
      const auto hash = GenHash(q[i]);
      mesh_particle_number_[hash]++;
      particle_position_[i] = hash;
    }
  }

  void MakeNextDest(const int32_t particle_number) {
    mesh_index_[0] = mesh_index2_[0] = 0;
    for (int32_t i = 0; i < number_of_mesh_; i++) {
      const auto g_ptr = mesh_index_[i] + mesh_particle_number_[i];
      mesh_index_[i + 1] = g_ptr;
      mesh_index2_[i + 1] = g_ptr;
    }

    for (int32_t i = 0; i < particle_number; i++) {
      const auto hash = particle_position_[i];
      const auto dst = mesh_index2_[hash];
      sort_buf_[i] = dst;
      ptcl_id_in_mesh_[dst] = i;
      mesh_index2_[hash]++;
    }

#ifdef DEBUG
    assert(mesh_index_[number_of_mesh_] == particle_number);
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
    Gather(q, data_buf_, particle_number, sort_buf_);
    Gather(p, data_buf_, particle_number, sort_buf_);
    std::iota(ptcl_id_in_mesh_, ptcl_id_in_mesh_ + particle_number, 0);
  }

  void CheckSorted(const Vec* q) const {
    for (int32_t mesh = 0; mesh < number_of_mesh_; mesh++) {
      const auto beg = mesh_index_[mesh    ];
      const auto end = mesh_index_[mesh + 1];
      for (int32_t i = beg; i < end; i++) {
        const auto hash = GenHash(q[i]);
        if (hash != mesh) {
          std::cerr << "particle data is not correctly sorted.\n";
          std::exit(1);
        }
      }
    }
  }

  void MakeNeighMeshPtclId() {
    for (int32_t imesh = 0; imesh < number_of_mesh_; imesh++) {
      ptcl_id_of_neigh_mesh_[imesh].clear();
      const auto imesh_beg = mesh_index_[imesh    ];
      const auto imesh_end = mesh_index_[imesh + 1];
      ptcl_id_of_neigh_mesh_[imesh].insert(ptcl_id_of_neigh_mesh_[imesh].end(),
                                           &ptcl_id_in_mesh_[imesh_beg],
                                           &ptcl_id_in_mesh_[imesh_end]);
      for (int32_t k = 0; k < NUM_NEIGH_MESH; k++) {
        const auto jmesh = neigh_mesh_id_[NUM_NEIGH_MESH * imesh + k];
        const auto jmesh_beg = mesh_index_[jmesh    ];
        const auto jmesh_end = mesh_index_[jmesh + 1];
        ptcl_id_of_neigh_mesh_[imesh].insert(ptcl_id_of_neigh_mesh_[imesh].end(),
                                             &ptcl_id_in_mesh_[jmesh_beg],
                                             &ptcl_id_in_mesh_[jmesh_end]);
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
    MakeNeighMeshPtclId();
    number_of_pairs_ = 0;
    const v8df vsl2 = _mm512_set1_pd(search_length2_);
    for (int32_t imesh = 0; imesh < number_of_mesh_; imesh++) {
      const auto imesh_beg = mesh_index_[imesh];
      const auto imesh_size = mesh_particle_number_[imesh];
      const int32_t* pid_of_neigh_mesh_loc = &ptcl_id_of_neigh_mesh_[imesh][0];
      const int32_t num_of_neigh_mesh = ptcl_id_of_neigh_mesh_[imesh].size();
      for (int32_t l = 0; l < imesh_size; l++) {
        const auto i = ptcl_id_in_mesh_[l + imesh_beg];
        v8df vqix = _mm512_set1_pd(q[i].x);
        v8df vqiy = _mm512_set1_pd(q[i].y);
        v8df vqiz = _mm512_set1_pd(q[i].z);

        v8di vi_id = _mm512_set1_epi64(i);
        const auto num_loop = num_of_neigh_mesh - (l + 1);
        for (int32_t k = 0; k < (num_loop / 8) * 8; k += 8) {
          v8di vj_id
            = _mm512_cvtepi32_epi64(_mm256_lddqu_si256(reinterpret_cast<const __m256i*>(&pid_of_neigh_mesh_loc[k + l + 1])));
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

        for (int32_t k = (num_loop / 8) * 8; k < num_loop; k++) {
          const auto j = pid_of_neigh_mesh_loc[k + l + 1];
          RegistInteractPair(q[i], q[j], i, j);
        }
      }
    }
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
    MakeNeighMeshPtclId();
    number_of_pairs_ = 0;

    const v8df vsl2 = _mm512_set1_pd(search_length2_);
    for (int32_t imesh = 0; imesh < number_of_mesh_; imesh++) {
      const auto imesh_beg  = mesh_index_[imesh    ];
      const auto imesh_size = mesh_index_[imesh + 1] - imesh_beg;
      const int32_t* pid_of_neigh_mesh_loc = &ptcl_id_of_neigh_mesh_[imesh][0];
      const int32_t  num_of_neigh_mesh     = ptcl_id_of_neigh_mesh_[imesh].size();
      for (int32_t l = 0; l < (imesh_size / 8) * 8 ; l += 8) {
        const auto i_a = ptcl_id_in_mesh_[l + imesh_beg    ];
        const auto i_b = ptcl_id_in_mesh_[l + imesh_beg + 1];
        const auto i_c = ptcl_id_in_mesh_[l + imesh_beg + 2];
        const auto i_d = ptcl_id_in_mesh_[l + imesh_beg + 3];
        const auto i_e = ptcl_id_in_mesh_[l + imesh_beg + 4];
        const auto i_f = ptcl_id_in_mesh_[l + imesh_beg + 5];
        const auto i_g = ptcl_id_in_mesh_[l + imesh_beg + 6];
        const auto i_h = ptcl_id_in_mesh_[l + imesh_beg + 7];

        v8di vi_id  = _mm512_set_epi64(i_h, i_g, i_f, i_e,
                                       i_d, i_c, i_b, i_a);
        v8di vindex = _mm512_slli_epi64(vi_id, 2);
        v8df vqix   = _mm512_i64gather_pd(vindex, &q[0].x, 8);
        v8df vqiy   = _mm512_i64gather_pd(vindex, &q[0].y, 8);
        v8df vqiz   = _mm512_i64gather_pd(vindex, &q[0].z, 8);

        for (int32_t k = l + 8; k < num_of_neigh_mesh; k++) {
          const auto j = pid_of_neigh_mesh_loc[k];
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

        // remaining pairs
        for (int k = 0; k < 7; k++) {
          const auto i_k = ptcl_id_in_mesh_[l + imesh_beg + k];
          for (int j = k + 1; j < 8; j++) {
            const auto i_j = ptcl_id_in_mesh_[l + imesh_beg + j];
            RegistInteractPair(q[i_k], q[i_j], i_k, i_j);
          }
        }
      }

      // remaining i loop
      for (int32_t l = (imesh_size / 8) * 8; l < imesh_size; l++) {
        const auto i = ptcl_id_in_mesh_[l + imesh_beg];
        const auto qi = q[i];
        for (int32_t k = l + 1; k < num_of_neigh_mesh; k++) {
          const auto j = pid_of_neigh_mesh_loc[k];
          RegistInteractPair(qi, q[j], i, j);
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

    key_pointer_[0] = key_pointer2_[0] = 0;
    for (int32_t i = 0; i < particle_number; i++) {
      const auto nei_ptr = key_pointer_[i] + number_of_partners_[i];
      key_pointer_[i + 1] = nei_ptr;
      key_pointer2_[i + 1] = nei_ptr;
    }

    auto id_k0 = key_partner_particles_[0][KEY];
    auto id_p0 = key_partner_particles_[0][PARTNER];
    auto next_dst0 = key_pointer2_[id_k0];
    for (int32_t i = 1; i < number_of_pairs_; i++) {
      // store  incr
      sorted_list_[next_dst0] = id_p0;
      key_pointer2_[id_k0] = next_dst0 + 1;

      // load next data
      auto id_k1 = key_partner_particles_[i][KEY];
      auto id_p1 = key_partner_particles_[i][PARTNER];
      auto next_dst1 = key_pointer2_[id_k1];

      id_k0 = id_k1;
      id_p0 = id_p1;
      next_dst0 = next_dst1;
    }
    // store incr
    sorted_list_[next_dst0] = id_p0;
    key_pointer2_[id_k0] = next_dst0 + 1;

#ifdef DEBUG
    assert(key_pointer_[particle_number] == number_of_pairs_);
#endif
  }

public:
  NeighListAVX512(const double search_length,
                const double Lx,
                const double Ly,
                const double Lz) {
    mesh_size_[X] = static_cast<int32_t>(Lx / search_length);
    mesh_size_[Y] = static_cast<int32_t>(Ly / search_length);
    mesh_size_[Z] = static_cast<int32_t>(Lz / search_length);
    number_of_mesh_ = mesh_size_[X] * mesh_size_[Y] * mesh_size_[Z];

    ms_.x = Lx / mesh_size_[X];
    ms_.y = Ly / mesh_size_[Y];
    ms_.z = Lz / mesh_size_[Z];

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
    ims_.x = 1.0 / ms_.x;
    ims_.y = 1.0 / ms_.y;
    ims_.z = 1.0 / ms_.z;

    Allocate(particle_number);
    MakeNeighMeshId();
    GenShflTable();
  }

  void MakeNeighList(Vec* q,
                     Vec* p,
                     const int32_t particle_number) {
    MakeMeshidOfPtcl(q, particle_number);
    MakeNextDest(particle_number);
    // SortPtclData(q, p, particle_number);
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

  int32_t* sorted_list() {
    return sorted_list_;
  }

  const int32_t* sorted_list() const {
    return sorted_list_;
  }

  int32_t* key_pointer() {
    return key_pointer_;
  }

  const int32_t* key_pointer() const {
    return key_pointer_;
  }

  int32_t* number_of_partners() {
    return number_of_partners_;
  }

  const int32_t* number_of_partners() const {
    return number_of_partners_;
  }
};
