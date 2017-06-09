#pragma once

#include <cassert>
#include <vector>
#include <numeric>

#include "dyalloc2d.hpp"
#include "simd_util.hpp"

template <typename Vec>
class NeighListAVX2 {
  enum : int32_t {X = 0, Y, Z};

  int32_t mesh_size_[3], number_of_mesh_ = -1;
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

  int32_t shfl_table_[16][8] {0};

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
                    const int32_t particle_number) {
    Gather(q, data_buf_, particle_number, sort_buf_);
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

  void MakePairListSIMD1x4SeqStore(const Vec* q,
                                   const int32_t particle_number) {
    MakeNeighMeshPtclId();
    number_of_pairs_ = 0;
    const v4df vsl2 = _mm256_set1_pd(search_length2_);
    for (int32_t imesh = 0; imesh < number_of_mesh_; imesh++) {
      const auto imesh_beg = mesh_index_[imesh];
      const auto imesh_size = mesh_particle_number_[imesh];
      const int32_t* pid_of_neigh_mesh_loc = &ptcl_id_of_neigh_mesh_[imesh][0];
      const int32_t num_of_neigh_mesh = ptcl_id_of_neigh_mesh_[imesh].size();
      for (int32_t l = 0; l < imesh_size; l++) {
        const auto i = ptcl_id_in_mesh_[l + imesh_beg];
        v4df vqix = _mm256_set1_pd(q[i].x);
        v4df vqiy = _mm256_set1_pd(q[i].y);
        v4df vqiz = _mm256_set1_pd(q[i].z);

        const auto num_loop = num_of_neigh_mesh - (l + 1);
        for (int32_t k = 0; k < (num_loop / 4) * 4; k += 4) {
          const auto ja = pid_of_neigh_mesh_loc[k + l + 1];
          const auto jb = pid_of_neigh_mesh_loc[k + l + 2];
          const auto jc = pid_of_neigh_mesh_loc[k + l + 3];
          const auto jd = pid_of_neigh_mesh_loc[k + l + 4];

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
          const auto j = pid_of_neigh_mesh_loc[k + l + 1];
          RegistInteractPair(q[i], q[j], i, j);
        }
      }
    }
  }

  void MakePairListSIMD4x1SeqStore(const Vec* q,
                                   const int32_t particle_number) {
    MakeNeighMeshPtclId();
    number_of_pairs_ = 0;
    const v4df vsl2 = _mm256_set_pd(search_length2_,
                                    search_length2_,
                                    search_length2_,
                                    search_length2_);
    for (int32_t imesh = 0; imesh < number_of_mesh_; imesh++) {
      const auto imesh_beg = mesh_index_[imesh    ];
      const auto imesh_end = mesh_index_[imesh + 1];
      const auto imesh_size = imesh_end - imesh_beg;
      const int32_t* pid_of_neigh_mesh_loc = &ptcl_id_of_neigh_mesh_[imesh][0];
      const int32_t num_of_neigh_mesh = ptcl_id_of_neigh_mesh_[imesh].size();
      for (int32_t l = 0; l < (imesh_size / 4) * 4 ; l += 4) {
        const auto i_a = ptcl_id_in_mesh_[l + imesh_beg    ];
        const auto i_b = ptcl_id_in_mesh_[l + imesh_beg + 1];
        const auto i_c = ptcl_id_in_mesh_[l + imesh_beg + 2];
        const auto i_d = ptcl_id_in_mesh_[l + imesh_beg + 3];

        v4df vqia = _mm256_load_pd(&q[i_a].x);
        v4df vqib = _mm256_load_pd(&q[i_b].x);
        v4df vqic = _mm256_load_pd(&q[i_c].x);
        v4df vqid = _mm256_load_pd(&q[i_d].x);

        v4df vqix, vqiy, vqiz;
        transpose_4x4(vqia, vqib, vqic, vqid, vqix, vqiy, vqiz);
        for (int32_t k = l + 4; k < num_of_neigh_mesh; k++) {
          const auto j = pid_of_neigh_mesh_loc[k];

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
      for (int32_t l = (imesh_size / 4) * 4; l < imesh_size; l++) {
        const auto i = ptcl_id_in_mesh_[l + imesh_beg];
        v4df vqix = _mm256_set1_pd(q[i].x);
        v4df vqiy = _mm256_set1_pd(q[i].y);
        v4df vqiz = _mm256_set1_pd(q[i].z);

        const auto num_loop = num_of_neigh_mesh - (l + 1);
        for (int32_t k = 0; k < (num_loop / 4) * 4; k += 4) {
          const auto ja = pid_of_neigh_mesh_loc[k + l + 1];
          const auto jb = pid_of_neigh_mesh_loc[k + l + 2];
          const auto jc = pid_of_neigh_mesh_loc[k + l + 3];
          const auto jd = pid_of_neigh_mesh_loc[k + l + 4];

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
          const auto j = pid_of_neigh_mesh_loc[k + l + 1];
          RegistInteractPair(q[i], q[j], i, j);
        }
      }
    }
  }

  void MakePairListSIMD1x4(const Vec* q,
                           const int32_t particle_number) {
    MakeNeighMeshPtclId();
    number_of_pairs_ = 0;
    const v4df vsl2 = _mm256_set1_pd(search_length2_);
    for (int32_t imesh = 0; imesh < number_of_mesh_; imesh++) {
      const auto imesh_beg = mesh_index_[imesh];
      const auto imesh_size = mesh_particle_number_[imesh];
      const int32_t* pid_of_neigh_mesh_loc = &ptcl_id_of_neigh_mesh_[imesh][0];
      const int32_t num_of_neigh_mesh = ptcl_id_of_neigh_mesh_[imesh].size();
      for (int32_t l = 0; l < imesh_size; l++) {
        const auto i = ptcl_id_in_mesh_[l + imesh_beg];
        v4df vqix = _mm256_set1_pd(q[i].x);
        v4df vqiy = _mm256_set1_pd(q[i].y);
        v4df vqiz = _mm256_set1_pd(q[i].z);
        v4di vi_id = _mm256_set1_epi64x(i);

        const auto num_loop = num_of_neigh_mesh - (l + 1);
        for (int32_t k = 0; k < (num_loop / 4) * 4; k += 4) {
          const auto ja = pid_of_neigh_mesh_loc[k + l + 1];
          const auto jb = pid_of_neigh_mesh_loc[k + l + 2];
          const auto jc = pid_of_neigh_mesh_loc[k + l + 3];
          const auto jd = pid_of_neigh_mesh_loc[k + l + 4];

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
          const auto j = pid_of_neigh_mesh_loc[k + l + 1];
          RegistInteractPair(q[i], q[j], i, j);
        }
      }
    }
  }

  void MakePairListSIMD4x1(const Vec* q,
                           const int32_t particle_number) {
    MakeNeighMeshPtclId();
    number_of_pairs_ = 0;
    const v4df vsl2 = _mm256_set1_pd(search_length2_);
    for (int32_t imesh = 0; imesh < number_of_mesh_; imesh++) {
      const auto imesh_beg = mesh_index_[imesh    ];
      const auto imesh_end = mesh_index_[imesh + 1];
      const auto imesh_size = imesh_end - imesh_beg;
      const int32_t* pid_of_neigh_mesh_loc = &ptcl_id_of_neigh_mesh_[imesh][0];
      const int32_t num_of_neigh_mesh = ptcl_id_of_neigh_mesh_[imesh].size();
      for (int32_t l = 0; l < (imesh_size / 4) * 4 ; l += 4) {
        const auto i_a = ptcl_id_in_mesh_[l + imesh_beg    ];
        const auto i_b = ptcl_id_in_mesh_[l + imesh_beg + 1];
        const auto i_c = ptcl_id_in_mesh_[l + imesh_beg + 2];
        const auto i_d = ptcl_id_in_mesh_[l + imesh_beg + 3];

        v4df vqia = _mm256_load_pd(&q[i_a].x);
        v4df vqib = _mm256_load_pd(&q[i_b].x);
        v4df vqic = _mm256_load_pd(&q[i_c].x);
        v4df vqid = _mm256_load_pd(&q[i_d].x);

        v4df vqix, vqiy, vqiz;
        transpose_4x4(vqia, vqib, vqic, vqid, vqix, vqiy, vqiz);

        v4di vi_id = _mm256_set_epi64x(i_d, i_c, i_b, i_a);
        for (int32_t k = l + 4; k < num_of_neigh_mesh; k++) {
          const auto j = pid_of_neigh_mesh_loc[k];
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
      for (int32_t l = (imesh_size / 4) * 4; l < imesh_size; l++) {
        const auto i = ptcl_id_in_mesh_[l + imesh_beg];
        v4df vqix = _mm256_set1_pd(q[i].x);
        v4df vqiy = _mm256_set1_pd(q[i].y);
        v4df vqiz = _mm256_set1_pd(q[i].z);
        v4di vi_id = _mm256_set1_epi64x(i);

        const auto num_loop = num_of_neigh_mesh - (l + 1);
        for (int32_t k = 0; k < (num_loop / 4) * 4; k += 4) {
          const auto ja = pid_of_neigh_mesh_loc[k + l + 1];
          const auto jb = pid_of_neigh_mesh_loc[k + l + 2];
          const auto jc = pid_of_neigh_mesh_loc[k + l + 3];
          const auto jd = pid_of_neigh_mesh_loc[k + l + 4];

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
          const auto j = pid_of_neigh_mesh_loc[k + l + 1];
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
  NeighListAVX2(const double search_length,
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
    ims_.x = 1.0 / ms_.x;
    ims_.y = 1.0 / ms_.y;
    ims_.z = 1.0 / ms_.z;

    Allocate(particle_number);
    MakeNeighMeshId();
    GenShflTable();
  }

  void MakeNeighList(Vec* q,
                     const int32_t particle_number) {
    MakeMeshidOfPtcl(q, particle_number);
    MakeNextDest(particle_number);
    // SortPtclData(q, particle_number);
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
