#pragma once

#include <cassert>
#include <vector>
#include <numeric>

template <typename Vec>
class NeighList {
  enum : int32_t {X = 0, Y, Z};

  int32_t mesh_size_[3], number_of_mesh_ = -1;
  Vec ms_, ims_;
  double search_length_ = 0.0, search_length2_ = 0.0;

  int32_t number_of_pairs_ = 0;

  int32_t* __restrict particle_position_ = nullptr;
  int32_t* __restrict neigh_mesh_id_ = nullptr;
  int32_t* __restrict mesh_particle_number_ = nullptr;
  int32_t* __restrict mesh_index_ = nullptr;
  int32_t* __restrict mesh_index2_ = nullptr;
  int32_t* __restrict ptcl_id_in_mesh_ = nullptr;
  int32_t* __restrict sort_buf_ = nullptr;

  int32_t* __restrict key_particles_ = nullptr;
  int32_t* __restrict partner_particles_ = nullptr;
  int32_t* __restrict sorted_list_ = nullptr;
  int32_t* __restrict number_of_partners_ = nullptr;
  int32_t* __restrict key_pointer_ = nullptr;
  int32_t* __restrict key_pointer2_ = nullptr;

  std::vector<int32_t>* __restrict ptcl_id_of_neigh_mesh_ = nullptr;

  Vec *data_buf_ = nullptr;

  enum : int32_t {
    MAX_PARTNERS = 100,
    NUM_NEIGH_MESH = 13,
    NUM_PTCL_IN_NEIGH_MESH = 500,
  };

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
    key_particles_         = new int32_t [MAX_PARTNERS * particle_number];
    partner_particles_     = new int32_t [MAX_PARTNERS * particle_number];
    sorted_list_           = new int32_t [MAX_PARTNERS * particle_number];
    number_of_partners_    = new int32_t [particle_number];
    key_pointer_           = new int32_t [particle_number + 1];
    key_pointer2_          = new int32_t [particle_number + 1];
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
    delete [] key_particles_;
    delete [] partner_particles_;
    delete [] sorted_list_;
    delete [] number_of_partners_;
    delete [] key_pointer_;
    delete [] key_pointer2_;
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

  void RegistInteractPair(const Vec& __restrict qi,
                          const Vec& __restrict qj,
                          const int32_t index1,
                          const int32_t index2) {
    const auto dx = qj.x - qi.x;
    const auto dy = qj.y - qi.y;
    const auto dz = qj.z - qi.z;
    const auto r2 = dx * dx + dy * dy + dz * dz;
    if (r2 > search_length2_) return;

    int i, j;
    if (index1 < index2) {
      i = index1;
      j = index2;
    } else {
      i = index2;
      j = index1;
    }
    key_particles_[number_of_pairs_] = i;
    partner_particles_[number_of_pairs_] = j;
    number_of_partners_[i]++;
    number_of_pairs_++;
  }

  void MakePairListNaive(const Vec* __restrict q,
                         const int32_t particle_number) {
    number_of_pairs_ = 0;
    std::fill(number_of_partners_,
              number_of_partners_ + particle_number,
              0);
    for (int32_t imesh = 0; imesh < number_of_mesh_; imesh++) {
      const auto imesh_beg = mesh_index_[imesh    ];
      const auto imesh_end = mesh_index_[imesh + 1];
      for (int32_t i = imesh_beg; i < imesh_end; i++) {
        const auto ptcl_id_i = ptcl_id_in_mesh_[i];
        const auto qi = q[ptcl_id_i];

        // for different mesh
        for (int32_t k = 0; k < NUM_NEIGH_MESH; k++) {
          const auto jmesh = neigh_mesh_id_[NUM_NEIGH_MESH * imesh + k];
          const auto jmesh_beg = mesh_index_[jmesh    ];
          const auto jmesh_end = mesh_index_[jmesh + 1];
          for (int32_t j = jmesh_beg; j < jmesh_end; j++) {
            const auto ptcl_id_j = ptcl_id_in_mesh_[j];
            RegistInteractPair(qi, q[ptcl_id_j], ptcl_id_i, ptcl_id_j);
          }
        }

        // for same mesh
        for (int32_t j = i + 1; j < imesh_end; j++) {
          const auto ptcl_id_j = ptcl_id_in_mesh_[j];
          RegistInteractPair(qi, q[ptcl_id_j], ptcl_id_i, ptcl_id_j);
        }
      }
    }
  }

  void MakePairListFusedLoop(const Vec* __restrict q,
                             const int32_t particle_number) {
    MakeNeighMeshPtclId();
    number_of_pairs_ = 0;
    std::fill(number_of_partners_,
              number_of_partners_ + particle_number,
              0);
    for (int32_t imesh = 0; imesh < number_of_mesh_; imesh++) {
      const auto imesh_beg = mesh_index_[imesh];
      const auto imesh_size = mesh_particle_number_[imesh];
      const int32_t* __restrict pid_of_neigh_mesh_loc = &ptcl_id_of_neigh_mesh_[imesh][0];
      const int32_t num_of_neigh_mesh = ptcl_id_of_neigh_mesh_[imesh].size();
      for (int32_t l = 0; l < imesh_size; l++) {
        const auto i = ptcl_id_in_mesh_[l + imesh_beg];
        const auto qi = q[i];
        for (int32_t k = l + 1; k < num_of_neigh_mesh; k++) {
          const auto j = pid_of_neigh_mesh_loc[k];
          RegistInteractPair(qi, q[j], i, j);
        }
      }
    }
  }

  void MakePairListFusedLoopSwp(const Vec* q,
                                const int32_t particle_number) {
    MakeNeighMeshPtclId();
    number_of_pairs_ = 0;
    std::fill(number_of_partners_,
              number_of_partners_ + particle_number,
              0);
    for (int32_t imesh = 0; imesh < number_of_mesh_; imesh++) {
      const auto imesh_beg = mesh_index_[imesh];
      const auto imesh_size = mesh_particle_number_[imesh];
      const int32_t* pid_of_neigh_mesh_loc = &ptcl_id_of_neigh_mesh_[imesh][0];
      const int32_t num_of_neigh_mesh = ptcl_id_of_neigh_mesh_[imesh].size();
      for (int32_t l = 0; l < imesh_size; l++) {
        const auto i  = ptcl_id_in_mesh_[l + imesh_beg];
        const auto qi = q[i];

        auto j_0  = pid_of_neigh_mesh_loc[l + 1];
        auto dx = q[j_0].x - qi.x;
        auto dy = q[j_0].y - qi.y;
        auto dz = q[j_0].z - qi.z;
        auto r2_0 = dx * dx + dy * dy + dz * dz;

        for (int32_t k = l + 2; k < num_of_neigh_mesh; k++) {
          if (r2_0 <= search_length2_) {
            // store load incr
            int id_k, id_p;
            if (i < j_0) {
              id_k = i;
              id_p = j_0;
            } else {
              id_k = j_0;
              id_p = i;
            }
            key_particles_[number_of_pairs_]     = id_k;
            partner_particles_[number_of_pairs_] = id_p;
            number_of_partners_[id_k]++;
            number_of_pairs_++;
          }

          const auto j_1 = pid_of_neigh_mesh_loc[k];
          dx = q[j_1].x - qi.x;
          dy = q[j_1].y - qi.y;
          dz = q[j_1].z - qi.z;
          const auto r2_1 = dx * dx + dy * dy + dz * dz;

          j_0  = j_1;
          r2_0 = r2_1;
        }
        if (r2_0 <= search_length2_) {
          int id_k, id_p;
          if (i < j_0) {
            id_k = i;
            id_p = j_0;
          } else {
            id_k = j_0;
            id_p = i;
          }
          key_particles_[number_of_pairs_]     = id_k;
          partner_particles_[number_of_pairs_] = id_p;
          number_of_partners_[id_k]++;
          number_of_pairs_++;
        }
      }
    }
  }

  void MakeNeighListForEachPtcl(const int32_t particle_number) {
    key_pointer_[0] = key_pointer2_[0] = 0;
    for (int32_t i = 0; i < particle_number; i++) {
      const auto nei_ptr = key_pointer_[i] + number_of_partners_[i];
      key_pointer_[i + 1] = nei_ptr;
      key_pointer2_[i + 1] = nei_ptr;
    }

    for (int32_t i = 0; i < number_of_pairs_; i++) {
      const auto i_id = key_particles_[i];
      sorted_list_[key_pointer2_[i_id]++] = partner_particles_[i];
    }

#ifdef DEBUG
    assert(key_pointer_[particle_number] == number_of_pairs_);
#endif
  }

public:
  NeighList(const double search_length,
            const double Lx,
            const double Ly,
            const double Lz) {
    mesh_size_[0] = static_cast<int32_t>(Lx / search_length);
    mesh_size_[1] = static_cast<int32_t>(Ly / search_length);
    mesh_size_[2] = static_cast<int32_t>(Lz / search_length);
    number_of_mesh_ = mesh_size_[0] * mesh_size_[1] * mesh_size_[2];

    ms_.x = Lx / mesh_size_[0];
    ms_.y = Ly / mesh_size_[1];
    ms_.z = Lz / mesh_size_[2];

    search_length_  = search_length;
    search_length2_ = search_length * search_length;
  }
  ~NeighList() {
    Deallocate();
  }

  // disable copy
  const NeighList<Vec>& operator = (const NeighList<Vec>& obj) = delete;
  NeighList<Vec>(const NeighList<Vec>& obj) = delete;

  // disable move
  NeighList<Vec>& operator = (NeighList<Vec>&& obj) = delete;
  NeighList<Vec>(NeighList<Vec>&& obj) = delete;

  void Initialize(const int32_t particle_number) {
    ims_.x = 1.0 / ms_.x;
    ims_.y = 1.0 / ms_.y;
    ims_.z = 1.0 / ms_.z;

    Allocate(particle_number);
    MakeNeighMeshId();
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

#ifdef WITHOUT_LOOP_FUSION
    MakePairListNaive(q, particle_number);
#elif defined LOOP_FUSION
    MakePairListFusedLoop(q, particle_number);
#elif defined LOOP_FUSION_SWP
    MakePairListFusedLoopSwp(q, particle_number);
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
