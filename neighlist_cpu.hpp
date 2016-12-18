#pragma once

#include <cassert>
#include <vector>
#include <numeric>

template <typename Vec>
class NeighList {
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

  int32_t *key_particles_ = nullptr, *partner_particles_ = nullptr;
  int32_t *neigh_list_ = nullptr, *number_of_partners_ = nullptr;
  int32_t *neigh_pointer_ = nullptr, *neigh_pointer_buf_ = nullptr;

  std::vector<int32_t>* ptcl_id_of_neigh_cell_ = nullptr;

  Vec *data_buf_ = nullptr;

  enum : int32_t {
    MAX_PARTNERS = 100,
    SORT_FREQ = 50,
    NUM_NEIGH_CELL = 13,
    NUM_PTCL_IN_NEIGH_CELL = 500,
  };

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
    key_particles_         = new int32_t [MAX_PARTNERS * particle_number];
    partner_particles_     = new int32_t [MAX_PARTNERS * particle_number];
    neigh_list_            = new int32_t [MAX_PARTNERS * particle_number];
    number_of_partners_    = new int32_t [particle_number];
    neigh_pointer_         = new int32_t [particle_number + 1];
    neigh_pointer_buf_     = new int32_t [particle_number + 1];
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
    delete [] key_particles_;
    delete [] partner_particles_;
    delete [] neigh_list_;
    delete [] number_of_partners_;
    delete [] neigh_pointer_;
    delete [] neigh_pointer_buf_;
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

  void RegistInteractPair(const Vec& qi,
                          const Vec& qj,
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

  void MakePairListNaive(const Vec* q,
                         const int32_t particle_number) {
    number_of_pairs_ = 0;
    std::fill(number_of_partners_,
              number_of_partners_ + particle_number,
              0);
    for (int32_t icell = 0; icell < all_cell_; icell++) {
      const auto icell_beg = cell_pointer_[icell    ];
      const auto icell_end = cell_pointer_[icell + 1];
      for (int32_t i = icell_beg; i < icell_end; i++) {
        const auto qi = q[i];

        // for different cell
        for (int32_t k = 0; k < NUM_NEIGH_CELL; k++) {
          const auto jcell = neigh_cell_id_[NUM_NEIGH_CELL * icell + k];
          const auto jcell_beg = cell_pointer_[jcell    ];
          const auto jcell_end = cell_pointer_[jcell + 1];
          for (int32_t j = jcell_beg; j < jcell_end; j++) {
            RegistInteractPair(qi, q[j], i, j);
          }
        }

        // for same cell
        for (int32_t j = i + 1; j < icell_end; j++) {
          RegistInteractPair(qi, q[j], i, j);
        }
      }
    }
  }

  void MakePairListFusedLoop(const Vec* q,
                             const int32_t particle_number) {
    MakeNeighCellPtclId();
    number_of_pairs_ = 0;
    std::fill(number_of_partners_,
              number_of_partners_ + particle_number,
              0);
    for (int32_t icell = 0; icell < all_cell_; icell++) {
      const auto icell_beg = cell_pointer_[icell];
      const auto icell_size = number_in_cell_[icell];
      const int32_t* pid_of_neigh_cell_loc = &ptcl_id_of_neigh_cell_[icell][0];
      const int32_t num_of_neigh_cell = ptcl_id_of_neigh_cell_[icell].size();
      for (int32_t l = 0; l < icell_size; l++) {
        const auto i = l + icell_beg;
        const auto qi = q[i];
        for (int32_t k = l + 1; k < num_of_neigh_cell; k++) {
          const auto j = pid_of_neigh_cell_loc[k];
          RegistInteractPair(qi, q[j], i, j);
        }
      }
    }
  }

  void MakePairListFusedLoopSwp(const Vec* q,
                                const int32_t particle_number) {
    MakeNeighCellPtclId();
    number_of_pairs_ = 0;
    std::fill(number_of_partners_,
              number_of_partners_ + particle_number,
              0);
    for (int32_t icell = 0; icell < all_cell_; icell++) {
      const auto icell_beg = cell_pointer_[icell];
      const auto icell_size = number_in_cell_[icell];
      const int32_t* pid_of_neigh_cell_loc = &ptcl_id_of_neigh_cell_[icell][0];
      const int32_t num_of_neigh_cell = ptcl_id_of_neigh_cell_[icell].size();
      for (int32_t l = 0; l < icell_size; l++) {
        const auto i = l + icell_beg;
        const auto qi = q[i];

        auto j_0  = pid_of_neigh_cell_loc[l + 1];
        auto dx = q[j_0].x - qi.x;
        auto dy = q[j_0].y - qi.y;
        auto dz = q[j_0].z - qi.z;
        auto r2_0 = dx * dx + dy * dy + dz * dz;

        for (int32_t k = l + 2; k < num_of_neigh_cell; k++) {
          if (r2_0 > search_length2_) goto NEXT;

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

        NEXT:
          const auto j_1 = pid_of_neigh_cell_loc[k];
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
    neigh_pointer_[0] = neigh_pointer_buf_[0] = 0;
    for (int32_t i = 0; i < particle_number; i++) {
      const auto nei_ptr = neigh_pointer_[i] + number_of_partners_[i];
      neigh_pointer_[i + 1] = nei_ptr;
      neigh_pointer_buf_[i + 1] = nei_ptr;
    }

    for (int32_t i = 0; i < number_of_pairs_; i++) {
      const auto i_id = key_particles_[i];
      neigh_list_[neigh_pointer_buf_[i_id]++] = partner_particles_[i];
    }

#ifdef DEBUG
    assert(neigh_pointer_[particle_number] == number_of_pairs_);
#endif
  }

public:
  NeighList(const double search_length,
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
    inv_cell_leng_.x = 1.0 / cell_leng_.x;
    inv_cell_leng_.y = 1.0 / cell_leng_.y;
    inv_cell_leng_.z = 1.0 / cell_leng_.z;

    Allocate(particle_number);
    MakeNeighCellId();
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
