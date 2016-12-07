#pragma once

#include <cassert>

template <typename Vec>
class NeighList {
  bool valid_ = false;
  int32_t grid_numb_[3], all_grid_ = -1;
  Vec grid_leng_, inv_grid_leng_;
  double search_length_  = 0.0, search_length2_ = 0.0;

  int32_t number_of_pairs_ = 0;

  int32_t *grid_id_of_ptcl_ = nullptr, *neigh_grid_id_ = nullptr;
  int32_t *number_in_grid_ = nullptr;
  int32_t *grid_pointer_ = nullptr, *grid_pointer_buf_ = nullptr;
  int32_t *ptcl_id_in_grid_ = nullptr;
  int32_t *next_dst_ = nullptr;

  int32_t *key_particles_ = nullptr, *partner_particles_ = nullptr;
  int32_t *neigh_list_ = nullptr, *number_of_partners_ = nullptr;
  int32_t *neigh_pointer_ = nullptr, *neigh_pointer_buf_ = nullptr;

  Vec *data_buf_ = nullptr;

  enum : int32_t {
    MAX_PARTNERS = 100,
    SORT_FREQ = 50,
    NUM_NEIGH_GRID = 13,
  };

  int32_t GenHash(const int32_t* idx) const {
    const auto ret = idx[0] + (idx[1] + idx[2] * grid_numb_[1]) * grid_numb_[0];
#ifdef DEBUG
    assert(ret >= 0);
    assert(ret < all_grid_);
#endif
    return ret;
  }

  int32_t GenHash(const Vec& q) const {
    int32_t idx[] = {
      static_cast<int32_t>(q.x * inv_grid_leng_.x),
      static_cast<int32_t>(q.y * inv_grid_leng_.y),
      static_cast<int32_t>(q.z * inv_grid_leng_.z)
    };
    ApplyPBC(idx);
    return GenHash(idx);
  }

  void ApplyPBC(int32_t* idx) const {
    for (int i = 0; i < 3; i++) {
      if (idx[i] < 0) idx[i] += grid_numb_[i];
      if (idx[i] >= grid_numb_[i]) idx[i] -= grid_numb_[i];
    }
  }

  void Allocate(const int32_t particle_number) {
    grid_id_of_ptcl_    = new int32_t [particle_number];
    neigh_grid_id_      = new int32_t [NUM_NEIGH_GRID * all_grid_];
    number_in_grid_     = new int32_t [all_grid_];
    grid_pointer_       = new int32_t [all_grid_ + 1];
    grid_pointer_buf_   = new int32_t [all_grid_ + 1];
    ptcl_id_in_grid_    = new int32_t [particle_number];
    next_dst_           = new int32_t [particle_number];
    key_particles_      = new int32_t [MAX_PARTNERS * particle_number];
    partner_particles_  = new int32_t [MAX_PARTNERS * particle_number];
    neigh_list_         = new int32_t [MAX_PARTNERS * particle_number];
    number_of_partners_ = new int32_t [particle_number];
    neigh_pointer_      = new int32_t [particle_number + 1];
    neigh_pointer_buf_  = new int32_t [particle_number + 1];

    data_buf_           = new Vec [particle_number];
  }

  void Deallocate() {
    delete [] grid_id_of_ptcl_;
    delete [] neigh_grid_id_;
    delete [] number_in_grid_;
    delete [] grid_pointer_;
    delete [] grid_pointer_buf_;
    delete [] ptcl_id_in_grid_;
    delete [] next_dst_;
    delete [] key_particles_;
    delete [] partner_particles_;
    delete [] neigh_list_;
    delete [] number_of_partners_;
    delete [] neigh_pointer_;
    delete [] neigh_pointer_buf_;

    delete [] data_buf_;
  }

  void MakeNeighGridId() {
    int32_t icell_id = 0;
    for (int32_t iz = 0; iz < grid_numb_[2]; iz++)
      for (int32_t iy = 0; iy < grid_numb_[1]; iy++)
        for (int32_t ix = 0; ix < grid_numb_[0]; ix++) {
          int32_t jcell_id = 0;
          for (int32_t jz = -1; jz < 2; jz++)
            for (int32_t jy = -1; jy < 2; jy++)
              for (int32_t jx = -1; jx < 2; jx++) {
                int32_t idx[] = { ix + jx, iy + jy, iz + jz };
                ApplyPBC(idx);
                neigh_grid_id_[NUM_NEIGH_GRID * icell_id + jcell_id] = GenHash(idx);
                jcell_id++;
                if (jcell_id == NUM_NEIGH_GRID) goto OUT;
              }
        OUT:
          icell_id++;
        }
#ifdef DEBUG
    assert(icell_id == all_grid_);
    for (int i = 0; i < all_grid_ * NUM_NEIGH_GRID; i++) {
      assert(neigh_grid_id_[i] >= 0);
      assert(neigh_grid_id_[i] < all_grid_);
    }
#endif
  }

  void MakeGrididOfPtcl(const Vec* q,
                        const int32_t particle_number) {
    std::fill(number_in_grid_,
              number_in_grid_ + all_grid_,
              0);
    for (int32_t i = 0; i < particle_number; i++) {
      const auto hash = GenHash(q[i]);
      number_in_grid_[hash]++;
      grid_id_of_ptcl_[i] = hash;
    }
  }

  void MakeNextDest(const int32_t particle_number) {
    grid_pointer_[0] = grid_pointer_buf_[0] = 0;
    for (int32_t i = 0; i < all_grid_; i++) {
      const auto g_ptr = grid_pointer_[i] + number_in_grid_[i];
      grid_pointer_[i + 1] = g_ptr;
      grid_pointer_buf_[i + 1] = g_ptr;
    }

    for (int32_t i = 0; i < particle_number; i++) {
      const auto hash = grid_id_of_ptcl_[i];
      const auto dst = grid_pointer_buf_[hash];
      next_dst_[i] = dst;
      ptcl_id_in_grid_[dst] = i;
      grid_pointer_buf_[hash]++;
    }

#ifdef DEBUG
    assert(grid_pointer_[all_grid_] == particle_number);
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
    // for (int32_t i = 0; i < particle_number; i++) ptcl_id_in_grid_[i] = i;
  }

  void CheckSorted(const Vec* q) const {
    for (int32_t grid = 0; grid < all_grid_; grid++) {
      const auto beg = grid_pointer_[grid    ];
      const auto end = grid_pointer_[grid + 1];
      for (int32_t i = beg; i < end; i++) {
        const auto hash = GenHash(q[i]);
        if (hash != grid) {
          std::cerr << "particle data is not correctly sorted.\n";
          std::exit(1);
        }
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

  void MakePairList(const Vec* q,
                    const int32_t particle_number) {
    number_of_pairs_ = 0;
    std::fill(number_of_partners_,
              number_of_partners_ + particle_number,
              0);
    for (int32_t igrid = 0; igrid < all_grid_; igrid++) {
      const auto igrid_beg = grid_pointer_[igrid    ];
      const auto igrid_end = grid_pointer_[igrid + 1];
      for (int32_t i = igrid_beg; i < igrid_end; i++) {
        const auto qi = q[i];

        // for different cell
        for (int32_t k = 0; k < NUM_NEIGH_GRID; k++) {
          const auto jgrid = neigh_grid_id_[NUM_NEIGH_GRID * igrid + k];
          const auto jgrid_beg = grid_pointer_[jgrid    ];
          const auto jgrid_end = grid_pointer_[jgrid + 1];
          for (int32_t j = jgrid_beg; j < jgrid_end; j++) {
            const auto qj = q[j];
            RegistInteractPair(qi, qj, i, j);
          }
        }

        // for same cell
        for (int32_t j = i + 1; j < igrid_end; j++) {
          const auto qj = q[j];
          RegistInteractPair(qi, qj, i, j);
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
    grid_numb_[0] = static_cast<int32_t>(Lx / search_length);
    grid_numb_[1] = static_cast<int32_t>(Ly / search_length);
    grid_numb_[2] = static_cast<int32_t>(Lz / search_length);
    all_grid_ = grid_numb_[0] * grid_numb_[1] * grid_numb_[2];

    grid_leng_.x = Lx / grid_numb_[0];
    grid_leng_.y = Ly / grid_numb_[1];
    grid_leng_.z = Lz / grid_numb_[2];

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
    inv_grid_leng_.x = 1.0 / grid_leng_.x;
    inv_grid_leng_.y = 1.0 / grid_leng_.y;
    inv_grid_leng_.z = 1.0 / grid_leng_.z;

    Allocate(particle_number);
    MakeNeighGridId();
  }

  void MakeNeighList(Vec* q,
                     Vec* p,
                     const int32_t particle_number) {
    if (valid_) return;
    MakeGrididOfPtcl(q, particle_number);
    MakeNextDest(particle_number);
    SortPtclData(q, p, particle_number);
#ifdef DEBUG
    CheckSorted(q);
#endif
    MakePairList(q, particle_number);
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
};
