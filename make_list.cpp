#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
#include "neighlist_cpu.hpp"

typedef double Dtype;
// typedef float Dtype;

const Dtype density = 1.0;
const int N = 400000; // maximum buffer size
const int MAX_NEIGH_N = 200;
const int LOOP = 100;
const Dtype L = 50.0;

const Dtype SEARCH_LENGTH = 3.3;
const Dtype SEARCH_LENGTH2 = SEARCH_LENGTH * SEARCH_LENGTH;

struct Vec {
#ifdef USE_VEC4
  Dtype x, y, z, w;
#else
  Dtype x, y, z;
#endif
};

template <typename Vec>
void add_particle(const Dtype x,
                  const Dtype y,
                  const Dtype z,
                  Vec* q,
                  int& particle_number) {
  static std::mt19937 mt(2);
  std::uniform_real_distribution<Dtype> ud(0.0, 0.1);
  q[particle_number].x = x + ud(mt);
  q[particle_number].y = y + ud(mt);
  q[particle_number].z = z + ud(mt);
  particle_number++;
}

template <typename Vec>
void init(Vec* q,
          int& particle_number) {
  const Dtype s = 1.0 / std::pow(density * 0.25, 1.0 / 3.0);
  const Dtype hs = s * 0.5;
  const int sx = static_cast<int>(L / s);
  const int sy = static_cast<int>(L / s);
  const int sz = static_cast<int>(L / s);
  for (int iz = 0; iz < sz; iz++) {
    for (int iy = 0; iy < sy; iy++) {
      for (int ix = 0; ix < sx; ix++) {
        const Dtype x = ix*s;
        const Dtype y = iy*s;
        const Dtype z = iz*s;
        add_particle(x     ,y   ,z, q, particle_number);
        add_particle(x     ,y+hs,z+hs, q, particle_number);
        add_particle(x+hs  ,y   ,z+hs, q, particle_number);
        add_particle(x+hs  ,y+hs,z, q, particle_number);
      }
    }
  }

  if (particle_number > N) {
    std::cerr << "particle_number " << particle_number << " exceeds maximum buffer size " << N << "\n";
    std::exit(EXIT_FAILURE);
  }
}

template <typename Vec>
void make_neighlist_bruteforce(const Vec* q,
                               const int32_t particle_number,
                               std::vector<int>& neighlist,
                               std::vector<int>& number_of_partners) {
  for (int i = 0; i < particle_number; i++) {
    int n_neigh = 0;
    for (int j = i + 1; j < particle_number; j++) {
      const auto qi = q[i];
      const auto qj = q[j];
      const auto drx = qj.x - qi.x;
      const auto dry = qj.y - qi.y;
      const auto drz = qj.z - qi.z;
      const auto dr2 = drx * drx + dry * dry + drz * drz;
      if (dr2 > SEARCH_LENGTH2) continue;
      neighlist[particle_number * n_neigh + i] = j;
      n_neigh++;
    }
    number_of_partners[i] = n_neigh;
  }
}

void make_sorted_list(const int32_t particle_number,
                     const std::vector<int>& neighlist,
                     const std::vector<int>& number_of_partners,
                     std::vector<int>& sorted_list,
                     std::vector<int>& neigh_pointer,
                     int32_t& number_of_pairs) {
  neigh_pointer[0] = 0;
  for (int i = 0; i < particle_number; i++) {
    neigh_pointer[i + 1] = neigh_pointer[i] + number_of_partners[i];
  }

  number_of_pairs = 0;
  for (int i = 0; i < particle_number; i++) {
    for (int k = 0; k < number_of_partners[i]; k++) {
      sorted_list[number_of_pairs++] = neighlist[particle_number * k + i];
    }
  }
}

void sort_neighlist(int* neigh_list,
                    const int* pointer,
                    const int32_t particle_number) {
  for (int i = 0; i < particle_number; i++) {
    const auto beg = pointer[i];
    const auto end = pointer[i + 1];
    std::sort(&neigh_list[beg], &neigh_list[end]);
  }
}

#define PRINT_WITH_TAG(ost, val) ost << #val << " " << val << "\n"

int main(int argc, char* argv[]) {
  // construct particles
  Vec* __restrict q = new Vec [N];
  Vec* __restrict p = new Vec [N];
  int particle_number = 0;
  init(q, particle_number);
  for (int i = 0; i < particle_number; i++) {
    p[i].x = p[i].y = p[i].z = 0.0;
  }

  // make neighbor list
  NeighList<Vec> nlist(SEARCH_LENGTH, L, L, L);
  nlist.Initialize(particle_number);
  const auto beg = std::chrono::system_clock::now();
  for (int i = 0; i < LOOP; i++) {
    nlist.MakeNeighList(q, p, particle_number);
  }
  const auto end = std::chrono::system_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
  std::cout << "# of particles " << particle_number
            << " " << elapsed << "[ms]\n";
  const auto number_of_pairs = nlist.number_of_pairs();
  int32_t* neigh_list  = nlist.neigh_list();
  int32_t* neigh_pointer = nlist.neigh_pointer();

  // reference
  std::vector<int> neigh_list_buf, number_of_partners, neigh_list_ref, neigh_pointer_ref;
  int32_t number_of_pairs_ref = 0;
  neigh_list_buf.resize(particle_number * MAX_NEIGH_N, -1);
  number_of_partners.resize(particle_number, 0);
  neigh_list_ref.resize(particle_number * MAX_NEIGH_N, -1);
  neigh_pointer_ref.resize(particle_number + 1, 0);
  make_neighlist_bruteforce(q,
                            particle_number,
                            neigh_list_buf,
                            number_of_partners);
  make_sorted_list(particle_number,
                   neigh_list_buf,
                   number_of_partners,
                   neigh_list_ref,
                   neigh_pointer_ref,
                   number_of_pairs_ref);

  // check the correctness of neighbor list
  if (number_of_pairs_ref != number_of_pairs) {
    std::cerr << "TEST fail\n";
    PRINT_WITH_TAG(std::cerr, number_of_pairs_ref);
    PRINT_WITH_TAG(std::cerr, number_of_pairs);
    std::exit(1);
  }

  for (int i = 0; i < particle_number + 1; i++) {
    if (neigh_pointer_ref[i] != neigh_pointer[i]) {
      std::cerr << "TEST fail\n";
      PRINT_WITH_TAG(std::cerr, i);
      PRINT_WITH_TAG(std::cerr, neigh_list_ref[i]);
      PRINT_WITH_TAG(std::cerr, neigh_list[i]);
      std::exit(1);
    }
  }

  sort_neighlist(neigh_list, neigh_pointer, particle_number);
  for (int i = 0; i < number_of_pairs; i++) {
    if (neigh_list_ref[i] != neigh_list[i]) {
      std::cerr << "TEST fail\n";
      PRINT_WITH_TAG(std::cerr, i);
      PRINT_WITH_TAG(std::cerr, neigh_list_ref[i]);
      PRINT_WITH_TAG(std::cerr, neigh_list[i]);
      std::exit(1);
    }
  }

  std::cerr << "TEST is passed.\n";

  delete [] q;
  delete [] p;
}
