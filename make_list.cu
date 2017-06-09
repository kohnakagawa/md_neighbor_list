#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

#if 1
typedef double Dtype;
typedef double4 Vectype;
#else
typedef float Dtype;
typedef float4 Vectype;
#endif

#include "cuda_ptr.cuh"
#include "neighlist_gpu.hpp"

const Dtype density = 1.0;
const int N = 400000; // maximum buffer size
const int MAX_NEIGH_N = 400;
const int LOOP = 100;
const Dtype L = 50.0;

const Dtype SEARCH_LENGTH = 3.3;
const Dtype SEARCH_LENGTH2 = SEARCH_LENGTH * SEARCH_LENGTH;

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

template <typename Vec1, typename Vec2>
void copy_vec(Vec1* v1,
              Vec2* v2,
              const int ptcl_num) {
  for (int i = 0; i < ptcl_num; i++) {
    v1[i].x = v2[i].x;
    v1[i].y = v2[i].y;
    v1[i].z = v2[i].z;
  }
}

template <typename Vec>
void make_neighlist_bruteforce(const Vec* q,
                               const int32_t particle_number,
                               std::vector<int>& neighlist,
                               std::vector<int>& number_of_partners) {
  for (int i = 0; i < particle_number; i++) {
    int n_neigh = 0;
    for (int j = 0; j < particle_number; j++) {
      if (i == j) continue;
      const auto drx = q[i].x - q[j].x;
      const auto dry = q[i].y - q[j].y;
      const auto drz = q[i].z - q[j].z;
      const auto dr2 = drx * drx + dry * dry + drz * drz;
      if (dr2 > SEARCH_LENGTH2) continue;
      neighlist[particle_number * n_neigh + i] = j;
      n_neigh++;
    }
    number_of_partners[i] = n_neigh;
  }
}

#define PRINT_WITH_TAG(ost, val) ost << #val << " " << val << "\n"

int main(int argc, char* argv[]) {
  int32_t tblock_size = 128, smem_hei = 7;
  if (argc != 3) {
    std::cerr << "argv[1] = thread block size (default 128).\n";
    std::cerr << "argv[2] = shared memory tile height (default 7, max 20).\n";
  } else {
    tblock_size = std::atoi(argv[1]);
    smem_hei    = std::atoi(argv[2]);
  }

  // buffer data
  cuda_ptr<Vectype> q, p;
  q.allocate(N); p.allocate(N);

  // initialize and copy to device
  int particle_number = 0;
  init(&q[0], particle_number);
  q.host2dev();

  // run gpu nearlist construction
  NeighListGPU<Vectype, Dtype> nlistmaker(SEARCH_LENGTH, L, L, L);
  nlistmaker.Initialize(particle_number);
  const auto beg = std::chrono::system_clock::now();
  for (int i = 0; i < LOOP; i++) {
    nlistmaker.MakeNeighList(q, particle_number, false, tblock_size, smem_hei);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  const auto end = std::chrono::system_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
  std::cout << "# of particles " << particle_number
            << " " << elapsed << "[ms]\n";
  const auto number_of_pairs = nlistmaker.number_of_pairs();

  // get gpu neighbor list
  cuda_ptr<int32_t>& neigh_list = nlistmaker.neigh_list();
  cuda_ptr<int32_t>& number_of_partners = nlistmaker.number_of_partners();

  // copy to host
  neigh_list.dev2host();
  number_of_partners.dev2host();
  q.dev2host();

  // run cpu reference
  std::vector<int> neigh_list_ref, number_of_partners_ref;
  neigh_list_ref.resize(particle_number * MAX_NEIGH_N, -1);
  number_of_partners_ref.resize(particle_number, -1);
  make_neighlist_bruteforce(&q[0],
                            particle_number,
                            neigh_list_ref,
                            number_of_partners_ref);
  const auto number_of_pairs_ref = std::accumulate(number_of_partners_ref.begin(),
                                                   number_of_partners_ref.end(),
                                                   0);

  // check the correctness of neighbor list
  if (number_of_pairs != number_of_pairs_ref) {
    std::cerr << "TEST fail\n";
    PRINT_WITH_TAG(std::cerr, number_of_pairs);
    PRINT_WITH_TAG(std::cerr, number_of_pairs_ref);
    std::exit(1);
  }

  for (int i = 0; i < particle_number; i++) {
    if (number_of_partners[i] != number_of_partners_ref[i]) {
      std::cerr << "TEST fail\n";
      PRINT_WITH_TAG(std::cerr, i);
      PRINT_WITH_TAG(std::cerr, number_of_partners[i]);
      PRINT_WITH_TAG(std::cerr, number_of_partners_ref[i]);
      std::exit(1);
    }
  }

  const auto max_num_of_partners = *std::max_element(number_of_partners_ref.begin(),
                                                     number_of_partners_ref.end());
  std::vector<int> ref_buf(max_num_of_partners, 0);
  std::vector<int> gpu_buf(max_num_of_partners, 0);
  for (int i = 0; i < particle_number; i++) {
    for (int j = 0; j < number_of_partners[i]; j++) {
      ref_buf[j] = neigh_list_ref[particle_number * j + i];
      gpu_buf[j] = neigh_list[particle_number * j + i];
    }
    std::sort(ref_buf.begin(), ref_buf.begin() + number_of_partners[i]);
    std::sort(gpu_buf.begin(), gpu_buf.begin() + number_of_partners[i]);

    for (int j = 0; j < number_of_partners[i]; j++) {
      const auto gpu_neigh_list = gpu_buf[j];
      const auto ref            = ref_buf[j];
      if (gpu_neigh_list != ref) {
        std::cerr << "TEST fail\n";
        PRINT_WITH_TAG(std::cerr, i);
        PRINT_WITH_TAG(std::cerr, j);
        PRINT_WITH_TAG(std::cerr, gpu_neigh_list);
        PRINT_WITH_TAG(std::cerr, ref);
        std::exit(1);
      }
    }
  }

  std::cerr << "TEST is passed.\n";
}
