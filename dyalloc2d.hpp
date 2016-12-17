#pragma once

#include <iostream>
#include <cstdlib>

template <typename T, int alignment>
void allocate2D_aligend(const int Nx,
                        const int Ny,
                        T**& ptr) {
  ptr = static_cast<T**>(malloc(Nx * sizeof(T*)));
  const auto stat = posix_memalign(reinterpret_cast<void**>(ptr),
                                   alignment,
                                   sizeof(T) * Nx * Ny);
  if (stat) {
    std::cerr << "error occurs at allocate2D_aligend\n";
    std::exit(1);
  }
  for (int i = 1; i < Nx; i++) ptr[i] = ptr[0] + i * Ny;
}

template <typename T>
void deallocate2D_aligend(T**& ptr) {
  free(ptr[0]);
  free(ptr);
  ptr = nullptr;
}

template <typename T>
void allocate2D(const int Nx,
                const int Ny,
                T**& ptr) {
  ptr = new T* [Nx];
  ptr[0] = new T[Nx * Ny];
  for (int i = 1; i < Nx; i++) ptr[i] = ptr[0] + i * Ny;
}

template <typename T>
void deallocate2D(T**& ptr) {
  delete [] ptr[0];
  delete [] ptr;
  ptr = nullptr;
}
