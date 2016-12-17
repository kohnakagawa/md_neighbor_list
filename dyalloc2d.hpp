#pragma once

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
