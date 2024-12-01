#include <iostream>
#include <chrono>
#include <immintrin.h>  // for AVX-512 intrinsics
#include <cstddef>
#include<random>
#include<omp.h>
#define BLOCK_SIZE 32
#define PREFETCH_DISTANCE 16


using namespace std;

void matTransponse(double** T, double** M, size_t N) {


    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            T[j][i] = M[i][j];
        }
    }


}





void matTransposeImp(double** T, double** M, size_t N) {
    if (N <= 128) {
        for (size_t i = 0; i < N; ++i) {
            size_t j;
            // Main vectorized loop
            for (j = 0; j + 3 < N; j += 4) {
                if (j + 4 < N) {
                    __builtin_prefetch(&M[i][j + 4], 0, 3);
                }
#ifdef __SSE2__
                __m256d data = _mm256_loadu_pd(&M[i][j]);
                _mm256_storeu_pd(&T[j][i], data);
#else
                T[j][i] = M[i][j];
                T[j + 1][i] = M[i][j + 1];
                T[j + 2][i] = M[i][j + 2];
                T[j + 3][i] = M[i][j + 3];
#endif
            }
            // Handle remaining elements
            for (; j < N; ++j) {
                T[j][i] = M[i][j];
            }
        }
    } else if (N <= 1024) {
        for (size_t i = 0; i < N; i += BLOCK_SIZE) {
            for (size_t j = 0; j < N; j += BLOCK_SIZE) {
                for (size_t ii = i; ii < i + BLOCK_SIZE && ii < N; ++ii) {
                    size_t jj;
                    // Main vectorized block loop
                    for (jj = j; jj + 3 < j + BLOCK_SIZE && jj + 3 < N; jj += 4) {
                        if (jj + 4 < j + BLOCK_SIZE && jj + 4 < N) {
                            __builtin_prefetch(&M[ii][jj + 4], 0, 3);
                        }
#ifdef __SSE2__
                        __m256d data = _mm256_loadu_pd(&M[ii][jj]);
                        _mm256_storeu_pd(&T[jj][ii], data);
#else
                        T[jj][ii] = M[ii][jj];
                        T[jj + 1][ii] = M[ii][jj + 1];
                        T[jj + 2][ii] = M[ii][jj + 2];
                        T[jj + 3][ii] = M[ii][jj + 3];
#endif
                    }
                    // Handle remaining block elements
                    for (; jj < j + BLOCK_SIZE && jj < N; ++jj) {
                        T[jj][ii] = M[ii][jj];
                    }
                }
            }
        }
    } else {
        for (size_t i = 0; i < N; i += BLOCK_SIZE) {
            for (size_t j = 0; j < N; j += BLOCK_SIZE) {
                for (size_t ii = i; ii < i + BLOCK_SIZE && ii < N; ++ii) {
                    size_t jj;
                    // Main vectorized block loop
                    for (jj = j; jj + 7 < j + BLOCK_SIZE && jj + 7 < N; jj += 8) {
                        if (jj + PREFETCH_DISTANCE < N) {
                            __builtin_prefetch(&M[ii][jj + PREFETCH_DISTANCE], 0, 3);
                        }
#ifdef __AVX512F__
                        __m512d data = _mm512_loadu_pd(&M[ii][jj]);
                        _mm512_storeu_pd(&T[jj][ii], data);
#elif defined(__SSE2__)
                        __m256d data1 = _mm256_loadu_pd(&M[ii][jj]);
                        __m256d data2 = _mm256_loadu_pd(&M[ii][jj + 4]);
                        _mm256_storeu_pd(&T[jj][ii], data1);
                        _mm256_storeu_pd(&T[jj + 4][ii], data2);
#else
                        T[jj][ii] = M[ii][jj];
                        T[jj + 1][ii] = M[ii][jj + 1];
                        T[jj + 2][ii] = M[ii][jj + 2];
                        T[jj + 3][ii] = M[ii][jj + 3];
                        T[jj + 4][ii] = M[ii][jj + 4];
                        T[jj + 5][ii] = M[ii][jj + 5];
                        T[jj + 6][ii] = M[ii][jj + 6];
                        T[jj + 7][ii] = M[ii][jj + 7];
#endif
                    }
                    // Handle remaining block elements
                    for (; jj < j + BLOCK_SIZE && jj < N; ++jj) {
                        T[jj][ii] = M[ii][jj];
                    }
                }
            }
        }
    }
}


void matTransposeOMP(double** T, double** M, size_t N) {
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; i += BLOCK_SIZE) {
        for (size_t j = 0; j < N; j += BLOCK_SIZE) {
            // transpose the block beginning at [i,j]
            for (size_t ii = i; ii < i + BLOCK_SIZE && ii < N; ++ii) {
                for (size_t jj = j; jj < j + BLOCK_SIZE && jj < N ; ++jj) {
                  
                    T[jj][ii] = M[ii][jj];
                }
            }
        }
    }
}



bool checkSym(double** M, size_t N) {
    for (size_t i = 0; i < N; i++) 
        for (size_t j = 0; j < N; j++) 
            if (M[i][j] != M[j][i]) 
                return false; 
    return true; 
}

bool checkSymIMP(double** M, size_t N) {
    bool symmetric = true;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j + 7 < N; j += 8) {
            // Prefetch delle future righe per ridurre i cache miss
            if (j + PREFETCH_DISTANCE < N) {
                __builtin_prefetch(&M[i][j + PREFETCH_DISTANCE], 0, 3);
                __builtin_prefetch(&M[j][i + PREFETCH_DISTANCE], 0, 3);
            }
#ifdef __AVX512F__
            // AVX-512 implementation
            __m512d row_i_1 = _mm512_loadu_pd(&M[i][j]);
            __m512d row_j_1 = _mm512_loadu_pd(&M[j][i]);
            __mmask8 cmp_1 = _mm512_cmp_pd_mask(row_i_1, row_j_1, _CMP_NEQ_OQ);
            if (cmp_1 != 0) {
                symmetric = false;
                return symmetric;
            }
#elif defined(__SSE2__)
            // AVX2/SSE implementation
            __m256d row_i_1 = _mm256_loadu_pd(&M[i][j]);
            __m256d row_j_1 = _mm256_loadu_pd(&M[j][i]);
            __m256d cmp_1 = _mm256_cmp_pd(row_i_1, row_j_1, _CMP_NEQ_OQ);
            if (!_mm256_testz_pd(cmp_1, cmp_1)) {
                symmetric = false;
                return symmetric;
            }
#else
            // Fallback with unrolling by 8
            if (M[i][j] != M[j][i] || M[i][j + 1] != M[j + 1][i] ||
                M[i][j + 2] != M[j + 2][i] || M[i][j + 3] != M[j + 3][i] ||
                M[i][j + 4] != M[j + 4][i] || M[i][j + 5] != M[j + 5][i] ||
                M[i][j + 6] != M[j + 6][i] || M[i][j + 7] != M[j + 7][i]) {
                symmetric = false;
                return symmetric;
            }
#endif
        }
        for (size_t j = (N & ~7); j < N; ++j) {
            if (M[i][j] != M[j][i]) {
                symmetric = false;
                return symmetric;
            }
        }
    }

    return symmetric;
}


bool checkSymOMP(double** M, size_t N) {

    bool symmetric = true;

     #pragma omp parallel
    {
        bool local_symmetric = true;
        #pragma omp for schedule(guided) nowait
        for (size_t i = 0; i < N-1; ++i) {
            if (!symmetric) continue;  // Skip remaining work if asymmetry found
            for (size_t j = i + 1; j < N; ++j) {
                if (M[i][j] != M[j ][i]) {
                    local_symmetric = false;
                    symmetric = false;
                    break;  // Exit inner loop
                }
            }
        }
        #pragma omp critical
        symmetric = symmetric && local_symmetric;
    }
    
    return symmetric;
}

 



int main() {
    
    size_t N;
    do {
    cout << "Insert the size of the matrix (from 16 to 4096): ";
    cin >> N;
    } while (N < 16 || N > 4096);
    
    
    
    omp_set_num_threads(8);
    auto** M = new double*[N];
    auto** T = new double*[N];    
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(1.0, 10.0);

    //Creating and initializing the Matrix
    for (size_t i = 0; i < N; i++) {
        M[i] = new double[N];
        T[i] = new double[N];

        for (size_t j = 0; j < N; j++)
            M[i][j] =  2;


    }

    auto start = chrono::high_resolution_clock::now();
    matTransponse(T, M, N);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> time = end - start;
    cout << "Basic Transpose: " << time.count() << "s" << endl;

    start = chrono::high_resolution_clock::now();
    matTransposeImp(T, M, N);
    end = chrono::high_resolution_clock::now();
    time = end - start;
    cout << "Improved Transpose: " << time.count() << "s" << endl;

    start = chrono::high_resolution_clock::now();
    matTransposeOMP(T, M, N);
    end = chrono::high_resolution_clock::now();
    time = end - start;
    cout << "OpenMP Transpose: " << time.count() << "s" << endl;
    
    
    start = chrono::high_resolution_clock::now();
    auto check = checkSym(M, N);
    end = chrono::high_resolution_clock::now();
    time = end - start;
    cout << "Basic Check Symmetry: " << time.count() << "s and the result is: " << check << endl;
    
    start = chrono::high_resolution_clock::now();
    check = checkSymIMP(M, N);
    end = chrono::high_resolution_clock::now();
    time = end - start;
    cout << "Implicit Check Symmetry: " << time.count() << "s and the result is: " << check << endl;
    
    start = chrono::high_resolution_clock::now();
    check = checkSymOMP(M, N);
    end = chrono::high_resolution_clock::now();
    time = end - start;
    cout << "OpenMP Check Symmetry: " << time.count() << "s and the result is: " << check << endl;

    return 0;
}



