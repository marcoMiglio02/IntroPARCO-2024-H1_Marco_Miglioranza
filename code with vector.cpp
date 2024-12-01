
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include <immintrin.h>  // per le istruzioni AVX-512

#define BLOCK_SIZE 32
#define PREFETCH_DISTANCE 16

using namespace std;


void matTransponse(std::vector<double>& T, const std::vector<double>& M, size_t N) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            T[j * N + i] = M[i * N + j];
        }
    }
}
void matTransposeImp(std::vector<double>& T, const std::vector<double>& M, size_t N) {
    if (N <= 128) {
        for (size_t i = 0; i < N; ++i) {
            size_t j;
            for (j = 0; j + 3 < N; j += 4) {
                if (j + 4 < N) {
                    __builtin_prefetch(&M[i * N + j + 4], 0, 3);
                }
#ifdef __SSE2__
                __m256d data = _mm256_loadu_pd(&M[i * N + j]);
                _mm256_storeu_pd(&T[j * N + i], data);
#else
                T[j * N + i] = M[i * N + j];
                T[(j + 1) * N + i] = M[i * N + j + 1];
                T[(j + 2) * N + i] = M[i * N + j + 2];
                T[(j + 3) * N + i] = M[i * N + j + 3];
#endif
            }
            for (; j < N; ++j) {
                T[j * N + i] = M[i * N + j];
            }
        }
    } else if (N <= 1024) {
        for (size_t i = 0; i < N; i += BLOCK_SIZE) {
            for (size_t j = 0; j < N; j += BLOCK_SIZE) {
                for (size_t ii = i; ii < i + BLOCK_SIZE && ii < N; ++ii) {
                    size_t jj;
                    for (jj = j; jj + 3 < j + BLOCK_SIZE && jj + 3 < N; jj += 4) {
                        if (jj + 4 < j + BLOCK_SIZE && jj + 4 < N) {
                            __builtin_prefetch(&M[ii * N + jj + 4], 0, 3);
                        }
#ifdef __SSE2__
                        __m256d data = _mm256_loadu_pd(&M[ii * N + jj]);
                        _mm256_storeu_pd(&T[jj * N + ii], data);
#else
                        T[jj * N + ii] = M[ii * N + jj];
                        T[(jj + 1) * N + ii] = M[ii * N + jj + 1];
                        T[(jj + 2) * N + ii] = M[ii * N + jj + 2];
                        T[(jj + 3) * N + ii] = M[ii * N + jj + 3];
#endif
                    }
                    for (; jj < j + BLOCK_SIZE && jj < N; ++jj) {
                        T[jj * N + ii] = M[ii * N + jj];
                    }
                }
            }
        }
    } else {
        // Per matrici più grandi
        for (size_t i = 0; i < N; i += BLOCK_SIZE) {
            for (size_t j = 0; j < N; j += BLOCK_SIZE) {
                for (size_t ii = i; ii < i + BLOCK_SIZE && ii < N; ++ii) {
                    size_t jj;
                    for (jj = j; jj + 7 < j + BLOCK_SIZE && jj + 7 < N; jj += 8) {
                        if (jj + PREFETCH_DISTANCE < N) {
                            __builtin_prefetch(&M[ii * N + jj + PREFETCH_DISTANCE], 0, 3);
                        }
#ifdef __AVX512F__
                        __m512d data = _mm512_loadu_pd(&M[ii * N + jj]);
                        _mm512_storeu_pd(&T[jj * N + ii], data);
#elif defined(__SSE2__)
                        __m256d data1 = _mm256_loadu_pd(&M[ii * N + jj]);
                        __m256d data2 = _mm256_loadu_pd(&M[ii * N + jj + 4]);
                        _mm256_storeu_pd(&T[jj * N + ii], data1);
                        _mm256_storeu_pd(&T[(jj + 4) * N + ii], data2);
#else
                        T[jj * N + ii] = M[ii * N + jj];
                        T[(jj + 1) * N + ii] = M[ii * N + jj + 1];
                        T[(jj + 2) * N + ii] = M[ii * N + jj + 2];
                        T[(jj + 3) * N + ii] = M[ii * N + jj + 3];
                        T[(jj + 4) * N + ii] = M[ii * N + jj + 4];
                        T[(jj + 5) * N + ii] = M[ii * N + jj + 5];
                        T[(jj + 6) * N + ii] = M[ii * N + jj + 6];
                        T[(jj + 7) * N + ii] = M[ii * N + jj + 7];
#endif
                    }
                    for (; jj < j + BLOCK_SIZE && jj < N; ++jj) {
                        T[jj * N + ii] = M[ii * N + jj];
                    }
                }
            }
        }
    }
}
    void matTransposeOMP(std::vector<double>& T, const std::vector<double>& M, size_t N) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; i += BLOCK_SIZE) {
        for (size_t j = 0; j < N; j += BLOCK_SIZE) {
            // Trasporre il blocco che inizia a [i, j]
            for (size_t ii = i; ii < i + BLOCK_SIZE && ii < N; ++ii) {
                for (size_t jj = j; jj < j + BLOCK_SIZE && jj < N; ++jj) {
                    T[jj * N + ii] = M[ii * N + jj];
                }
            }
        }
    }
}

bool checkSym(const std::vector<double>& M, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            if (M[i * N + j] != M[j * N + i]) {
                return false;
            }
        }
    }
    return true;
}


bool checkSymIMP(const std::vector<double>& M, size_t N) {
    bool symmetric = true;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j + 7 < N; j += 8) {
            // Prefetch delle future righe per ridurre i cache miss
            if (j + PREFETCH_DISTANCE < N) {
                __builtin_prefetch(&M[i * N + j + PREFETCH_DISTANCE], 0, 3);
                __builtin_prefetch(&M[j * N + i + PREFETCH_DISTANCE], 0, 3);
            }
#ifdef __AVX512F__
            // AVX-512 implementation: carico 8 double in parallelo
            __m512d row_i = _mm512_loadu_pd(&M[i * N + j]);
            __m512d row_j = _mm512_loadu_pd(&M[j * N + i]);
            __mmask8 cmp = _mm512_cmp_pd_mask(row_i, row_j, _CMP_NEQ_OQ);
            if (cmp != 0) {
                symmetric = false;
                return symmetric;
            }
#elif defined(__SSE2__)
            // AVX2/SSE2 implementation: carico 4 double alla volta
            __m256d row_i_1 = _mm256_loadu_pd(&M[i * N + j]);
            __m256d row_j_1 = _mm256_loadu_pd(&M[j * N + i]);
            __m256d cmp_1 = _mm256_cmp_pd(row_i_1, row_j_1, _CMP_NEQ_OQ);
            if (!_mm256_testz_pd(cmp_1, cmp_1)) {
                symmetric = false;
                return symmetric;
            }

            __m256d row_i_2 = _mm256_loadu_pd(&M[i * N + j + 4]);
            __m256d row_j_2 = _mm256_loadu_pd(&M[(j + 4) * N + i]);
            __m256d cmp_2 = _mm256_cmp_pd(row_i_2, row_j_2, _CMP_NEQ_OQ);
            if (!_mm256_testz_pd(cmp_2, cmp_2)) {
                symmetric = false;
                return symmetric;
            }
#else
            // Fallback manuale senza AVX/SIMD, usando unrolling
            if (M[i * N + j] != M[j * N + i] ||
                M[i * N + j + 1] != M[(j + 1) * N + i] ||
                M[i * N + j + 2] != M[(j + 2) * N + i] ||
                M[i * N + j + 3] != M[(j + 3) * N + i] ||
                M[i * N + j + 4] != M[(j + 4) * N + i] ||
                M[i * N + j + 5] != M[(j + 5) * N + i] ||
                M[i * N + j + 6] != M[(j + 6) * N + i] ||
                M[i * N + j + 7] != M[(j + 7) * N + i]) {
                symmetric = false;
                return symmetric;
            }
#endif
        }
        // Gestione degli elementi rimanenti che non sono divisibili per 8
        for (size_t j = (N & ~7); j < N; ++j) {
            if (M[i * N + j] != M[j * N + i]) {
                symmetric = false;
                return symmetric;
            }
        }
    }

    return symmetric;
}


bool checkSymOMP(const std::vector<double>& M, size_t N) {

    bool symmetric = true;

     #pragma omp parallel
    {
        bool local_symmetric = true;
        #pragma omp for schedule(guided) nowait
        for (size_t i = 0; i < N-1; ++i) {
            if (!symmetric) continue;  // Skip remaining work if asymmetry found
            for (size_t j = i + 1; j < N; ++j) {
                if (M[i * N + j] != M[j * N + i]) {
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

    // Imposta il numero di thread per OpenMP
    omp_set_num_threads(8);

    // Creiamo i vettori che rappresentano le matrici M e T (inizializzate a 2.0)
    std::vector<double> M(N * N, 2.0);
    std::vector<double> T(N * N, 0.0);

    // Timer per la misurazione del tempo
    auto start = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> time;

    // Trasposizione base
    start = chrono::high_resolution_clock::now();
    matTransponse(T, M, N);
    end = chrono::high_resolution_clock::now();
    time = end - start;
    cout << "Basic Transpose: " << time.count() << "s" << endl;

    // Trasposizione migliorata
    start = chrono::high_resolution_clock::now();
    matTransposeImp(T, M, N);
    end = chrono::high_resolution_clock::now();
    time = end - start;
    cout << "Improved Transpose: " << time.count() << "s" << endl;

    // Trasposizione parallela con OpenMP
    start = chrono::high_resolution_clock::now();
    matTransposeOMP(T, M, N);
    end = chrono::high_resolution_clock::now();
    time = end - start;
    cout << "OpenMP Transpose: " << time.count() << "s" << endl;

    // Verifica della simmetria di base
    start = chrono::high_resolution_clock::now();
    bool check = checkSym(M, N);
    end = chrono::high_resolution_clock::now();
    time = end - start;
    cout << "Basic Check Symmetry: " << time.count() << "s and the result is: " << (check ? "true" : "false") << endl;

    // Verifica della simmetria ottimizzata
    start = chrono::high_resolution_clock::now();
    check = checkSymIMP(M, N);
    end = chrono::high_resolution_clock::now();
    time = end - start;
    cout << "Improved Check Symmetry: " << time.count() << "s and the result is: " << (check ? "true" : "false") << endl;

    // Verifica della simmetria parallela con OpenMP
    start = chrono::high_resolution_clock::now();
    check = checkSymOMP(M, N);
    end = chrono::high_resolution_clock::now();
    time = end - start;
    cout << "OpenMP Check Symmetry: " << time.count() << "s and the result is: " << (check ? "true" : "false") << endl;

    return 0;
}