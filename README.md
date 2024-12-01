# **High Performance Matrix Operations: OpenMP and SIMD Implementation**

## **Overview**
The report will seek to do the implementation and optimization for both the matrix transposition and symmetry checking using OpenMP and Single Instruction Multiple Data. These are considered to be valuable contributions toward accelerating such low-level operations with modern, parallel computing techniques. There has been a great deal of improvement in speed up and efficiency realized in the work of this kind - the use of small and big matrices.
It gives an overview of the importance of matrix transposition and symmetry verification in scientific computing, giving reasons for performance with parallelization.

---

## **Contents**
1. **Introduction**  
Matrix transposition is crucial in robotics, mathematics, physics, scientific computing, machine learning, and data processing. It's essential in high-performance computing environments, where optimization can significantly improve execution time and resource usage. This project aims to design and implement three methods of matrix transpose and symmetry verification using OpenMP, identify performance bottlenecks, and determine the best technique for the project problem..

2. **Methodology**  
   Describes the computational environment and tools used:
   - **CPU**: Intel Xeon Gold 6252N with 96 logical CPUs, AVX-512, AVX2, and SSE2 support.
   - **Technologies**:
     - OpenMP: For explicit parallelization on the CPU.
     - SIMD (AVX/SSE): For implicit parallelization through vectorization.
   - **Compiler**: GNU Compiler Collection (`g++`) with performance-focused flags.

3. **Algorithms**  
   Details the transposition and symmetry verification algorithms, highlighting computational hotspots targeted for optimization.

4. **Optimizations**
   - **Sequential Implementation**  
     Baseline algorithm without parallelization or optimizations.
   - **SIMD-Based Optimization**  
     Leverages AVX-512, AVX2, and SSE2 for implicit parallelism.  
     Prefetching and blocking techniques to improve memory access efficiency.
   - **OpenMP Optimization**  
     Employs loop collapsing, multi-threading, and load balancing to improve performance for large matrices.  
     Blocking and guided scheduling enhance cache locality and workload distribution.

5. **Performance Analysis**  
   Compares execution times for sequential, SIMD, and OpenMP implementations across a range of matrix sizes.  
   Includes speedup graphs, efficiency metrics, and bottleneck analysis.



---

## **Key Findings**
1. **OpenMP**: Achieved significant speedups for large matrices by utilizing multi-threading and cache-aware blocking techniques.  
2. **SIMD**: Delivered superior performance for small matrices due to efficient vectorized computations and minimal overhead.  
3. **Scalability**: OpenMP efficiency decreased with excessive thread count, while SIMD faced limitations from memory bandwidth.  
4. **Data Structures**: `double**` provided better performance than vector-based structures due to the need for contiguous memory for SIMD optimizations.

---

## **How to Run the Code**

### **Compilation**
The project supports compilation with the GNU Compiler Collection (`g++`) and requires OpenMP support.  
Run the following commands to compile the source code:

```bash
# Compile the OpenMP implementation
g++ -O2 -march=native -ffast-math -fopenmp -funroll-loops -o es es.cpp
Dependencies
```
Ensure the g++ compiler supports OpenMP.
AVX-512 or AVX2 support is recommended for optimal performance.

### Execute
```bash
./es
```

# **Conclusion**

This project demonstrated that significant performance improvements can be achieved in matrix transposition and symmetry verification using parallel computing techniques. While OpenMP excelled in handling large matrices due to its scalability and multi-threading capabilities, SIMD optimizations provided a lightweight and efficient solution for smaller matrices.

For more details on performance analysis, please refer to the full report available in this repository.
