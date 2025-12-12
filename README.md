# Advanced Tutorial: Matrix Transpose using MPI and OpenMP
## By: Sumaya Houssini Mohamed
## Introducton:
This tutorial aims at showing the difference in performance when using pure MPI and MPI + OpenMP for matrix tranpose. 

## Project Structure:
The project looks like this:  
```` ``` ````
.
├── C 
│   ├── main.cpp 
│   └── main_solutions.cpp
├── CMakeLists.txt
├── compile.sh
├── examples
│   ├── CMakeLists.txt
│   └── example.cpp
├── LICENSE
├── README.md
└── src
    ├── CMakeLists.txt
    ├── LICENSE
    ├── README.md
    ├── src.cpp
    ├── src.hpp
    └── tests
        ├── CMakeLists.txt
        ├── test_finalize.cpp
        ├── test_init.cpp
        └── unit_test.cpp
```` ``` ````