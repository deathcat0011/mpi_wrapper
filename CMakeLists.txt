cmake_minimum_required(VERSION 3.18.0)

if (WIN32)
   message("")
   message("")
   message(SEND_ERROR " ------ OPENMPI NOT SUPPORTED ON WINDOWS ------ ")
   message("")
   message("")
endif()

project(mpi_wrapper VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")

set(CMAKE_C_COMPILER "mpic")
set(CMAKE_CXX_COMPILER "mpic++")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)

add_library(mpi_wrapper INTERFACE include/mpi_wrapper.hpp)

target_include_directories(mpi_wrapper INTERFACE "include/")
set_target_properties(mpi_wrapper PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(mpi_wrapper INTERFACE "-DOMPI_CXX=clang++ -DOMPI_MPICC=clang")

target_link_libraries(mpi_wrapper
   ${OpenMP_CXX_LIBRARIES}
   # OpenMPI::OpenMPI_CXX
   # OpenMPI::OpenMPI_C
)
set_property(TARGET mpi_wrapper PROPERTY INTERPROCEDURAL_OPTIMIZATION True)

add_library(mpi_wrapper_shared SHARED include/mpi_wrapper.hpp)
target_include_directories(mpi_wrapper_shared INTERFACE "include/")
set_target_properties(mpi_wrapper_shared PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(mpi_wrapper_shared INTERFACE "-DOMPI_CXX=clang++ -DOMPI_MPICC=clang")

target_link_libraries(mpi_wrapper_shared
   ${OpenMP_CXX_LIBRARIES}
   # OpenMPI::OpenMPI_CXX
   # OpenMPI::OpenMPI_C
)
set_property(TARGET mpi_wrapper_shared PROPERTY INTERPROCEDURAL_OPTIMIZATION True)


add_library(mpi_wrapper_static STATIC include/mpi_wrapper.hpp)
target_include_directories(mpi_wrapper_static INTERFACE "include/")
set_target_properties(mpi_wrapper_static PROPERTIES LINKER_LANGUAGE CXX)
target_compile_definitions(mpi_wrapper_static INTERFACE "-DOMPI_CXX=clang++ -DOMPI_MPICC=clang")

target_link_libraries(mpi_wrapper_static
   ${OpenMP_CXX_LIBRARIES}
   # OpenMPI::OpenMPI_CXX
   # OpenMPI::OpenMPI_C
)
set_property(TARGET mpi_wrapper_static PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
