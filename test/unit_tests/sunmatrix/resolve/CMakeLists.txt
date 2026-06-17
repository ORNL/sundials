# ------------------------------------------------------------------------------
# Programmer(s): Jeffery Zhang
# ------------------------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2025-2026, Lawrence Livermore National Security,
# University of Maryland Baltimore County, and the SUNDIALS contributors.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ------------------------------------------------------------------------------

set(examples_list
  "test_sunmatrix_resolve.cpp\;\;")

# Add source directory to include directories
include_directories(..)

foreach(example_tuple ${examples_list})
  # parse the example tuple
  list(GET example_tuple 0 example)
  list(GET example_tuple 1 example_args)
  list(GET example_tuple 2 example_type)

  # extract the file name without extension
  get_filename_component(example_target ${example} NAME_WE)

  if(NOT TARGET ${example_target})
    # example source files
    sundials_add_executable(
      ${example_target}
      ${example}
      ../test_sunmatrix.c
      ../test_sunmatrix.h)

    # folder for IDEs
    set_target_properties(${example_target} PROPERTIES FOLDER "Examples")

    # libraries to link against
    target_link_libraries(
      ${example_target}
      PRIVATE sundials_core
              sundials_sunmatrixresolve
              SUNDIALS::ReSolve
              ${EXE_EXTRA_LINK_LIBS})
              
    # Target GPU backends if necessary
    if(SUNDIALS_RESOLVE_BACKENDS MATCHES "CUDA")
      target_link_libraries(${example_target} PRIVATE SUNDIALS::ReSolve_CUDA)
    elseif(SUNDIALS_RESOLVE_BACKENDS MATCHES "HIP")
      target_link_libraries(${example_target} PRIVATE SUNDIALS::ReSolve_HIP)
    endif()

  endif()

  # set test name
  if("${example_args}" STREQUAL "")
    set(test_name ${example_target})
  else()
    string(REGEX REPLACE " " "_" test_name ${example_target}_${example_args})
  endif()

  # add to regression tests
  sundials_add_test(
    ${test_name} ${example_target}
    TEST_ARGS ${example_args}
    EXAMPLE_TYPE ${example_type}
    NODIFF)

endforeach()