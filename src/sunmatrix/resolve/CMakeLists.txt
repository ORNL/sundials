# ---------------------------------------------------------------
# Programmer(s): Jeffery Zhang
# ---------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2025-2026, Lawrence Livermore National Security,
# University of Maryland Baltimore County, and the SUNDIALS contributors.
# Copyright (c) 2013-2025, Lawrence Livermore National Security
# and Southern Methodist University.
# Copyright (c) 2002-2013, Lawrence Livermore National Security.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ---------------------------------------------------------------

# TODO Add ReSolve Backends for now hardcode cpu

install(
  CODE "MESSAGE(\"\nInstall SUNMATRIX_RESOLVE with CPU backend(s)\n\")"
)

# if(SUNDIALS_MAGMA_BACKENDS MATCHES "CUDA")
#   set_source_files_properties(sunmatrix_magmadense.cpp PROPERTIES LANGUAGE CUDA)
#   set(_libs_needed sundials_nveccuda)
# elseif(SUNDIALS_MAGMA_BACKENDS MATCHES "HIP")
#   set_source_files_properties(sunmatrix_magmadense.cpp PROPERTIES LANGUAGE CXX)
#   set(_libs_needed sundials_nvechip hip::device)
# endif()

# Add the sunmatrix_resolve library
sundials_add_library(
  sundials_sunmatrixresolve
  SOURCES sunmatrix_resolve.cpp
  HEADERS ${SUNDIALS_SOURCE_DIR}/include/sunmatrix/sunmatrix_resolve.h
  INCLUDE_SUBDIR sunmatrix
  LINK_LIBRARIES PUBLIC sundials_core
  OBJECT_LIBRARIES
  INCLUDE_DIRECTORIES PUBLIC ${ReSolve_INCLUDE_DIR}
  LINK_LIBRARIES PUBLIC SUNDIALS::ReSolve ${_libs_needed}
  OUTPUT_NAME sundials_sunmatrixresolve
  VERSION ${sunmatrixlib_VERSION}
  SOVERSION ${sunmatrixlib_SOVERSION})

message(
  STATUS
    "Added SUNMATRIX_RESOLVE module with CPU backend(s)"
)
