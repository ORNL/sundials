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

install(
  CODE "MESSAGE(\"\nInstall SUNLINSOL_RESOLVE with ${SUNDIALS_RESOLVE_BACKENDS} backend(s)\n\")"
)

if(SUNDIALS_RESOLVE_BACKENDS MATCHES "CUDA")
  set(_libs_needed sundials_nveccuda)
elseif(SUNDIALS_RESOLVE_BACKENDS MATCHES "HIP")
  set(_libs_needed sundials_nvechip)
endif()

# Add the sunlinsol_resolve library
sundials_add_library(
  sundials_sunlinsolresolve
  SOURCES sunlinsol_resolve.cpp
  HEADERS ${SUNDIALS_SOURCE_DIR}/include/sunlinsol/sunlinsol_resolve.hpp
  INCLUDE_SUBDIR sunlinsol
  LINK_LIBRARIES PUBLIC sundials_core
  OBJECT_LIBRARIES
  INCLUDE_DIRECTORIES PUBLIC ${ReSolve_INCLUDE_DIR}
  LINK_LIBRARIES PUBLIC SUNDIALS::ReSolve sundials_sunmatrixresolve
                 ${_libs_needed}
  OUTPUT_NAME sundials_sunlinsolresolve
  VERSION ${sunlinsollib_VERSION}
  SOVERSION ${sunlinsollib_SOVERSION})

message(
  STATUS
    "Added SUNLINSOL_RESOLVE module with ${SUNDIALS_RESOLVE_BACKENDS} backend(s)"
)
