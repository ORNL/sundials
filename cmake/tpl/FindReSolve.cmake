# ------------------------------------------------------------------------------
# Programmer(s): Slaven Peles @ ORNL
# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------
# ReSolve find module that creates an imported target for ReSolve.
# The target is SUNDIALS::ReSolve.
#
# The variable ReSolve_DIR can be used to control where the module
# looks for the library (path to the root of a ReSolve installation or
# to a directory containing ReSolveConfig.cmake).
#
# The variable ReSolve_INCLUDE_DIR can be used to set the include path.
# The variable ReSolve_LIBRARY_DIR can be used to set the library path.
#
# This module also defines variables, but it is best to use the defined
# target to ensure includes and compile/link options are correctly passed
# to consumers.
#
#   ReSolve_FOUND       - system has the ReSolve library
#   ReSolve_LIBRARY     - the ReSolve library
#   ReSolve_INCLUDE_DIR - the ReSolve include path
#   ReSolve_LIBRARIES   - all libraries needed for ReSolve
# ------------------------------------------------------------------------------

# Prefer the upstream CMake config file if the user did not point to a specific
# include/library directory.

if(NOT
   (ReSolve_INCLUDE_DIR
    OR ReSolve_LIBRARY_DIR
    OR ReSolve_LIBRARY))

  find_package(
    ReSolve
    CONFIG
    QUIET
    PATHS
    "${ReSolve_DIR}"
    PATH_SUFFIXES
    lib/cmake/ReSolve
    cmake/ReSolve)

  if(ReSolve_FOUND AND TARGET ReSolve::ReSolve)
    if(NOT TARGET SUNDIALS::ReSolve)
      add_library(SUNDIALS::ReSolve ALIAS ReSolve::ReSolve)
    endif()

    # Check for CUDA backend
    if(TARGET ReSolve::CUDA)
      set(RESOLVE_CUDA_FOUND
          TRUE
          CACHE BOOL "ReSolve CUDA backend found")
      if(NOT TARGET SUNDIALS::ReSolve_CUDA)
        add_library(SUNDIALS::ReSolve_CUDA ALIAS ReSolve::resolve_backend_cuda)
      endif()
    endif()

    # Check for HIP backend
    if(TARGET ReSolve::HIP)
      set(RESOLVE_HIP_FOUND
          TRUE
          CACHE BOOL "ReSolve HIP backend found")
      if(NOT TARGET SUNDIALS::ReSolve_HIP)
        add_library(SUNDIALS::ReSolve_HIP ALIAS ReSolve::resolve_backend_hip)
      endif()
    endif()
    return()
  endif()

endif()

# Fall back to manual detection using ReSolve_DIR, ReSolve_INCLUDE_DIR, and
# ReSolve_LIBRARY_DIR.

# Find the include directory
find_path(
  ReSolve_INCLUDE_DIR resolve/SystemSolver.hpp
  PATHS "${ReSolve_DIR}"
  PATH_SUFFIXES include
  DOC "ReSolve include directory")

if(ReSolve_LIBRARY)
  get_filename_component(ReSolve_LIBRARY_DIR "${ReSolve_LIBRARY}" PATH)
  set(ReSolve_LIBRARY_DIR
      "${ReSolve_LIBRARY_DIR}"
      CACHE PATH "" FORCE)
else()
  find_library(
    ReSolve_LIBRARY resolve
    PATHS "${ReSolve_DIR}" "${ReSolve_LIBRARY_DIR}"
    PATH_SUFFIXES lib lib64
    DOC "ReSolve library")
endif()
mark_as_advanced(ReSolve_LIBRARY)

set(ReSolve_LIBRARIES "${ReSolve_LIBRARY}")

# Set package variables including ReSolve_FOUND
find_package_handle_standard_args(
  ReSolve REQUIRED_VARS ReSolve_LIBRARY ReSolve_LIBRARIES ReSolve_INCLUDE_DIR)

# Create the SUNDIALS::ReSolve imported target
if(ReSolve_FOUND)

  if(NOT TARGET SUNDIALS::ReSolve)
    add_library(SUNDIALS::ReSolve UNKNOWN IMPORTED)
  endif()

  set_target_properties(
    SUNDIALS::ReSolve
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ReSolve_INCLUDE_DIR}"
               INTERFACE_LINK_LIBRARIES "${ReSolve_LIBRARIES}"
               IMPORTED_LOCATION "${ReSolve_LIBRARY}")

endif()
