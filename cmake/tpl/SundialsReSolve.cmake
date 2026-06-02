# -----------------------------------------------------------------------------
# Programmer(s): Slaven Peles @ ORNL
# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------
# Module to find and setup ReSolve.
# ReSolve is a GPU-friendly linear solver library developed at ORNL:
# https://github.com/ORNL/ReSolve
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Section 1: Include guard
# -----------------------------------------------------------------------------

include_guard(GLOBAL)

# -----------------------------------------------------------------------------
# Section 2: Check to make sure options are compatible
# -----------------------------------------------------------------------------

# ReSolve is a C++ library; CXX must be enabled (SundialsSetupCompilers.cmake
# gates include(SundialsSetupCXX) on SUNDIALS_ENABLE_RESOLVE).
if(NOT CMAKE_CXX_COMPILER_LOADED)
  message(FATAL_ERROR "ReSolve requires C++ but no C++ compiler was found. "
                      "Enable a C++ compiler or set CMAKE_CXX_COMPILER.")
endif()

if(CMAKE_CXX_STANDARD LESS "14")
  message(FATAL_ERROR "CMAKE_CXX_STANDARD must be >= 14 when using ReSolve")
endif()

# -----------------------------------------------------------------------------
# Section 3: Find the TPL
# -----------------------------------------------------------------------------

find_package(ReSolve REQUIRED)

message(STATUS "ReSolve_LIBRARIES:   ${ReSolve_LIBRARIES}")
message(STATUS "ReSolve_INCLUDE_DIR: ${ReSolve_INCLUDE_DIR}")

# -----------------------------------------------------------------------------
# Section 4: Test the TPL
# -----------------------------------------------------------------------------

if(SUNDIALS_ENABLE_RESOLVE_CHECKS)

  message(CHECK_START "Testing ReSolve")

  set(TEST_DIR ${PROJECT_BINARY_DIR}/RESOLVE_TEST)

  # Use the self-contained Common.hpp rather than SystemSolver.hpp; the latter
  # has missing internal includes in some ReSolve versions.
  file(
    WRITE ${TEST_DIR}/test.cpp
    "\#include <resolve/Common.hpp>\n" "int main(void) {\n"
    "  ReSolve::real_type x = ReSolve::constants::ONE;\n" "  (void)x;\n"
    "  return 0;\n" "}\n")

  try_compile(
    COMPILE_OK ${TEST_DIR}
    ${TEST_DIR}/test.cpp
    LINK_LIBRARIES SUNDIALS::ReSolve
    OUTPUT_VARIABLE COMPILE_OUTPUT)

  if(COMPILE_OK)
    message(CHECK_PASS "success")
  else()
    message(CHECK_FAIL "failed")
    file(WRITE ${TEST_DIR}/compile.out "${COMPILE_OUTPUT}")
    message(
      FATAL_ERROR
        "Could not compile ReSolve test. Check output in ${TEST_DIR}/compile.out"
    )
  endif()

else()
  message(STATUS "Skipped ReSolve checks.")
endif()
