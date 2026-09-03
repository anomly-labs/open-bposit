# Copyright (c) 2026 Anomly, Inc.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.22...4.2)
project(metal_example_bposit_exact_gemm)
add_executable(metal_example_bposit_exact_gemm)
target_sources(metal_example_bposit_exact_gemm PRIVATE bposit_exact_gemm.cpp)
if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(metal_example_bposit_exact_gemm PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(metal_example_bposit_exact_gemm TT::CommonPCH)
endif()
