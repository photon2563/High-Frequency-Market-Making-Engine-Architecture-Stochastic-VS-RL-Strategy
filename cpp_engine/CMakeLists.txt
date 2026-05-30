cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(hfmm-architecture)

# Find LibTorch.
# We will download it dynamically or expect CMAKE_PREFIX_PATH to point to it
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Define our executable
add_executable(inference_engine src/inference_engine.cpp)
target_link_libraries(inference_engine "${TORCH_LIBRARIES}")
set_property(TARGET inference_engine PROPERTY CXX_STANDARD 17)

# Ensure proper RPATH if running on macOS/Linux so it can find libtorch.dylib/so
if (APPLE)
  set_target_properties(inference_engine PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@loader_path/../lib;@loader_path"
  )
endif()
