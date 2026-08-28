cmake_minimum_required(VERSION 3.25)
project(xtllm VERSION 0.3.0 LANGUAGES CXX)

if(NOT WIN32)
  message(FATAL_ERROR "This snapshot's native runtime currently requires Windows. See docs/LINUX.md.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Vulkan REQUIRED)
find_program(GLSLANG_VALIDATOR NAMES glslangValidator glslang REQUIRED)

file(GLOB XTLLM_SHADERS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dsv4_*.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/qwen35_*.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/qwen36_*.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/qwen38_*.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/qwen_next_*.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/longcat_*.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/nemotron3_*.comp")
list(APPEND XTLLM_SHADERS
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/step37_rmsnorm.comp"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/step37_swiglu.comp")
list(REMOVE_DUPLICATES XTLLM_SHADERS)

set(XTLLM_SPIRV)
foreach(SHADER IN LISTS XTLLM_SHADERS)
  get_filename_component(NAME "${SHADER}" NAME_WE)
  set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.comp.spv")
  add_custom_command(OUTPUT "${OUTPUT}"
    COMMAND "${GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -S comp
            -o "${OUTPUT}" "${SHADER}"
    DEPENDS "${SHADER}" VERBATIM)
  list(APPEND XTLLM_SPIRV "${OUTPUT}")
endforeach()
add_custom_target(xtllm_shaders ALL DEPENDS ${XTLLM_SPIRV})

function(xtllm_native_backend TARGET SOURCE)
  add_executable(${TARGET} ${SOURCE})
  add_dependencies(${TARGET} xtllm_shaders)
  target_include_directories(${TARGET} PRIVATE ${Vulkan_INCLUDE_DIRS})
  target_compile_definitions(${TARGET} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_options(${TARGET} PRIVATE $<$<CXX_COMPILER_ID:Clang,GNU>:-O2>)
  target_link_libraries(${TARGET} PRIVATE psapi)
  set_target_properties(${TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endfunction()

# The original four backends remain in the auto-detecting executable.  The
# three newer architectures retain separate shape-specialized translation
# units so consolidation cannot flatten their proven cache/kernel policies.
xtllm_native_backend(xtllm src/ovllm_longctx.cpp)
xtllm_native_backend(xtllm-qwen38-flash-next src/qwen38_flash_next.cpp)
xtllm_native_backend(xtllm-qwen3-coder-next src/m16_qwen35.cpp)
target_compile_definitions(xtllm-qwen3-coder-next PRIVATE XTLLM_QWEN3_CODER_NEXT)
xtllm_native_backend(xtllm-longcat-flash-lite src/longcat_flash_lite.cpp)
