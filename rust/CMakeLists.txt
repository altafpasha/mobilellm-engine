cmake_minimum_required(VERSION 3.22)
project(mobilellm_rust LANGUAGES C CXX)

# Find Cargo executable
find_program(CARGO_BIN cargo)

if (NOT CARGO_BIN)
    message(FATAL_ERROR "MOBILELLM_ENABLE_RUST is ON, but 'cargo' binary was not found in PATH. Install Rust toolchain or configure -DMOBILELLM_ENABLE_RUST=OFF.")
endif()

message(STATUS "MobileLLM Rust: Found cargo at ${CARGO_BIN}")

# Determine build profile
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(RUST_PROFILE_DIR "debug")
    set(CARGO_BUILD_FLAGS "")
else()
    set(RUST_PROFILE_DIR "release")
    set(CARGO_BUILD_FLAGS "--release")
endif()

# Determine Rust target triple and linker configuration
set(RUST_TARGET_ENV "")
if (ANDROID)
    if (ANDROID_ABI STREQUAL "arm64-v8a")
        set(RUST_TARGET "aarch64-linux-android")
    elseif (ANDROID_ABI STREQUAL "x86_64")
        set(RUST_TARGET "x86_64-linux-android")
    elseif (ANDROID_ABI STREQUAL "armeabi-v7a")
        set(RUST_TARGET "armv7-linux-androideabi")
    else()
        message(FATAL_ERROR "Unsupported Android ABI for Rust: ${ANDROID_ABI}")
    endif()

    list(APPEND CARGO_BUILD_FLAGS "--target" "${RUST_TARGET}")
    set(RUST_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/${RUST_TARGET}/${RUST_PROFILE_DIR}")

    # Configure NDK Clang cross-linker
    string(TOUPPER "${RUST_TARGET}" RUST_TARGET_UPPER)
    string(REPLACE "-" "_" RUST_TARGET_UPPER "${RUST_TARGET_UPPER}")
    set(LINKER_VAR "CARGO_TARGET_${RUST_TARGET_UPPER}_LINKER")
    set(RUST_TARGET_ENV "${LINKER_VAR}=${CMAKE_C_COMPILER}")

    # Enforce 16 KB page-size linker requirements for Android 15
    set(ENV{RUSTFLAGS} "-Clink-arg=-Wl,-z,max-page-size=16384")
else()
    set(RUST_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/${RUST_PROFILE_DIR}")
endif()

# Library file names
if (WIN32)
    set(RUST_LIB_STATIC "${RUST_OUT_DIR}/mobilellm_rust_ffi.lib")
    set(RUST_LIB_SHARED "${RUST_OUT_DIR}/mobilellm_rust_ffi.dll")
elseif (APPLE)
    set(RUST_LIB_STATIC "${RUST_OUT_DIR}/libmobilellm_rust_ffi.a")
    set(RUST_LIB_SHARED "${RUST_OUT_DIR}/libmobilellm_rust_ffi.dylib")
else()
    set(RUST_LIB_STATIC "${RUST_OUT_DIR}/libmobilellm_rust_ffi.a")
    set(RUST_LIB_SHARED "${RUST_OUT_DIR}/libmobilellm_rust_ffi.so")
endif()

# Custom command to invoke Cargo build for mobilellm-ffi crate
add_custom_command(
    OUTPUT ${RUST_LIB_STATIC}
    COMMAND ${CMAKE_COMMAND} -E env ${RUST_TARGET_ENV}
            ${CARGO_BIN} build
            --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
            --package mobilellm-ffi
            ${CARGO_BUILD_FLAGS}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Compiling MobileLLM Rust workspace (${RUST_PROFILE_DIR})"
    VERBATIM
)

add_custom_target(mobilellm_rust_cargo_target ALL
    DEPENDS ${RUST_LIB_STATIC}
)

# Imported static library target
add_library(mobilellm_rust_ffi STATIC IMPORTED GLOBAL)
add_dependencies(mobilellm_rust_ffi mobilellm_rust_cargo_target)

set_target_properties(mobilellm_rust_ffi PROPERTIES
    IMPORTED_LOCATION "${RUST_LIB_STATIC}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/crates/mobilellm-ffi/include"
)

message(STATUS "MobileLLM Rust: Target mobilellm_rust_ffi configured (${RUST_LIB_STATIC})")
