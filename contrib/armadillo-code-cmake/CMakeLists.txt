option(ENABLE_ARMADILLO "Enable armadillo" ${ENABLE_LIBRARIES})

if (NOT ENABLE_ARMADILLO)
    message(STATUS "Not using armadillo")
    return()
endif()

set (LIBRARY_DIR "${CMAKE_SOURCE_DIR}/contrib/armadillo-code")

add_library(_armadillo INTERFACE)
target_include_directories(_armadillo SYSTEM BEFORE INTERFACE "${CMAKE_SOURCE_DIR}/contrib/armadillo-code/include" "${CMAKE_SOURCE_DIR}/contrib/armadillo-code/include/armadillo_bits")

add_library(ch_contrib::armadillo ALIAS _armadillo)