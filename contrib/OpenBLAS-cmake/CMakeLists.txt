# enable_language(Fortran)

option(ENABLE_OPENBLAS "Enable openblas" ${ENABLE_LIBRARIES})

if (NOT ENABLE_OPENBLAS)
    message(STATUS "Not using openblas")
    return()
endif()

# ENABLE_LANGUAGE(Fortran)
# enable_language(fortran)

set (LIBRARY_DIR "${CMAKE_SOURCE_DIR}/contrib/OpenBLAS")

set(SRC
    "${LIBRARY_DIR}/lapack-netlib/CBLAS/src/cblas_dsyrk.c"
)

set (HDRS 
    "${LIBRARY_DIR}/lapack-netlib/CBLAS/include/cblas.h"
    "${LIBRARY_DIR}/lapack-netlib/CBLAS/include/cblas_f77.h"
)

add_library(_openblas ${SRC} ${HDRS})
# message(
#   FATAL ${PROJECT_BINARY_DIR}
# )

target_include_directories(_openblas SYSTEM PUBLIC "${LIBRARY_DIR}" )
set_target_properties(_openblas PROPERTIES COMPILE_FLAGS
      -DNOFORTRAN=1)
add_library(ch_contrib::openblas ALIAS _openblas)
