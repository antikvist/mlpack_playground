option (ENABLE_MIMALLOC "Enable mimalloc allocator" ${ENABLE_LIBRARIES})

if (NOT EXISTS "${CMAKE_SOURCE_DIR}/contrib/mimalloc/include/mimalloc.h")
   message (WARNING "submodule contrib/mimalloc is missing. to fix try run: \n git submodule update --init --recursive")
endif ()

set (LIBRARY_DIR ${CMAKE_SOURCE_DIR}/contrib/mimalloc)

set (SRCS
    ${LIBRARY_DIR}/src/stats.c
    ${LIBRARY_DIR}/src/random.c
    ${LIBRARY_DIR}/src/os.c
    ${LIBRARY_DIR}/src/bitmap.c
    ${LIBRARY_DIR}/src/arena.c
    ${LIBRARY_DIR}/src/segment-cache.c
    ${LIBRARY_DIR}/src/segment.c
    ${LIBRARY_DIR}/src/page.c
    ${LIBRARY_DIR}/src/alloc.c
    ${LIBRARY_DIR}/src/alloc-aligned.c
    ${LIBRARY_DIR}/src/alloc-posix.c
    ${LIBRARY_DIR}/src/heap.c
    ${LIBRARY_DIR}/src/options.c
    ${LIBRARY_DIR}/src/init.c
)

if (ENABLE_MIMALLOC)
    set (MIMALLOC_INCLUDE_DIR ${SOURCE_DIR}/contrib/mimalloc/include)
    set (USE_MIMALLOC 1)
    set (MIMALLOC_LIBRARY mimalloc-static)
    message (STATUS "Using mimalloc: ${MIMALLOC_INCLUDE_DIR} : ${MIMALLOC_LIBRARY}")

    add_library(_mimalloc ${SRCS})
    target_include_directories(_mimalloc PUBLIC "${CMAKE_SOURCE_DIR}/contrib/mimalloc/include")
    add_library(ch_contrib::mimalloc ALIAS _mimalloc)
endif ()
