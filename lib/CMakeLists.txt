add_library(
    greetings
    hello.h
    hello.cpp
)

target_include_directories(greetings PUBLIC "${CMAKE_SOURCE_DIR}/lib")

if (TARGET ch_contrib::mimalloc)
    message("if (TARGET ch_contrib::mimalloc)")
    target_link_libraries(greetings ch_contrib::mimalloc)
endif()


if (TARGET ch_contrib::armadillo)
    message("if (TARGET ch_contrib::armadillo)")
    target_link_libraries(greetings ch_contrib::armadillo)
endif()

if (TARGET ch_contrib::cereal)
    message("if (TARGET ch_contrib::cereal)")
    target_link_libraries(greetings ch_contrib::cereal)
endif()

if (TARGET ch_contrib::boost_serialization)
    message("if (TARGET ch_contrib::boost_serialization)")
    target_link_libraries(greetings ch_contrib::boost_serialization)
endif()

if (TARGET ch_contrib::openblas)
    message("if (TARGET ch_contrib::openblas)")
    target_link_libraries(greetings ch_contrib::openblas)
endif()

if (TARGET ch_contrib::mlpack)
    message("if (TARGET ch_contrib::mlpack)")
    target_link_libraries(greetings ch_contrib::mlpack)
endif()

# if (USE_MIMALLOC)
#     message("if (USE_MIMALLOC)")
#     target_include_directories (clickhouse_common_io SYSTEM BEFORE PUBLIC ${MIMALLOC_INCLUDE_DIR})
#     target_link_libraries (clickhouse_common_io PRIVATE ${MIMALLOC_LIBRARY})
# endif ()
