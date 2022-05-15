option(ENABLE_CEREAL "Enable cereal" ${ENABLE_LIBRARIES})

if (NOT ENABLE_CEREAL)
    message(STATUS "Not using cereal")
    return()
endif()

set (LIBRARY_DIR "${CMAKE_SOURCE_DIR}/contrib/cereal")



add_library(_cereal INTERFACE)
target_include_directories(_cereal SYSTEM BEFORE INTERFACE "${CMAKE_SOURCE_DIR}/contrib/cereal/include")

add_library(ch_contrib::cereal ALIAS _cereal)