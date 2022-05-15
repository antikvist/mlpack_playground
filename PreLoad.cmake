find_program(NINJA_PATH ninja)
if (NINJA_PATH)
    set(CMAKE_GENERATOR "Ninja" CACHE INTERNAL "" FORCE)
endif ()
