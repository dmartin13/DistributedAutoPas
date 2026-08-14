add_executable(minimal_mpi minimal_mpi.cpp)
target_link_libraries(minimal_mpi PRIVATE distributed_autopas)

option(
    DISTRIBUTED_AUTOPAS_BUILD_MDFLEX_EXAMPLE
    "Build the md-flexible DistributedAutoPas example"
    ON
)

if(DISTRIBUTED_AUTOPAS_BUILD_MDFLEX_EXAMPLE)
    add_subdirectory(md-flexible)
endif()
