include(FetchContent)

#install Eigen library
FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/qwertyquerty/eigen.git
)

FetchContent_MakeAvailable(eigen)

FILE(GLOB_RECURSE RAY_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*")
set(RAY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include)

message(STATUS "Building rayEngine")

add_library(rayEngine ${RAY_SRC})
target_include_directories(rayEngine PUBLIC ${RAY_HEADERS})
target_link_libraries(rayEngine PUBLIC eigen)
