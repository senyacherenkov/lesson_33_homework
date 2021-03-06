cmake_minimum_required(VERSION 3.8)

project(realty VERSION 0.0.$ENV{TRAVIS_BUILD_NUMBER})

set(CMAKE_BUILD_TYPE DEBUG)
set( CMAKE_CXX_FLAGS "-fno-stack-protector" )

add_executable(rclst rclst.cpp utility.h utility.cpp)
add_executable(rclss rclss.cpp utility.h utility.cpp)

set_target_properties(rclst rclss PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  COMPILE_OPTIONS -Wpedantic -Wall -Wextra
)

target_link_libraries(rclst -lpthread -lopenblas -ldlib)
target_link_libraries(rclss -lpthread -lopenblas -ldlib)

install(TARGETS rclst RUNTIME DESTINATION bin)
install(TARGETS rclss RUNTIME DESTINATION bin)

set(CPACK_GENERATOR DEB)

set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

set(CPACK_PACKAGE_CONTACT senyacherenkov@yandex.ru)

include (CPack)
