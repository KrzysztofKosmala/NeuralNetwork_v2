cmake_minimum_required(VERSION 3.4)

project(Program)

#specify set of project's source files
set(SOURCE_FILES
        src/main.cpp
        )
#add following files to executable errortxt 'Program'
add_executable(Program ${SOURCE_FILES})

#add libraries from 'Library' project
target_link_libraries(Program Library)

#add header files from 'include' directory of main project
target_include_directories(Program PUBLIC include)
