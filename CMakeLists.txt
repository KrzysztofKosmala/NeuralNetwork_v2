
#Minimalna wersja cmake konieczna do zbudowania tego projektu
cmake_minimum_required(VERSION 3.4)

#Parametry konfiguracyjne platformy programistycznej i flagi kompilatora
set(CMAKE_CXX_COMPILER g++)#g++ jako kompilator C++
set(CMAKE_C_COMPILER gcc)#gcc jako kompilator C
set(CMAKE_CXX_STANDARD 11)#Standard C++ 11, bez tego parametru g++ użyje wartości domyślnej. Dla g++ od wersji 6 jest to C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)#Określony wyżej standard C++ jest wymagany

#Profil debug i maksymalny poziom debug info g3 oraz format debug info na dwarf-4 (musi być gdb w wersji 7). BUILD_TYPE może być ustawiony na:
#empty,
#Debug,
#Release,
#RelWithDebInfo
#MinSizeRel
#Ustawiono dodatkowo flagi debug generujące informację dla debuggera
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "-g3 -gdwarf-4")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -gdwarf-4")
#Ustaw nazwę projektu i język na C++
project(zadanie2)
enable_language(CXX)

set(ARMADILLO_LIBRARY "C:\\armadillo\\lib")
set(ARMADILLO_INCLUDE_DIR "C:\\armadillo\\include")
find_package(Armadillo REQUIRED)


if(Armadillo_FOUND)

    message(STATUS "ARMADILLO_INCLUDE_DIRS: ${ARMADILLO_INCLUDE_DIR}")
    message(STATUS "ARMADILLO_VERSION_MAJOR")
    message(STATUS "ARMADILLO_VERSION_STRING")
    message(STATUS "ARMADILLO_VERSION_NAME")


    include_directories(${ARMADILLO_INCLUDE_DIRS})

endif()

find_package(Boost 1.57.0 COMPONENTS system filesystem REQUIRED)

if(Boost_FOUND)

    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")

    include_directories(${Boost_INCLUDE_DIRS})

endif()
#target_link_libraries(${Boost_LIBRARIES})
#W specjalnym pliku compile_commands.json można podejrzeć wszystkie operacje kompilatora wraz z parametrami,
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE TRUE)
#uwzględnij podprojek znajdujący się w podkatalogu program. Musi być w nim pliki CMakeLists.txt
add_subdirectory(program)
add_subdirectory(library)
#Włączenie testów powoduje pojawienie się celu test w makefiles
enable_testing()