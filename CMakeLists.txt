cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME TPGVVCPartDatabase)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# *******************************************
# *********** GEGELATI LIBRARY **************
# *******************************************

# Find GEGELATI : find the gegelatilib-x.y.z (most recent : 0.5.1) folder in the lib directory
if(WIN32)
    set(LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
    MESSAGE("LIBS_DIR : ${LIBS_DIR}")
    file(GLOB GEGELATI_ROOT_DIR "${LIBS_DIR}/gegelatilib-[\\.|0-9]*")
    MESSAGE("GEGELATI_ROOT_DIR : ${GEGELATI_ROOT_DIR}")
    set(GEGELATI_DIR ${GEGELATI_ROOT_DIR}/CMake)
    MESSAGE("GEGELATI_DIR : ${GEGELATI_DIR}")
endif()
find_package(GEGELATI)

# Copy GEGELATI's dll
if (WIN32)
    file(GLOB
         GEGELATI_DLL
         ${GEGELATI_ROOT_DIR}/bin/*.dll
    )
    MESSAGE("Copy GEGELATI DLLs into ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    if(NOT ${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
        file(COPY ${GEGELATI_DLL} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    else()
        file(COPY ${GEGELATI_DLL} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug)
        file(COPY ${GEGELATI_DLL} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release)
    endif()
endif()

# For Linux 
if(NOT ${CMAKE_GENERATOR} MATCHES "Visual Studio.*")

	# Link with pthread
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

	# Set to 1 to activate debug compilation (0 for release)
	#set(DEBUG 0)

	if(${DEBUG})
			MESSAGE("Generate Debug project")
			set(CMAKE_BUILD_TYPE Debug)
			set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Debug)
			set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -pg -Wall")
	else()
			MESSAGE("Generate Release project")
			set(CMAKE_BUILD_TYPE Release)
			set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Release)
			set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
	endif()
	#add libmath during non visual studio builds
	set(CMAKE_EXTRA_LIB m)
else()
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# *******************************************
# ************ PROJECT CONFIG ***************
# *******************************************

include_directories(${GEGELATI_INCLUDE_DIRS})
add_executable(${PROJECT_NAME}
               ../src/main.cpp
               ../src/PartCU.cpp
               ../include/PartCU.h
               ../params.json
)
target_link_libraries(${PROJECT_NAME} ${GEGELATI_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE ROOT_DIR="${CMAKE_SOURCE_DIR}")
