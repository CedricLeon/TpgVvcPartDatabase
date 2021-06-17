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

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -g")

	#add libmath during non visual studio builds
	set(CMAKE_EXTRA_LIB m)
else()
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# *******************************************
# ************ PROJECT CONFIG ***************
# *******************************************

# Add definitions for testing purposes (-DTESTING=1)
if(${TESTING})
    MESSAGE("Testing mode")
    add_definitions(-DNO_CONSOLE_CONTROL=1)
endif()

# Include GEGELATI
include_directories(${GEGELATI_INCLUDE_DIRS})

# ************ DEFAULT SOLUTION (F1) ***************
# Create default executable (from the classification environment, Cf. 4EIIS8)
set(DEFAULT_EXE_NAME ${PROJECT_NAME}_classificationEnv)
add_executable(${DEFAULT_EXE_NAME}
               ../src/classification/classifTPG.cpp
               ../src/classification/PartCU.cpp
               ../include/classification/PartCU.h
               ../params.json
)
# Add GEGELATI and CMAKE_SOURCE_DIR
target_link_libraries(${DEFAULT_EXE_NAME} ${GEGELATI_LIBRARIES})
target_compile_definitions(${DEFAULT_EXE_NAME} PRIVATE ROOT_DIR="${CMAKE_SOURCE_DIR}")

# ************ BINARY TPGs TRAINING SOLUTION ***************
# Create a second executable for the training of one binary TPG (from the default environment, Cf. stage)
set(BINARYTPG_EXE_NAME ${PROJECT_NAME}_binary)
add_executable(${BINARYTPG_EXE_NAME}
        ../src/binary/binaryTPGs.cpp
        ../src/binary/defaultBinaryEnv.cpp
        ../include/binary/defaultBinaryEnv.h
        ../src/binary/classBinaryEnv.cpp
        ../include/binary/classBinaryEnv.h
        ../params.json
)
# Add GEGELATI and CMAKE_SOURCE_DIR
target_link_libraries(${BINARYTPG_EXE_NAME} ${GEGELATI_LIBRARIES})
target_compile_definitions(${BINARYTPG_EXE_NAME} PRIVATE ROOT_DIR="${CMAKE_SOURCE_DIR}")

# ************ BINARY TPGs INFERENCE SOLUTION ***************
# This executable load 6 binary TPGs (each specialized in a different action) and compute their mean score on a database
set(INFERENCEBINARYTPG_EXE_NAME ${PROJECT_NAME}_inferenceBinary)
add_executable(${INFERENCEBINARYTPG_EXE_NAME}
        ../src/binary/inferenceBinaryTPGs.cpp
        ../src/binary/defaultBinaryEnv.cpp
        ../include/binary/defaultBinaryEnv.h
        ../src/binary/classBinaryEnv.cpp
        ../include/binary/classBinaryEnv.h
        )
# Add GEGELATI and CMAKE_SOURCE_DIR
target_link_libraries(${INFERENCEBINARYTPG_EXE_NAME} ${GEGELATI_LIBRARIES})
target_compile_definitions(${INFERENCEBINARYTPG_EXE_NAME} PRIVATE ROOT_DIR="${CMAKE_SOURCE_DIR}")

# ************ FEATURES SOLUTION (F1) ***************
# This executable trains a TPG in the classification environment (6 actions) from the CU pre-calculated features
set(FEATURES_EXE_NAME ${PROJECT_NAME}_featuresEnv)
add_executable(${FEATURES_EXE_NAME}
        ../src/features/featuresTPG.cpp
        ../src/features/FeaturesEnv.cpp
        ../include/features/FeaturesEnv.h
        ../params.json
        )
# Add GEGELATI and CMAKE_SOURCE_DIR
target_link_libraries(${FEATURES_EXE_NAME} ${GEGELATI_LIBRARIES})
target_compile_definitions(${FEATURES_EXE_NAME} PRIVATE ROOT_DIR="${CMAKE_SOURCE_DIR}")

