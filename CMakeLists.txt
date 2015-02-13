cmake_minimum_required(VERSION 2.8.9)

project(SlicerTissue)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://www.example.com/Slicer/Extensions/SlicerTissue")
set(EXTENSION_CATEGORY "Simulation")
set(EXTENSION_CONTRIBUTORS "Steve Pieper (Isomics, Inc.), Anonymous")
set(EXTENSION_DESCRIPTION "Modules to perform soft tissue simulation and analysis.")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/SlicerTissue.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/SlicerTissue/Screenshots/1.png")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(TissueSimulation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
