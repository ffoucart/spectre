# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Cce)

set(LIBRARY_SOURCES
  BoundaryData.cpp
  Equations.cpp
  GaugeTransformBoundaryData.cpp
  InitializeCce.cpp
  LinearOperators.cpp
  LinearSolve.cpp
  PrecomputeCceDependencies.cpp
  ReadBoundaryDataH5.cpp
  ReducedWorldtubeModeRecorder.cpp
  SpecBoundaryData.cpp
  ScriPlusValues.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  LinearSolver
  PUBLIC
  DataStructures
  Domain
  ErrorHandling
  IO
  LinearOperators
  Spectral
  Utilities
  )
