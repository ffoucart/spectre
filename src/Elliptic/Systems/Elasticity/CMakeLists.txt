# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY Elasticity)

set(LIBRARY_SOURCES
  Equations.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE LinearOperators
  INTERFACE Spectral
  )
