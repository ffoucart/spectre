# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY DomainTimeDependence)

set(LIBRARY_SOURCES
  CubicScale.cpp
  None.cpp
  RegisterDerivedWithCharm.cpp
  UniformTranslation.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE CoordinateMaps
  INTERFACE DataStructures
  INTERFACE Domain
  INTERFACE ErrorHandling
  INTERFACE FunctionsOfTime
  INTERFACE Utilities
  )
