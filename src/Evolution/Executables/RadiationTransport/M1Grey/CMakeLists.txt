# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  GeneralRelativitySolutions
  Hydro
  IO
  Informer
  LinearOperators
  M1Grey
  MathFunctions
  RelativisticEulerSolutions
  SlopeLimiters
  Time
  Utilities
  )

add_spectre_parallel_executable(
  EvolveM1Grey
  Evolution/Executables/RadiationTransport/M1Grey
  EvolutionMetavars
  "${LIBS_TO_LINK}"
  )
