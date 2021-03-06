# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
  CoordinateMaps
  DiscontinuousGalerkin
  DomainCreators
  GeneralRelativitySolutions
  GrMhdAnalyticData
  GrMhdSolutions
  Hydro
  IO
  Informer
  Limiters
  LinearOperators
  MathFunctions
  RelativisticEulerSolutions
  Time
  Utilities
  ValenciaDivClean
  )

function(add_grmhd_executable EXECUTABLE_NAME INITIAL_DATA)
  add_spectre_parallel_executable(
    "Evolve${EXECUTABLE_NAME}"
    EvolveValenciaDivClean
    Evolution/Executables/GrMhd/ValenciaDivClean
    "EvolutionMetavars<${INITIAL_DATA}>"
    "${LIBS_TO_LINK}"
    )
endfunction(add_grmhd_executable)

add_grmhd_executable(
  FishboneMoncriefDisk
  RelativisticEuler::Solutions::FishboneMoncriefDisk,Horizon
  )
target_link_libraries(
  EvolveFishboneMoncriefDisk
  PUBLIC
  ApparentHorizons
  Interpolation
  )

add_grmhd_executable(
  TovStar
  RelativisticEuler::Solutions::TovStar<gr::Solutions::TovSolution>
  )

add_grmhd_executable(
  AlfvenWave
  grmhd::Solutions::AlfvenWave
  )

add_grmhd_executable(
  BondiMichel
  grmhd::Solutions::BondiMichel
  )

add_grmhd_executable(
  KomissarovShock
  grmhd::Solutions::KomissarovShock
  )

add_grmhd_executable(
  ValenciaDivCleanSmoothFlow
  grmhd::Solutions::SmoothFlow
  )

add_grmhd_executable(
  BondiHoyleAccretion
  grmhd::AnalyticData::BondiHoyleAccretion
  )

add_grmhd_executable(
  CylindricalBlastWave
  grmhd::AnalyticData::CylindricalBlastWave
  )

add_grmhd_executable(
  MagneticFieldLoop
  grmhd::AnalyticData::MagneticFieldLoop
  )

add_grmhd_executable(
  MagneticRotor
  grmhd::AnalyticData::MagneticRotor
  )

add_grmhd_executable(
  MagnetizedFmDisk
  grmhd::AnalyticData::MagnetizedFmDisk
  )

add_grmhd_executable(
  OrszagTangVortex
  grmhd::AnalyticData::OrszagTangVortex
  )
