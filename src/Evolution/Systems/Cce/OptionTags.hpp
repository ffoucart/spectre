// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Options/Options.hpp"

namespace Cce {
/// \cond
class Interpolator;
/// \endcond

namespace OptionTags {

/// Option group
struct Cce {
  static constexpr OptionString help = {"Options for the Cce evolution system"};
};

struct LMax {
  using type = size_t;
  static constexpr OptionString help{
      "Maximum l value for spin-weighted spherical harmonics"};
  using group = Cce;
};

struct ObservationLMax {
  using type = size_t;
  static constexpr OptionString help{"Maximum l value for swsh output"};
  using group = Cce;
};

struct NumberOfRadialPoints {
  using type = size_t;
  static constexpr OptionString help{
      "Number of radial grid points in the spherical domain"};
  using group = Cce;
};

struct EndTime {
  using type = double;
  static constexpr OptionString help{"End time for the Cce Evolution."};
  static double default_value() noexcept {
    return std::numeric_limits<double>::infinity();
  }
  using group = Cce;
};

struct StartTime {
  using type = double;
  static constexpr OptionString help{
      "Cce Start time (default to earliest possible time)."};
  static double default_value() noexcept {
    return -std::numeric_limits<double>::infinity();
  }
  using group = Cce;
};

struct TargetStepSize {
  using type = double;
  static constexpr OptionString help{"Target time step size for Cce Evolution"};
  using group = Cce;
};

struct BoundaryDataFilename {
  using type = std::string;
  static constexpr OptionString help{
      "H5 file to read the wordltube data from."};
  using group = Cce;
};

struct H5LookaheadTimes {
  using type = size_t;
  static constexpr OptionString help{
      "Number of times steps from the h5 to cache each read."};
  static size_t default_value() noexcept { return 200; }
  using group = Cce;
};

struct H5Interpolator {
  using type = std::unique_ptr<Interpolator>;
  static constexpr OptionString help{
      "The interpolator for imported h5 worldtube data."};
  using group = Cce;
};

struct ScriInterpolationOrder {
  static std::string name() noexcept { return "ScriInterpOrder"; }
  using type = size_t;
  static constexpr OptionString help{"Order of time interpolation at scri+."};
  static size_t default_value() noexcept { return 5; }
  using group = Cce;
};
}  // namespace OptionTags

namespace InitializationTags {
/// An initialization tag that constructs a `WorldtubeDataManager` from options
struct H5WorldtubeBoundaryDataManager : db::SimpleTag {
  using type = WorldtubeDataManager;
  using option_tags =
      tmpl::list<OptionTags::LMax, OptionTags::BoundaryDataFilename,
                 OptionTags::H5LookaheadTimes, OptionTags::H5Interpolator>;

  static WorldtubeDataManager create_from_options(
      const size_t l_max, const std::string& filename,
      const size_t number_of_lookahead_times,
      const std::unique_ptr<intrp::SpanInterpolator>& interpolator) noexcept {
    return WorldtubeDataManager{
        std::make_unique<SpecWorldtubeH5BufferUpdater>(filename), l_max,
        number_of_lookahead_times, interpolator->get_clone()};
  }
};

}  // namespace InitializationTags
}  // namespace Cce