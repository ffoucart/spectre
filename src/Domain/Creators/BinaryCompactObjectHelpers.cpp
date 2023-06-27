// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/BinaryCompactObjectHelpers.hpp"

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::creators::bco {
std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>>
create_grid_anchors(const std::array<double, 3>& center_a,
                    const std::array<double, 3>& center_b) {
  std::unordered_map<std::string, tnsr::I<double, 3, Frame::Grid>> result{};
  const auto add_center = [&result](const std::array<double, 3>& center,
                                    const std::string& suffix) {
    result["Center" + suffix] = tnsr::I<double, 3, Frame::Grid>{center};
  };

  return result;
}

TimeDependentMapOptions::TimeDependentMapOptions(
    double initial_time,
    std::optional<ExpansionMapOptions> expansion_map_options,
    std::optional<RotationMapOptions> rotation_options,
    std::optional<ShapeMapOptions<domain::ObjectLabel::A>> shape_options_A,
    std::optional<ShapeMapOptions<domain::ObjectLabel::B>> shape_options_B,
    const Options::Context& context)
    : initial_time_(initial_time),
      expansion_map_options_(expansion_map_options),
      rotation_options_(rotation_options),
      shape_options_A_(shape_options_A),
      shape_options_B_(shape_options_B) {
  if (not(expansion_map_options_.has_value() or rotation_options_.has_value() or
          shape_options_A_.has_value() or shape_options_B_.has_value())) {
    PARSE_ERROR(context,
                "Time dependent map options were specified, but all options "
                "were 'None'. If you don't want time dependent maps, specify "
                "'None' for the TimeDependentMapOptions. If you want time "
                "dependent maps, specify options for at least one map.");
  }

  const auto check_l_max = [&context](const auto& shape_option,
                                      const domain::ObjectLabel label) {
    if (shape_option.has_value() and shape_option.value().l_max <= 1) {
      PARSE_ERROR(context, "Initial LMax for object "
                               << label << " must be 2 or greater but is "
                               << shape_option.value().l_max << " instead.");
    }
  };

  check_l_max(shape_options_A_, domain::ObjectLabel::A);
  check_l_max(shape_options_B_, domain::ObjectLabel::B);
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
TimeDependentMapOptions::create_functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      result{};

  // Get existing function of time names that are used for the maps and assign
  // their initial expiration time to infinity (i.e. not expiring)
  std::unordered_map<std::string, double> expiration_times{
      {expansion_name, std::numeric_limits<double>::infinity()},
      {rotation_name, std::numeric_limits<double>::infinity()},
      {gsl::at(size_names, 0), std::numeric_limits<double>::infinity()},
      {gsl::at(size_names, 1), std::numeric_limits<double>::infinity()},
      {gsl::at(shape_names, 0), std::numeric_limits<double>::infinity()},
      {gsl::at(shape_names, 1), std::numeric_limits<double>::infinity()}};

  // If we have control systems, overwrite these expiration times with the ones
  // supplied by the control system
  for (const auto& [name, expr_time] : initial_expiration_times) {
    expiration_times[name] = expr_time;
  }

  // ExpansionMap FunctionOfTime for the function \f$a(t)\f$ in the
  // domain::CoordinateMaps::TimeDependent::CubicScale map
  if (expansion_map_options_.has_value()) {
    result[expansion_name] =
        std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
            initial_time_,
            std::array<DataVector, 3>{
                {{gsl::at(expansion_map_options_.value().initial_values, 0)},
                 {gsl::at(expansion_map_options_.value().initial_values, 1)},
                 {0.0}}},
            expiration_times.at(expansion_name));

    // ExpansionMap FunctionOfTime for the function \f$b(t)\f$ in the
    // domain::CoordinateMaps::TimeDependent::CubicScale map
    result[expansion_outer_boundary_name] =
        std::make_unique<FunctionsOfTime::FixedSpeedCubic>(
            1.0, initial_time_,
            expansion_map_options_.value().outer_boundary_velocity,
            expansion_map_options_.value().outer_boundary_decay_time);
  }

  // RotationMap FunctionOfTime for the rotation angles about each
  // axis.  The initial rotation angles don't matter as we never
  // actually use the angles themselves. We only use their derivatives
  // (omega) to determine map parameters. In theory we could determine
  // each initial angle from the input axis-angle representation, but
  // we don't need to.
  if (rotation_options_.has_value()) {
    result[rotation_name] = std::make_unique<
        FunctionsOfTime::QuaternionFunctionOfTime<3>>(
        initial_time_,
        std::array<DataVector, 1>{DataVector{1.0, 0.0, 0.0, 0.0}},
        std::array<DataVector, 4>{
            {{3, 0.0},
             {gsl::at(rotation_options_.value().initial_angular_velocity, 0),
              gsl::at(rotation_options_.value().initial_angular_velocity, 1),
              gsl::at(rotation_options_.value().initial_angular_velocity, 2)},
             {3, 0.0},
             {3, 0.0}}},
        expiration_times.at(rotation_name));
  }

  // Size and Shape FunctionOfTime for objects A and B
  for (size_t i = 0; i < shape_names.size(); i++) {
    if (i == 0 ? shape_options_A_.has_value() : shape_options_B_.has_value()) {
      const auto make_initial_size_values = [](const auto& lambda_options) {
        return std::array<DataVector, 4>{
            {{gsl::at(lambda_options.value().initial_size_values, 0)},
             {gsl::at(lambda_options.value().initial_size_values, 1)},
             {gsl::at(lambda_options.value().initial_size_values, 2)},
             {0.0}}};
      };

      size_t initial_l_max = 0;
      initial_l_max = i == 0 ? shape_options_A_.value().l_max
                             : shape_options_B_.value().l_max;
      const std::array<DataVector, 4> initial_size_values =
          i == 0 ? make_initial_size_values(shape_options_A_)
                 : make_initial_size_values(shape_options_B_);
      const DataVector shape_zeros{
          ylm::Spherepack::spectral_size(initial_l_max, initial_l_max), 0.0};

      result[gsl::at(shape_names, i)] =
          std::make_unique<FunctionsOfTime::PiecewisePolynomial<2>>(
              initial_time_,
              std::array<DataVector, 3>{shape_zeros, shape_zeros, shape_zeros},
              expiration_times.at(gsl::at(shape_names, i)));
      result[gsl::at(size_names, i)] =
          std::make_unique<FunctionsOfTime::PiecewisePolynomial<3>>(
              initial_time_, initial_size_values,
              expiration_times.at(gsl::at(size_names, i)));
    }
  }

  return result;
}

void TimeDependentMapOptions::build_maps(
    const std::array<std::array<double, 3>, 2>& centers,
    const std::array<std::optional<double>, 2>& object_inner_radii,
    const std::array<std::optional<double>, 2>& object_outer_radii,
    const double domain_outer_radius) {
  if (expansion_map_options_.has_value()) {
    expansion_map_ = CubicScaleMap{domain_outer_radius, expansion_name,
                                   expansion_outer_boundary_name};
  }
  if (rotation_options_.has_value()) {
    rotation_map_ = RotationMap3D{rotation_name};
  }
  // If either the size or shape map options are specific then we add the shape
  // map.
  for (size_t i = 0; i < 2; i++) {
    if (gsl::at(object_inner_radii, i).has_value() and
        gsl::at(object_outer_radii, i).has_value()) {
      if (not(i == 0 ? shape_options_A_.has_value()
                     : shape_options_B_.has_value())) {
        ERROR_NO_TRACE("Trying to build the shape map for object "
                       << get_output(i == 0 ? domain::ObjectLabel::A
                                            : domain::ObjectLabel::B)
                       << ", but no time dependent map options were specified "
                          "for that object.");
      }
      std::unique_ptr<domain::CoordinateMaps::ShapeMapTransitionFunctions::
                          ShapeMapTransitionFunction>
          transition_func = std::make_unique<
              domain::CoordinateMaps::ShapeMapTransitionFunctions::
                  SphereTransition>(gsl::at(object_inner_radii, i).value(),
                                    gsl::at(object_outer_radii, i).value());

      size_t initial_l_max = 0;
      initial_l_max = i == 0 ? shape_options_A_.value().l_max
                             : shape_options_B_.value().l_max;

      gsl::at(shape_maps_, i) =
          ShapeCoordMap{gsl::at(centers, i),     initial_l_max,
                        initial_l_max,           std::move(transition_func),
                        gsl::at(shape_names, i), gsl::at(size_names, i)};
    }
  }
}

TimeDependentMapOptions::MapType<Frame::Distorted, Frame::Inertial>
TimeDependentMapOptions::distorted_to_inertial_map(
    const bool include_distorted_map) const {
  if (include_distorted_map) {
    if (expansion_map_.has_value() and rotation_map_.has_value()) {
      return std::make_unique<ER_DI>(expansion_map_.value(),
                                     rotation_map_.value());
    } else if (expansion_map_.has_value()) {
      return std::make_unique<E_DI>(expansion_map_.value());
    } else if (rotation_map_.has_value()) {
      return std::make_unique<R_DI>(rotation_map_.value());
    } else {
      return nullptr;
    }
  } else {
    return nullptr;
  }
}

template <domain::ObjectLabel Object>
TimeDependentMapOptions::MapType<Frame::Grid, Frame::Distorted>
TimeDependentMapOptions::grid_to_distorted_map(
    const bool include_distorted_map) const {
  if (include_distorted_map) {
    const size_t index = get_index(Object);
    if (not gsl::at(shape_maps_, index).has_value()) {
      ERROR(
          "Requesting grid to distorted map with distorted frame but shape map "
          "options were not specified.");
    }
    return std::make_unique<S_GD>(gsl::at(shape_maps_, index).value());
  } else {
    return nullptr;
  }
}

template <domain::ObjectLabel Object>
TimeDependentMapOptions::MapType<Frame::Grid, Frame::Inertial>
TimeDependentMapOptions::grid_to_inertial_map(
    const bool include_distorted_map) const {
  if (include_distorted_map) {
    const size_t index = get_index(Object);
    if (not gsl::at(shape_maps_, index).has_value()) {
      ERROR(
          "Requesting grid to inertial map with distorted frame but shape map "
          "options were not specified.");
    }
    if (expansion_map_.has_value() and rotation_map_.has_value()) {
      return std::make_unique<SER_GI>(gsl::at(shape_maps_, index).value(),
                                      expansion_map_.value(),
                                      rotation_map_.value());
    } else if (expansion_map_.has_value()) {
      return std::make_unique<SE_GI>(gsl::at(shape_maps_, index).value(),
                                     expansion_map_.value());
    } else if (rotation_map_.has_value()) {
      return std::make_unique<SR_GI>(gsl::at(shape_maps_, index).value(),
                                     rotation_map_.value());
    } else {
      return std::make_unique<S_GI>(gsl::at(shape_maps_, index).value());
    }
  } else {
    if (expansion_map_.has_value() and rotation_map_.has_value()) {
      return std::make_unique<ER_GI>(expansion_map_.value(),
                                     rotation_map_.value());
    } else if (expansion_map_.has_value()) {
      return std::make_unique<E_GI>(expansion_map_.value());
    } else if (rotation_map_.has_value()) {
      return std::make_unique<R_GI>(rotation_map_.value());
    } else {
      return nullptr;
    }
  }
}

size_t TimeDependentMapOptions::get_index(const domain::ObjectLabel object) {
  ASSERT(object == domain::ObjectLabel::A or object == domain::ObjectLabel::B,
         "object label for TimeDependentMapOptions must be either A or B, not"
             << object);
  return object == domain::ObjectLabel::A ? 0_st : 1_st;
}

#define OBJECT(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template TimeDependentMapOptions::MapType<Frame::Grid, Frame::Distorted>  \
  TimeDependentMapOptions::grid_to_distorted_map<OBJECT(data)>(bool) const; \
  template TimeDependentMapOptions::MapType<Frame::Grid, Frame::Inertial>   \
  TimeDependentMapOptions::grid_to_inertial_map<OBJECT(data)>(bool) const;

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (domain::ObjectLabel::A, domain::ObjectLabel::B,
                         domain::ObjectLabel::None))

#undef OBJECT
#undef INSTANTIATE
}  // namespace domain::creators::bco
