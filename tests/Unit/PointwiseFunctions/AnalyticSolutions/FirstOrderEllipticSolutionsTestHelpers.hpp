// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Elliptic/FirstOrderOperator.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Numeric.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace FirstOrderEllipticSolutionsTestHelpers {

namespace Tags {

template <typename Tag>
struct OperatorAppliedTo : db::SimpleTag, db::PrefixTag {
  using type = typename Tag::type;
  using tag = Tag;
};

}  // namespace Tags

namespace detail {

template <typename System,
          typename FluxesArgs =
              tmpl::transform<typename System::fluxes::argument_tags,
                              tmpl::bind<db::const_item_type, tmpl::_1>>,
          typename FluxesArgsIndices =
              std::make_index_sequence<tmpl::size<FluxesArgs>::value>,
          typename SourcesArgs =
              tmpl::transform<typename System::sources::argument_tags,
                              tmpl::bind<db::const_item_type, tmpl::_1>>,
          typename SourcesArgsIndices =
              std::make_index_sequence<tmpl::size<SourcesArgs>::value>>
struct ExpandTuplesImpl;

template <typename System, typename... FluxesArgs, size_t... FluxesArgsIndices,
          typename... SourcesArgs, size_t... SourcesArgsIndices>
struct ExpandTuplesImpl<System, tmpl::list<FluxesArgs...>,
                        std::index_sequence<FluxesArgsIndices...>,
                        tmpl::list<SourcesArgs...>,
                        std::index_sequence<SourcesArgsIndices...>> {
  static constexpr size_t volume_dim = System::volume_dim;
  using all_fields = db::get_variables_tags_list<typename System::fields_tag>;
  using primal_fields = typename System::primal_fields;
  using auxiliary_fields = typename System::auxiliary_fields;

  static auto first_order_fluxes(
      const Variables<all_fields>& vars,
      const typename System::fluxes& fluxes_computer,
      const std::tuple<FluxesArgs...>& fluxes_args) noexcept {
    return elliptic::first_order_fluxes<volume_dim, primal_fields,
                                        auxiliary_fields>(
        vars, fluxes_computer, get<FluxesArgsIndices>(fluxes_args)...);
  }

  static auto first_order_sources(
      const Variables<all_fields>& vars,
      const std::tuple<SourcesArgs...>& sources_args) noexcept {
    return elliptic::first_order_sources<primal_fields, auxiliary_fields,
                                         typename System::sources>(
        vars, get<SourcesArgsIndices>(sources_args)...);
  }
};

}  // namespace detail

/// \ingroup TestingFrameworkGroup
/// Test that the `solution` numerically solves the `System` on the given grid
/// for the given tolerance
template <typename System, typename SolutionType, typename... Maps,
          typename... FluxesArgs, typename... SourcesArgs>
void verify_solution(
    const SolutionType& solution,
    const typename System::fluxes& fluxes_computer,
    const Mesh<System::volume_dim>& mesh,
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, Maps...>
        coord_map,
    const double tolerance,
    const std::tuple<FluxesArgs...>& fluxes_args = std::tuple<>{},
    const std::tuple<SourcesArgs...>& sources_args = std::tuple<>{}) {
  using all_fields = db::get_variables_tags_list<typename System::fields_tag>;

  const size_t num_points = mesh.number_of_grid_points();
  const auto logical_coords = logical_coordinates(mesh);
  const auto inertial_coords = coord_map(logical_coords);
  const auto solution_fields = variables_from_tagged_tuple(
      solution.variables(inertial_coords, all_fields{}));

  // Apply operator to solution fields
  auto fluxes = detail::ExpandTuplesImpl<System>::first_order_fluxes(
      solution_fields, fluxes_computer, fluxes_args);
  auto div_fluxes = divergence(std::move(fluxes), mesh,
                               coord_map.inv_jacobian(logical_coords));
  auto sources = detail::ExpandTuplesImpl<System>::first_order_sources(
      solution_fields, sources_args);
  Variables<db::wrap_tags_in<Tags::OperatorAppliedTo, all_fields>>
      operator_applied_to_fields{num_points};
  elliptic::first_order_operator(make_not_null(&operator_applied_to_fields),
                                 std::move(div_fluxes), std::move(sources));

  // Set fixed sources to zero for auxiliary fields, and retrieve the fixed
  // sources for primal fields from the solution
  Variables<db::wrap_tags_in<::Tags::FixedSource, all_fields>> fixed_sources{
      num_points, 0.};
  fixed_sources.assign_subset(solution.variables(
      inertial_coords,
      db::wrap_tags_in<::Tags::FixedSource, typename System::primal_fields>{}));

  // Check error norms against the given tolerance
  tmpl::for_each<all_fields>([&operator_applied_to_fields, &fixed_sources,
                              &tolerance](auto field_tag_v) {
    using field_tag = tmpl::type_from<decltype(field_tag_v)>;
    const auto& operator_applied_to_field =
        get<Tags::OperatorAppliedTo<field_tag>>(operator_applied_to_fields);
    const auto& fixed_source =
        get<::Tags::FixedSource<field_tag>>(fixed_sources);
    double l2_error_square = 0.;
    double linf_error = 0.;
    for (size_t i = 0; i < operator_applied_to_field.size(); ++i) {
      const auto error = abs(operator_applied_to_field[i] - fixed_source[i]);
      l2_error_square += alg::accumulate(square(error), 0.) / error.size();
      const double component_linf_error = *alg::max_element(error);
      if (component_linf_error > linf_error) {
        linf_error = component_linf_error;
      }
    }
    const double l2_error =
        sqrt(l2_error_square / operator_applied_to_field.size());
    CAPTURE(db::tag_name<field_tag>());
    CAPTURE(l2_error);
    CAPTURE(linf_error);
    CHECK(l2_error < tolerance);
    CHECK(linf_error < tolerance);
  });
}

/*!
 * \ingroup TestingFrameworkGroup
 * Test that the `solution` numerically solves the `System` on the given grid
 * and that the discretization error decreases as expected for a smooth
 * function.
 *
 * \details We expect exponential convergence for a smooth solution, so the
 * tolerance is computed as
 *
 * \f{equation}
 * C_1 \exp{\left(-C_2 * N_\mathrm{points}\right)}
 * \f}
 *
 * where \f$C_1\f$ is the `tolerance_offset`, \f$C_2\f$ is the
 * `tolerance_scaling` and \f$N_\mathrm{points}\f$ is the number of grid points
 * per dimension.
 */
template <typename System, typename SolutionType,
          size_t Dim = System::volume_dim, typename... Maps>
void verify_smooth_solution(
    const SolutionType& solution,
    const typename System::fluxes& fluxes_computer,
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, Maps...>&
        coord_map,
    const double tolerance_offset, const double tolerance_scaling) {
  INFO("Verify smooth solution");
  for (size_t num_points = Spectral::minimum_number_of_points<
           Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>;
       num_points <=
       Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
       num_points++) {
    CAPTURE(num_points);
    const double tolerance =
        tolerance_offset * exp(-tolerance_scaling * num_points);
    CAPTURE(tolerance);
    FirstOrderEllipticSolutionsTestHelpers::verify_solution<System>(
        solution, fluxes_computer,
        Mesh<Dim>{num_points, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto},
        coord_map, tolerance);
  }
}

/*!
 * \ingroup TestingFrameworkGroup
 * Test that the `solution` numerically solves the `System` on the given grid
 * and that the discretization error decreases as a power law.
 *
 * \details The tolerance is computed as
 *
 * \f{equation}
 * C \left(N_\mathrm{points}\right)^{-p}
 * \f}
 *
 * where \f$C\f$ is the `tolerance_offset`, \f$p\f$ is the `tolerance_pow` and
 * \f$N_\mathrm{points}\f$ is the number of grid points per dimension.
 */
template <typename System, typename SolutionType,
          size_t Dim = System::volume_dim, typename... Maps>
void verify_solution_with_power_law_convergence(
    const SolutionType& solution,
    const typename System::fluxes& fluxes_computer,
    const domain::CoordinateMap<Frame::Logical, Frame::Inertial, Maps...>&
        coord_map,
    const double tolerance_offset, const double tolerance_pow) {
  INFO("Verify solution with power-law convergence");
  for (size_t num_points = Spectral::minimum_number_of_points<
           Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>;
       num_points <=
       Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
       num_points++) {
    CAPTURE(num_points);
    const double tolerance = tolerance_offset * pow(num_points, -tolerance_pow);
    CAPTURE(tolerance);
    FirstOrderEllipticSolutionsTestHelpers::verify_solution<System>(
        solution, fluxes_computer,
        Mesh<Dim>{num_points, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto},
        coord_map, tolerance);
  }
}

}  // namespace FirstOrderEllipticSolutionsTestHelpers
