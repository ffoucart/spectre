// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"  // IWYU pragma: keep
#include "Evolution/Actions/ComputeVolumeFluxes.hpp"
#include "Evolution/Actions/ComputeVolumeSources.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Evolution/DiscontinuousGalerkin/Observe.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/LimiterActions.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/Tags.hpp"
#include "Evolution/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "Evolution/EventsAndTriggers/EventsAndTriggers.hpp"  // IWYU pragma: keep
//#include "Evolution/Systems/RadiationTransport/M1Grey/Fluxes.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/M1Closure.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Sources.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "IO/Observer/Actions.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesLocalTimeStepping.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyFluxes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/LocalLaxFriedrichs.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/GotoAction.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/Actions/FinalTime.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"  // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"
#include "Time/StepChoosers/Cfl.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/Increase.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_ConstGlobalCache;
}  // namespace Parallel
/// \endcond

struct EvolutionMetavars {
  // To switch which analytic solution is evolved you only need to change the
  // line `using analytic_solution = ...;` and include the header file for the
  // solution.
  // TODO Analytic solution
  // using analytic_solution =
  // RelativisticEuler::Solutions::FishboneMoncriefDisk;

  // TODO : System
  // using system = grmhd::ValenciaDivClean::System<
  //   typename analytic_solution::equation_of_state_type>;
  static constexpr size_t thermodynamic_dim = system::thermodynamic_dim;
  using temporal_id = Tags::TimeId;
  static constexpr bool local_time_stepping = false;
  using analytic_solution_tag = OptionTags::AnalyticSolution<analytic_solution>;
  using boundary_condition_tag = analytic_solution_tag;
  using equation_of_state_tag = hydro::Tags::EquationOfState<
      typename analytic_solution_tag::type::equation_of_state_type>;
  // TODO : Analytic variables to be used, with different neutrino species
  // using analytic_variables_tags =
  // typename system::primitive_variables_tag::tags_list;
  using normal_dot_numerical_flux = OptionTags::NumericalFluxParams<
      dg::NumericalFluxes::LocalLaxFriedrichs<system>>;
  // Variables to limit
  // TODO Different neutrino species
  // using limiter = OptionTags::SlopeLimiterParams<SlopeLimiters::Minmod<
  // 3, tmpl::list<RadiationTransport::M1Grey::Tags::TildeE,
  // RadiationTransport::M1Grey::Tags::TildeS>>>>;

  // public for use by the Charm++ registration code
  using events =
      tmpl::list <
      dg::Events::Registrars::Observe<
          3, tmpl::append<db::get_variables_tags_list<system::variables_tag>,
                          analytic_variables_tags>>;
  using triggers = Triggers::time_triggers;

  using step_choosers =
      tmpl::list<StepChoosers::Registrars::Cfl<3, Frame::Inertial>,
                 StepChoosers::Registrars::Constant,
                 StepChoosers::Registrars::Increase>;

  using observed_reduction_data_tags =
      observers::collect_reduction_data_tags<Event<events>::creatable_classes>;

  using compute_rhs = tmpl::flatten<tmpl::list<
      Actions::ComputeVolumeFluxes,
      dg::Actions::SendDataForFluxes<EvolutionMetavars>,
      Actions::ComputeVolumeSources, Actions::ComputeTimeDerivative,
      dg::Actions::ImposeDirichletBoundaryConditions<EvolutionMetavars>,
      dg::Actions::ReceiveDataForFluxes<EvolutionMetavars>,
      tmpl::conditional_t<local_time_stepping, tmpl::list<>,
                          dg::Actions::ApplyFluxes>,
      Actions::RecordTimeStepperData>>;

  // TODO Probably will need variable fixing here...
  using update_variables =
      tmpl::flatten <
      tmpl::list<
          tmpl::conditional_t<local_time_stepping,
                              dg::Actions::ApplyBoundaryFluxesLocalTimeStepping,
                              tmpl::list<>>,
          Actions::UpdateU, SlopeLimiters::Actions::SendData<EvolutionMetavars>,
          SlopeLimiters::Actions::Limit<EvolutionMetavars>> /

          struct EvolvePhaseStart;
  using component_list = tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      DgElementArray<
          EvolutionMetavars, RadiationTransport::M1Grey::Actions::Initialize<3>,
          tmpl::flatten<tmpl::list<
              SelfStart::self_start_procedure<compute_rhs, update_variables>,
              Actions::Label<EvolvePhaseStart>, Actions::AdvanceTime,
              Actions::RunEventsAndTriggers, Actions::FinalTime,
              tmpl::conditional_t<local_time_stepping,
                                  Actions::ChangeStepSize<step_choosers>,
                                  tmpl::list<>>,
              compute_rhs, update_variables,
              Actions::Goto<EvolvePhaseStart>>>>>;

  using const_global_cache_tag_list =
      tmpl::list<analytic_solution_tag,
                 OptionTags::TypedTimeStepper<tmpl::conditional_t<
                     local_time_stepping, LtsTimeStepper, TimeStepper>>,
                 OptionTags::EventsAndTriggers<events, triggers>>;

  using domain_creator_tag = OptionTags::DomainCreator<3, Frame::Inertial>;

  struct ObservationType {};
  using element_observation_type = ObservationType;

  static constexpr OptionString help{
      "Evolve the M1 equations of radiation transport.\n\n"};

  enum class Phase { Initialization, RegisterWithObserver, Evolve, Exit };

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          EvolutionMetavars>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::RegisterWithObserver;
      case Phase::RegisterWithObserver:
        return Phase::Evolve;
      case Phase::Evolve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> an integral "
            "value?");
    }
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<EvolutionMetavars::step_choosers>>,
    &Parallel::register_derived_classes_with_charm<StepController>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
