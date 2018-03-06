// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions used by Charm++ to register classes/chares and entry
/// methods

#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>

#include "Parallel/CharmRegistration.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Main.hpp"
#include "Utilities/NoSuchType.hpp"

namespace Parallel {
namespace charmxx {
/// \cond
std::unique_ptr<RegistrationHelper>* charm_register_list = nullptr;
size_t charm_register_list_capacity = 0;
size_t charm_register_list_size = 0;

ReducerFunctions* charm_reducer_functions_list = nullptr;
size_t charm_reducer_functions_capacity = 0;
size_t charm_reducer_functions_size = 0;
std::unordered_map<size_t, CkReduction::reducerType> charm_reducer_functions{};
/// \endcond

/*!
 * \ingroup CharmExtensionsGroup
 * \brief Registers parallel components and their entry methods with Charm++
 */
void register_parallel_components() noexcept {
  static bool done_registration{false};
  if (done_registration) {
    return;  // LCOV_EXCL_LINE
  }
  done_registration = true;

  // Charm++ requires the order of registration to be the same across all
  // processors. To make sure this is satisfied regardless of any possible weird
  // behavior with static variable initialization we sort the list before
  // registering anything.
  // clang-tidy: do not user pointer arithmetic
  std::sort(charm_register_list,
            charm_register_list + charm_register_list_size,  // NOLINT
            [](const std::unique_ptr<RegistrationHelper>& a,
               const std::unique_ptr<RegistrationHelper>& b) {
              return a->name() < b->name();
            });

  // register chares and parallel components
  for (size_t i = 0;
       i < charm_register_list_size and i < charm_register_list_capacity; ++i) {
    // clang-tidy: do not use pointer arithmetic
    if (charm_register_list[i]->is_registering_chare()) {  // NOLINT
      charm_register_list[i]->register_with_charm();       // NOLINT
    }
  }
  // register reduction and simple actions, as well as receive_data
  for (size_t i = 0;
       i < charm_register_list_size and i < charm_register_list_capacity; ++i) {
    // clang-tidy: do not use pointer arithmetic
    if (not charm_register_list[i]->is_registering_chare()) {  // NOLINT
      charm_register_list[i]->register_with_charm();           // NOLINT
    }
  }
  delete[] charm_register_list;
}

/*!
 * \ingroup CharmExtensionsGroup
 * \brief Takes `charm_reducer_functions_list` and creates an unordered_map from
 * hashes of the functions to `CkReduction::reducerType`, which is used when
 * calling custom reductions.
 *
 * \note The reason `delete[]` isn't call here (or at all on
 * `charm_reducer_functions_list`) is because it has not been tested whether
 * that is safe to do in non-SMP builds.
 *
 * \see Parallel::contribute_to_reduction()
 */
void register_custom_reducer_functions() noexcept {
  for (size_t i = 0; i < charm_reducer_functions_size; ++i) {
    // clang-tidy: do not use pointer arithmetic
    charm_reducer_functions.emplace(
        std::hash<Parallel::charmxx::ReducerFunctions>{}(
            charm_reducer_functions_list[i]),                        // NOLINT
        CkReduction::addReducer(*charm_reducer_functions_list[i]));  // NOLINT
  }
}

/*!
 * \ingroup CharmExtensionsGroup
 * \brief Register all init_node and init_proc functions with Charm++
 *
 * \note The function that registers the custom reducers is the first init_node
 * function registered.
 */
void register_init_node_and_proc() noexcept {
  static bool done_registration{false};
  if (done_registration) {
    return;  // LCOV_EXCL_LINE
  }
  done_registration = true;
  // We explicitly register custom reducer functions first.
  _registerInitCall(register_custom_reducer_functions, 1);
  for (auto& init_node_func : charm_init_node_funcs) {
    _registerInitCall(*init_node_func, 1);
  }

  for (auto& init_proc_func : charm_init_proc_funcs) {
    _registerInitCall(*init_proc_func, 0);
  }
}
}  // namespace charmxx
}  // namespace Parallel

/*!
 * \ingroup CharmExtensionsGroup
 * \brief The function Charm++ calls to register reducers, chares, and entry
 * methods.
 *
 * The name of this function is crucial since it is called by Charm++ code.
 * Normally this function is generated by the `charmc` parser while generating
 * header files from interface files. The layer we built on Charm++ does not use
 * interface files directly and so we need to supply our `CkRegisterMainModule`
 * function. This function is also responsible for registering all parallel
 * components, all entry methods, and all custom reduction functions.
 */
extern "C" void CkRegisterMainModule() {
  (void)charmxx_main_component{
      Parallel::charmxx::MainChareRegistrationConstructor{}};
  Parallel::charmxx::register_init_node_and_proc();
  Parallel::charmxx::register_parallel_components();
}
