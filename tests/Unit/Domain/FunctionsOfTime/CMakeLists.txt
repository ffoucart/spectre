# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_FunctionsOfTime")

set(LIBRARY_SOURCES
  Test_PiecewisePolynomial.cpp
  Test_SettleToConstant.cpp
  )

add_test_library(
  ${LIBRARY}
  "Domain/FunctionsOfTime"
  "${LIBRARY_SOURCES}"
  "DataStructures;FunctionsOfTime"
  )
