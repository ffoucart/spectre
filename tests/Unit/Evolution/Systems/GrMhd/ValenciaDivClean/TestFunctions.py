# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def b_dot_v(magnetic_field, spatial_velocity, spatial_metric):
    return np.einsum("ab, ab", spatial_metric,
                     np.outer(magnetic_field, spatial_velocity))


def bsq(magnetic_field, spatial_metric):
    return np.einsum("ab, ab", spatial_metric,
                     np.outer(magnetic_field, magnetic_field))


def magnetic_field_one_form(magnetic_field, spatial_metric):
    return np.einsum("a, ia", magnetic_field, spatial_metric)


def p_star(pressure, b_dot_v, bsq, lorentz_factor):
    return pressure + 0.5 * (b_dot_v**2 + bsq / lorentz_factor**2)


def spatial_velocity_one_form(spatial_velocity, spatial_metric):
    return np.einsum("a, ia", spatial_velocity, spatial_metric)


def vsq(spatial_velocity, spatial_metric):
    return np.einsum("ab, ab", spatial_metric,
                     np.outer(spatial_velocity, spatial_velocity))


# Functions for testing ConservativeFromPrimitive.cpp
def tilde_d(rest_mass_density, specific_enthalpy, pressure, spatial_velocity,
            lorentz_factor, magnetic_field, sqrt_det_spatial_metric,
            spatial_metric, divergence_cleaning_field):
    return lorentz_factor * rest_mass_density * sqrt_det_spatial_metric


def tilde_tau(rest_mass_density, specific_enthalpy, pressure, spatial_velocity,
              lorentz_factor, magnetic_field, sqrt_det_spatial_metric,
              spatial_metric, divergence_cleaning_field):
    return ((rest_mass_density * specific_enthalpy * lorentz_factor**2 -
             pressure - rest_mass_density * lorentz_factor +
             0.5 * bsq(magnetic_field, spatial_metric) *
             (1.0 + vsq(spatial_velocity, spatial_metric)) -
             0.5 * b_dot_v(magnetic_field, spatial_velocity, spatial_metric)) *
            sqrt_det_spatial_metric)


def tilde_s(rest_mass_density, specific_enthalpy, pressure, spatial_velocity,
            lorentz_factor, magnetic_field, sqrt_det_spatial_metric,
            spatial_metric, divergence_cleaning_field):
    return ((spatial_velocity_one_form(spatial_velocity, spatial_metric) *
             (lorentz_factor**2 * specific_enthalpy * rest_mass_density + bsq(
                 magnetic_field, spatial_metric)) -
             magnetic_field_one_form(magnetic_field, spatial_metric) * b_dot_v(
                 magnetic_field, spatial_velocity, spatial_metric)) *
            sqrt_det_spatial_metric)


def tilde_b(rest_mass_density, specific_enthalpy, pressure, spatial_velocity,
            lorentz_factor, magnetic_field, sqrt_det_spatial_metric,
            spatial_metric, divergence_cleaning_field):
    return sqrt_det_spatial_metric * magnetic_field


def tilde_phi(rest_mass_density, specific_enthalpy, pressure, spatial_velocity,
              lorentz_factor, magnetic_field, sqrt_det_spatial_metric,
              spatial_metric, divergence_cleaning_field):
    return sqrt_det_spatial_metric * divergence_cleaning_field


# End functions for testing ConservativeFromPrimitive.cpp


# Functions for testing Fluxes.cpp
def tilde_d_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
                 sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                 pressure, spatial_velocity, lorentz_factor, magnetic_field):
    return tilde_d * (lapse * spatial_velocity - shift)


def tilde_tau_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
                   shift, sqrt_det_spatial_metric, spatial_metric,
                   inv_spatial_metric, pressure, spatial_velocity,
                   lorentz_factor, magnetic_field):
    b_dot_v_ = b_dot_v(magnetic_field, spatial_velocity, spatial_metric)
    return (sqrt_det_spatial_metric * lapse * p_star(
        pressure, b_dot_v_, bsq(magnetic_field, spatial_metric),
        lorentz_factor) * spatial_velocity + tilde_tau *
            (lapse * spatial_velocity - shift) - lapse * b_dot_v_ * tilde_b)


def tilde_s_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
                 sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                 pressure, spatial_velocity, lorentz_factor, magnetic_field):
    b_dot_v_ = b_dot_v(magnetic_field, spatial_velocity, spatial_metric)
    b_i = (magnetic_field_one_form(magnetic_field, spatial_metric)
           / lorentz_factor + spatial_velocity_one_form(
               spatial_velocity, spatial_metric) * lorentz_factor * b_dot_v_)
    result = np.outer(lapse * spatial_velocity - shift, tilde_s)
    result -= lapse / lorentz_factor * np.outer(tilde_b, b_i)
    result += (sqrt_det_spatial_metric * lapse * p_star(
        pressure, b_dot_v_, bsq(magnetic_field, spatial_metric),
        lorentz_factor) * np.identity(shift.size))
    return result


def tilde_b_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
                 sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
                 pressure, spatial_velocity, lorentz_factor, magnetic_field):
    result = np.outer(lapse * spatial_velocity - shift, tilde_b)
    result += lapse * inv_spatial_metric * tilde_phi
    result -= lapse * np.outer(tilde_b, spatial_velocity)
    return result


def tilde_phi_flux(tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse,
                   shift, sqrt_det_spatial_metric, spatial_metric,
                   inv_spatial_metric, pressure, spatial_velocity,
                   lorentz_factor, magnetic_field):
    return lapse * tilde_b - tilde_phi * shift


# End functions for testing Fluxes.cpp