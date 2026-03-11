"""Warp.fem integrands for single-element comparison.

These must live in a real .py file so inspect.getsource() works
(required by @fem.integrand decorator).
"""

import warp as wp
import warp.fem as fem


@fem.integrand
def pos_rhs_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    x = fem.position(domain, s)
    return wp.dot(x, v(s))


@fem.integrand
def pos_mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def internal_force(s: fem.Sample, v: fem.Field, u_cur: fem.Field,
                   mu_f: fem.Field, lam_f: fem.Field):
    F = wp.identity(n=3, dtype=float) + fem.grad(u_cur, s)
    J = wp.determinant(F)
    mu_val = mu_f(s)
    lam_val = lam_f(s)
    F_inv_T = wp.transpose(wp.inverse(F))
    cofactor = J * F_inv_T
    P = mu_val * F + (lam_val * (J - 1.0) - mu_val) * cofactor
    return wp.ddot(P, fem.grad(v, s))


@fem.integrand
def tangent_stiffness(s: fem.Sample, u: fem.Field, v: fem.Field,
                      u_cur: fem.Field, mu_f: fem.Field, lam_f: fem.Field):
    """Full Newton tangent for Neo-Hookean: dP/dF : delta_F

    P = mu*F + p*cof where p = lam*(J-1) - mu, cof = J*F^{-T}
    dP = mu*dF + lam*J^2*(F^{-T}:dF)*F^{-T}
         + p*J*[(F^{-T}:dF)*F^{-T} - F^{-T}*dF^T*F^{-T}]
    """
    grad_v = fem.grad(v, s)
    grad_du = fem.grad(u, s)
    F = wp.identity(n=3, dtype=float) + fem.grad(u_cur, s)
    J = wp.determinant(F)
    mu_val = mu_f(s)
    lam_val = lam_f(s)
    F_inv_T = wp.transpose(wp.inverse(F))

    p = lam_val * (J - 1.0) - mu_val

    # Term 1: mu * (grad_du : grad_v)
    t1 = mu_val * wp.ddot(grad_v, grad_du)

    # Term 2: (lam*J^2 + p*J) * (F^{-T}:grad_du) * (F^{-T}:grad_v)
    coeff2 = lam_val * J * J + p * J
    t2 = coeff2 * wp.ddot(F_inv_T, grad_du) * wp.ddot(F_inv_T, grad_v)

    # Term 3: -p*J * (F^{-T} @ grad_du^T @ F^{-T}) : grad_v
    temp = F_inv_T * wp.transpose(grad_du) * F_inv_T
    t3 = -p * J * wp.ddot(temp, grad_v)

    return t1 + t2 + t3


@fem.integrand
def bottom_projector(s: fem.Sample, domain: fem.Domain,
                     u: fem.Field, v: fem.Field):
    nor = fem.normal(domain, s)
    w = wp.max(0.0, -nor[2])  # bottom face normal is (0, 0, -1) i.e. z-min
    return w * wp.dot(u(s), v(s))
