"""Warp.fem single-element Newton-Raphson solve.

Run from Slicer after setting up slicer._warp2 dict with geometry info.

Two solve modes:
  - Force-driven: apply external force at top corner, solve for equilibrium
  - Displacement-driven: prescribe displacement at top corner via BSR projection
"""

import sys
import numpy
import warp as wp
import warp.fem as fem
import warp.examples.fem.utils as fem_example_utils

if 'warpfem_integrands' in sys.modules:
    del sys.modules['warpfem_integrands']
import warpfem_integrands as wfi


def _project_dof_bc(K, rhs, dof_idx):
    """Project a single DOF out of a BSR system (set row/col to identity, zero rhs)."""
    offsets = K.offsets.numpy()
    columns = K.columns.numpy()
    values = K.values.numpy()

    row_start = offsets[dof_idx]
    row_end = offsets[dof_idx + 1]

    # Zero this DOF's row, set diagonal to identity
    for k in range(row_start, row_end):
        col = columns[k]
        if col == dof_idx:
            values[k] = numpy.eye(3, dtype=values.dtype)
        else:
            values[k] = numpy.zeros((3, 3), dtype=values.dtype)

    # Zero the column entries for symmetry
    n_rows = len(offsets) - 1
    for row in range(n_rows):
        if row == dof_idx:
            continue
        rs = offsets[row]
        re = offsets[row + 1]
        for k in range(rs, re):
            if columns[k] == dof_idx:
                values[k] = numpy.zeros((3, 3), dtype=values.dtype)
                break

    K.values.assign(wp.array(values, dtype=K.values.dtype))

    rhs_np = rhs.numpy()
    rhs_np[dof_idx] = [0., 0., 0.]
    rhs.assign(wp.array(rhs_np, dtype=rhs.dtype))


def solve_warpfem_force(warp_info, force_vec, n_load_steps=5, alpha=1.0,
                        newton_tol=1e-3, max_newton=20):
    """Solve warp.fem element with force-driven Newton-Raphson.

    Only bottom face is fixed (Dirichlet BC).
    External force is applied at the top corner DOF.
    """
    geo = warp_info['geo']
    u_space = warp_info['u_space']
    bottom_dofs = warp_info['bottom_dofs']
    top_corner_dof = warp_info['top_corner_dof']
    n_dof = warp_info['n_dof']
    domain = warp_info['domain']

    E, nu = 1e4, 0.3
    mu_val = E / (2.0 * (1.0 + nu))
    lam_val = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    mat_space = fem.make_polynomial_space(geo, degree=0, discontinuous=True, dtype=float)
    mu_field = mat_space.make_field()
    lambda_field = mat_space.make_field()
    mu_field.dof_values.assign(wp.array([mu_val], dtype=float))
    lambda_field.dof_values.assign(wp.array([lam_val], dtype=float))

    u_field = u_space.make_field()
    test = fem.make_test(space=u_space, domain=domain)
    trial = fem.make_trial(space=u_space, domain=domain)

    boundary = fem.BoundarySides(geo)
    u_bd_test = fem.make_test(space=u_space, domain=boundary)
    u_bd_trial = fem.make_trial(space=u_space, domain=boundary)
    u_bd_matrix = fem.integrate(
        wfi.bottom_projector,
        fields={"u": u_bd_trial, "v": u_bd_test},
        assembly="nodal",
    )

    bc_dofs_set = set(bottom_dofs)
    log = []

    for load_step in range(n_load_steps):
        frac = (load_step + 1) / n_load_steps
        f_ext = force_vec * frac

        converged = False
        for it in range(max_newton):
            K = fem.integrate(
                wfi.tangent_stiffness,
                fields={"u": trial, "v": test, "u_cur": u_field,
                        "mu_f": mu_field, "lam_f": lambda_field},
            )
            f_int = fem.integrate(
                wfi.internal_force,
                fields={"v": test, "u_cur": u_field,
                        "mu_f": mu_field, "lam_f": lambda_field},
                output_dtype=wp.vec3d,
            )

            rhs_np = -f_int.numpy()
            rhs_np[top_corner_dof] += f_ext

            for di in bc_dofs_set:
                rhs_np[di] = [0., 0., 0.]

            res_norm = numpy.linalg.norm(rhs_np)

            rhs = wp.array(rhs_np, dtype=wp.vec3d)
            fem.project_linear_system(K, rhs, u_bd_matrix)

            du = wp.zeros(n_dof, dtype=wp.vec3d)
            cg_res, ncg = fem_example_utils.bsr_cg(
                K, b=rhs, x=du, quiet=True, tol=1e-10, max_iters=500
            )

            du_np = du.numpy() * alpha
            for di in bc_dofs_set:
                du_np[di] = [0., 0., 0.]

            du_norm = numpy.linalg.norm(du_np)

            if numpy.isnan(du_norm):
                log.append(f"  Load {frac:.0%} iter {it}: NaN! res={res_norm:.3e}")
                break

            u_vals = u_field.dof_values.numpy()
            u_vals += du_np.astype(numpy.float32)
            u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))

            if it < 3 or it % 5 == 0:
                log.append(f"  Load {frac:.0%} iter {it}: |du|={du_norm:.3e} res={res_norm:.3e} CG={ncg}")

            if du_norm < newton_tol:
                converged = True
                log.append(f"  Load {frac:.0%}: converged iter {it}")
                break

        if not converged:
            max_d = numpy.max(numpy.linalg.norm(u_field.dof_values.numpy(), axis=1))
            log.append(f"  Load {frac:.0%}: NOT converged, max|u|={max_d:.2f}")

    u_final = u_field.dof_values.numpy()
    warp_max = numpy.max(numpy.linalg.norm(u_final, axis=1))
    log.append(f"Final max disp: {warp_max:.3f} mm")

    return {
        'displacements': u_final,
        'u_field': u_field,
        'log': log,
    }


def solve_warpfem_disp(warp_info, prescribed_disp, n_load_steps=5, alpha=1.0,
                       newton_tol=1e-3, max_newton=20):
    """Solve with prescribed displacement.

    Bottom face via fem.project_linear_system, top corner via manual BSR projection.
    """
    geo = warp_info['geo']
    u_space = warp_info['u_space']
    bottom_dofs = warp_info['bottom_dofs']
    top_corner_dof = warp_info['top_corner_dof']
    n_dof = warp_info['n_dof']
    domain = warp_info['domain']

    E, nu = 1e4, 0.3
    mu_val = E / (2.0 * (1.0 + nu))
    lam_val = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    mat_space = fem.make_polynomial_space(geo, degree=0, discontinuous=True, dtype=float)
    mu_field = mat_space.make_field()
    lambda_field = mat_space.make_field()
    mu_field.dof_values.assign(wp.array([mu_val], dtype=float))
    lambda_field.dof_values.assign(wp.array([lam_val], dtype=float))

    u_field = u_space.make_field()
    test = fem.make_test(space=u_space, domain=domain)
    trial = fem.make_trial(space=u_space, domain=domain)

    boundary = fem.BoundarySides(geo)
    u_bd_test = fem.make_test(space=u_space, domain=boundary)
    u_bd_trial = fem.make_trial(space=u_space, domain=boundary)
    u_bd_matrix = fem.integrate(
        wfi.bottom_projector,
        fields={"u": u_bd_trial, "v": u_bd_test},
        assembly="nodal",
    )

    bc_dofs_set = set(bottom_dofs)
    bc_dofs_set.add(top_corner_dof)
    log = []

    for load_step in range(n_load_steps):
        frac = (load_step + 1) / n_load_steps
        target = prescribed_disp * frac

        # Set prescribed displacement
        u_vals = u_field.dof_values.numpy()
        u_vals[top_corner_dof] = target.astype(numpy.float32)
        u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))

        converged = False
        for it in range(max_newton):
            K = fem.integrate(
                wfi.tangent_stiffness,
                fields={"u": trial, "v": test, "u_cur": u_field,
                        "mu_f": mu_field, "lam_f": lambda_field},
            )
            f_int = fem.integrate(
                wfi.internal_force,
                fields={"v": test, "u_cur": u_field,
                        "mu_f": mu_field, "lam_f": lambda_field},
                output_dtype=wp.vec3d,
            )

            rhs_np = -f_int.numpy()
            for di in bc_dofs_set:
                rhs_np[di] = [0., 0., 0.]
            rhs = wp.array(rhs_np, dtype=wp.vec3d)

            # Project bottom face BC
            fem.project_linear_system(K, rhs, u_bd_matrix)

            # Project top corner BC (manual BSR modification)
            _project_dof_bc(K, rhs, top_corner_dof)

            du = wp.zeros(n_dof, dtype=wp.vec3d)
            cg_res, ncg = fem_example_utils.bsr_cg(
                K, b=rhs, x=du, quiet=True, tol=1e-10, max_iters=500
            )

            du_np = du.numpy() * alpha
            du_norm = numpy.linalg.norm(du_np)

            if numpy.isnan(du_norm):
                log.append(f"  Load {frac:.0%} iter {it}: NaN!")
                break

            u_vals = u_field.dof_values.numpy()
            u_vals += du_np.astype(numpy.float32)
            # Re-enforce top corner exactly
            u_vals[top_corner_dof] = target.astype(numpy.float32)
            u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))

            if it < 3 or it % 5 == 0:
                log.append(f"  Load {frac:.0%} iter {it}: |du|={du_norm:.3e} CG={ncg}")

            if du_norm < newton_tol:
                converged = True
                log.append(f"  Load {frac:.0%}: converged iter {it}")
                break

        if not converged:
            max_d = numpy.max(numpy.linalg.norm(u_field.dof_values.numpy(), axis=1))
            log.append(f"  Load {frac:.0%}: NOT converged, max|u|={max_d:.2f}")

    u_final = u_field.dof_values.numpy()
    warp_max = numpy.max(numpy.linalg.norm(u_final, axis=1))
    log.append(f"Final max disp: {warp_max:.3f} mm")

    return {
        'displacements': u_final,
        'u_field': u_field,
        'log': log,
    }
