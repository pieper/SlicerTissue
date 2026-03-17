# NewtonTissue

> **Note:** This project is new and experimental. It was developed with the assistance of AI coding tools. Further testing and validation are required before using it in production or clinical workflows.

High-level Python API for soft tissue finite element simulation, built on the [Newton](https://github.com/newton-physics/newton) GPU-accelerated physics engine and [NVIDIA Warp](https://github.com/NVIDIA/warp).

NewtonTissue provides a clean interface for defining tetrahedral meshes, material properties, boundary conditions, and loading — then solving via Newton's VBD or XPBD solvers with GPU acceleration.

## Features

- **Tetrahedral FEM** with Neo-Hookean hyperelasticity via the Newton backend (VBD/XPBD solvers)
- **Serendipity hexahedral elements** (20-node) via warp.fem directly, with an incremental Newton-Raphson solver for nonlinear Neo-Hookean analysis
- **Material library** including predefined tissue types (e.g. prostate peripheral/transition zone) and support for per-element heterogeneous properties
- **Flexible boundary conditions**: fix by node index, bounding box, or arbitrary predicate
- **Loading types**: point forces, body forces, gravity, and time-varying prescribed displacements
- **Static and dynamic solves** with configurable time stepping, substeps, and damping

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+ and an NVIDIA GPU for accelerated solving (CPU fallback available).

## Quick start

```python
from newton_tissue import TissueModel, TissueSolver, IsotropicMaterial, FixedBC, Gravity

model = TissueModel(
    nodes=positions,          # (N, 3) array in meters
    elements=tet_connectivity, # (M, 4) array of node indices
    material=IsotropicMaterial(E=3000.0, nu=0.45),
    boundary_conditions=[FixedBC(node_ids=[0, 1, 2, 3])],
    loading=[Gravity()],
)

solver = TissueSolver(model)
results = solver.solve_static()
print(results.displacements)
```

See the [examples/](examples/) directory for more complete scenarios including layered palpation and cantilever beam tests.

## License

Apache-2.0
