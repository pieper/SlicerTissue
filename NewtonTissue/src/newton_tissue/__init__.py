"""NewtonTissue: High-level FEM API for soft tissue simulation.

Uses Newton (GPU physics engine) and Warp (NVIDIA GPU compute) as backends
for tetrahedral mesh simulation with Neo-Hookean hyperelasticity.

Basic usage::

    import numpy as np
    from newton_tissue import TissueModel, TissueSolver, IsotropicMaterial, FixedByBox, Gravity

    # Define a single tetrahedron
    nodes = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=np.float64)
    elements = np.array([[0, 1, 2, 3]], dtype=np.int32)

    # Create model
    model = TissueModel(
        nodes=nodes,
        elements=elements,
        material=IsotropicMaterial(E=20e3, nu=0.48, density=1040.0),
        boundary_conditions=[FixedByBox([0, 0, 0], [0.01, 1, 1])],
        loading_conditions=[Gravity()],
    )

    # Solve
    solver = TissueSolver(model)
    result = solver.step_dummy()  # Use step() when GPU backend is connected
    print(result.positions)
"""

from .boundary import BoundaryCondition, FixedBC, FixedByBox, FixedByPredicate
from .hex_model import HexTissueModel
from .hex_solver import HexTissueSolver
from .loading import BodyForce, Gravity, LoadingCondition, PointForce, PrescribedDisplacement
from .materials import (
    PROSTATE_PERIPHERAL,
    PROSTATE_TRANSITION,
    AnisotropicMaterial,
    HeterogeneousMaterial,
    IsotropicMaterial,
    Material,
    NodalMaterial,
)
from .model import TissueModel
from .mpm import AttachmentSet, MPMMaterial, MPMSimulator
from .results import SimulationResults
from .solver import TissueSolver

__version__ = "0.1.0"

__all__ = [
    "TissueModel",
    "TissueSolver",
    "HexTissueModel",
    "HexTissueSolver",
    "SimulationResults",
    "Material",
    "IsotropicMaterial",
    "HeterogeneousMaterial",
    "NodalMaterial",
    "AnisotropicMaterial",
    "PROSTATE_PERIPHERAL",
    "PROSTATE_TRANSITION",
    "MPMMaterial",
    "MPMSimulator",
    "AttachmentSet",
    "BoundaryCondition",
    "FixedBC",
    "FixedByPredicate",
    "FixedByBox",
    "LoadingCondition",
    "PointForce",
    "BodyForce",
    "Gravity",
    "PrescribedDisplacement",
]
