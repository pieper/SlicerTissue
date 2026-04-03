import os
import unittest
import numpy
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *

import festiv

#
# TissueSimulation
#

class TissueSimulation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "TissueSimulation" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Simulation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Steve Pieper (Isomics, Inc.)"]
    self.parent.helpText = """
    This is an interface to the festiv finite element solver tools.
    """
    self.parent.acknowledgementText = """
    This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
    and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# TissueSimulationWidget
#

class TissueSimulationWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    # reload and run specific tests
    scenarios = ("OneElement", "GluedBeam", "TwoElements", "Slab", "Subdivision", "Stack3Compare", "Newton", "MPM", "LayeredTissueHex", "LayeredAnisotropicTissue", "ResolutionCompare", "WarpMPM", "CTHeadMPM")
    for scenario in scenarios:
      button = qt.QPushButton("Reload and Test %s" % scenario)
      button.toolTip = "Reload this module and then run the %s self test." % scenario
      parametersFormLayout.addRow(button)
      button.connect('clicked()', lambda scenario=scenario: self.onReloadAndTest(scenario=scenario))

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onReloadAndTest)

    # Add vertical spacer
    self.layout.addStretch(1)

  def cleanup(self):
    pass

  def onApplyButton(self):
    logic = TissueSimulationLogic()
    logic.run(self.inputSelector.currentNode(), self.outputSelector.currentNode(), enableScreenshotsFlag,screenshotScaleFactor)


#
# TissueSimulationLogic
#

class TissueSimulationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, structure=None):
    if not structure:
      structure = festiv.structure.structure()
    self.structure = structure
    self.gridder = festiv.el_grid.gridder(self.structure)
    self._updatingNodeControlPoints = False
    self.fiducialList = None
    self.model = None

  def createModel(self):
    self.gridder._steps = (9,)*6
    self.gridder.surface_grid()
    # TODO
    # load directly to slicer node
    surfacePath = slicer.app.temporaryPath + 'oneElement.vtk'
    self.gridder.write_grid(surfacePath)
    self.model = slicer.util.loadModel(surfacePath)
    displayNode = self.model.GetDisplayNode()
    displayNode.SetBackfaceCulling(0)
    displayNode.SetEdgeVisibility(1)

  def updateModel(self):
    modelPoints = slicer.util.array(self.model.GetID())
    if modelPoints is None:
      # older slicer without direct access to point arrays
      p = self.model.GetPolyData().GetPoints().GetData()
      modelPoints = vtk.util.numpy_support.vtk_to_numpy(p)
    self.gridder.surface_grid()
    modelPoints[:] = numpy.array(self.gridder._points)
    self.model.GetPolyData().GetPoints().GetData().Modified()
    self.model.GetPolyData().GetPoints().Modified()

  def setControlPointListDisplay(self,fiducialList):
    displayNode = fiducialList.GetDisplayNode()
    # TODO: pick appropriate defaults
    # 135,135,84
    displayNode.SetTextScale(2.)
    displayNode.SetGlyphScale(5.)
    displayNode.SetGlyphTypeFromString('Sphere3D')
    displayNode.SetColor((0.6,0.6,0.2))
    displayNode.SetSelectedColor((1,1,0))
    displayNode.SetActiveColor((1,.5,0))
    unconstrained = slicer.vtkMRMLMarkupsFiducialDisplayNode.SnapModeUnconstrained
    displayNode.SetSnapMode(unconstrained)
    #displayNode.GetAnnotationTextDisplayNode().SetColor((1,1,0))
    displayNode.SetVisibility(True)

  def createNodeControlPoints(self,name='N'):
    """Add a fiducial for each node in the structure
    """

    markupsLogic = slicer.modules.markups.logic()
    originalActiveListID = markupsLogic.GetActiveListID()
    slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)

    # make the pointList list if required
    self.fiducialList = markupsLogic.AddNewMarkupsNode("vtkMRMLMarkupsFiducialNode", name)
    self.setControlPointListDisplay(self.fiducialList)

    # make this active so that the fids will be added to it
    markupsLogic.SetActiveListID(self.fiducialList)

    # make a fiducial for each node, indicating fixity
    # - index in fiducial list is equal to node index in _nodes list
    for node in self.structure._nodes:
      pu = node.pu()
      self.fiducialList.AddControlPoint(*pu)
      fiducialIndex = self.fiducialList.GetNumberOfControlPoints()-1

      self.fiducialList.SetNthControlPointLabel(fiducialIndex, name)
      nodeFixed = node._fixed.max() > 0
      self.fiducialList.SetNthControlPointSelected(fiducialIndex, not nodeFixed)
      self.fiducialList.SetNthControlPointLocked(fiducialIndex, not nodeFixed)

    # observe list for changes
    self.fiducialList.AddObserver( self.fiducialList.PointModifiedEvent,
      lambda caller,event: self.onControlPointMoved(caller))
    self.fiducialList.AddObserver( self.fiducialList.PointEndInteractionEvent,
        lambda caller,event: self.onControlPointEndMoving(caller))

    try:
      originalActiveList = slicer.util.getNode(originalActiveListID)
      markupsLogic.SetActiveListID(originalActiveList)
    except slicer.util.MRMLNodeNotFoundException:
      pass
    slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

  def onControlPointMoved(self,fiducialList):
    """Callback when fiducialList's point has been changed."""
    if self._updatingNodeControlPoints:
      return
    slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
    nodeCount = self.fiducialList.GetNumberOfControlPoints()
    for nodeIndex in range(nodeCount):
      node = self.structure._nodes[nodeIndex]
      point = [0,]*3
      self.fiducialList.GetNthControlPointPosition(nodeIndex,point)
      node._u = numpy.array(point) - node._p
    self.updateFromStructure()
    slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

  def onControlPointEndMoving(caller,event):
    pass

  def updateFromStructure(self):
    # update structure (TODO: save decomposed matrix in structure.py)
    self.structure.apply_bc()
    self.structure.solve()
    self.structure.updateNodes()

    self.updateModel()

    # refresh node fiducials
    self._updatingNodeControlPoints = True
    nodeCount = self.fiducialList.GetNumberOfControlPoints()
    for nodeIndex in range(nodeCount):
      node = self.structure._nodes[nodeIndex]
      pu = node.pu()
      self.fiducialList.SetNthControlPointPosition(nodeIndex,*pu)
    self._updatingNodeControlPoints = False

  def run(self, structure=None):
    """
    Runs the full simulation pipeline: matrix assembly, solve, and visualization.
    If a structure is passed, it will be used for the simulation. Otherwise,
    the logic's internal structure is used.
    """
    if structure:
      self.structure = structure
      self.gridder = festiv.el_grid.gridder(self.structure)

    # 1. Create the stiffness matrix, apply boundary conditions, and solve
    self.structure.make_K()
    self.structure.apply_bc()
    self.structure.solve()

    # 2. Visualize the results in Slicer
    self.createModel()
    self.updateModel()  # overwrite LPS→RAS transform from VTK load
    self.createNodeControlPoints()

    # 3. Make the logic accessible for interactive debugging from the console
    slicer.tissueLogic = self


class TissueSimulationTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self,scenario=None):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    if scenario is None or scenario == "OneElement":
      self.test_TissueSimulation1()
    elif scenario == "GluedBeam":
      self.test_TissueSimulation2()
    elif scenario == "TwoElements":
      self.test_TissueSimulation3()
    elif scenario == "Slab":
      self.test_TissueSimulation_Slab()
    elif scenario == "Subdivision":
      self.test_TissueSimulation_Subdivision()
    elif scenario == "Stack3Compare":
      self.test_Stack3Compare()
    elif scenario == "Newton":
      self.test_NewtonPackage()
    elif scenario == "MPM":
      self.test_MPMSimulation()
    elif scenario == "LayeredTissueHex":
      self.test_LayeredTissueHex()
    elif scenario == "LayeredAnisotropicTissue":
      self.test_LayeredAnisotropicTissue()
    elif scenario == "ResolutionCompare":
      self.test_ResolutionCompare()
    elif scenario == "WarpMPM":
      self.test_WarpMPM()
    elif scenario == "CTHeadMPM":
      self.test_CTHeadMPM()
    else:
      self.delayDisplay(f"Unknown test scenario: {scenario}", 3000)
      self.fail(f"Unknown test scenario: {scenario}")

  def test_TissueSimulation1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests sould exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test",100)

    #
    # festiv
    # finite element soft tissue interactive visualization
    #
    # pieper@isomics.com
    # copyright 2009 All rights reserved
    #
    #
    # oneelement.py
    # - simplest example with one element
    #

    import festiv
    import festiv.structure
    import festiv.element
    import festiv.node
    import festiv.meshing
    import festiv.el_grid

    import importlib

    importlib.reload(festiv.structure)
    importlib.reload(festiv.element)
    importlib.reload(festiv.node)
    importlib.reload(festiv.meshing)
    importlib.reload(festiv.el_grid)

    # make a structure
    logic = TissueSimulationLogic()
    s = logic.structure
    iso20 = festiv.isomap.iso20()

    # add an element
    element = festiv.element.element20()
    s._elements.append(element)

    # create the nodes and make the element 40mm on a side
    for i in range(20):
      node = festiv.node.node()
      node._p = numpy.array(iso20.__unit_nodes__[i]) * 20
      s._nodes.append( node )
      element._nodes[i] = node

    # set fixed boundary conditions on the lower face
    for node in element.face_nodes(1):
      node._fixed.fill(1)

    if False:
      # move all the top nodes up by 5
      for node in element.face_nodes(0):
        node._u.fill(0)
        node._u[2] = 5
        node._fixed.fill(1)

    # grab and move a node at the top corner by a fixed offset
    if True:
        node = s._elements[0]._nodes[0]
        node._u = numpy.array([10,10,10])
        node._fixed.fill(1)


    # create the stiffness matrix
    s.make_K()
    s.apply_bc()
    s.solve()

    #
    # now visualize
    #
    logic.createModel()
    logic.updateModel()  # overwrite LPS→RAS transform from VTK load
    logic.createNodeControlPoints()

    slicer.tissueLogic = logic

    self.delayDisplay('Test passed!')

  def test_Stack3Compare(self):
    """Side-by-side 3-element stack: festiv (linear elastic) vs warp.fem (Neo-Hookean).
    Runs the stack3_compare.py script which creates interactive mirrored boundary conditions.
    """
    self.delayDisplay("Starting Stack3 Compare test", 100)

    import os
    scriptPath = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "NewtonTissue", "examples", "stack3_compare.py")
    if not os.path.exists(scriptPath):
      self.fail(f"stack3_compare.py not found at {scriptPath}")

    g = globals().copy()
    g['__file__'] = scriptPath
    exec(open(scriptPath, encoding='utf-8').read(), g)

    self.delayDisplay('Stack3 Compare test passed!')

  def test_NewtonPackage(self):
    """
    Test if 'newton' package can be installed and imported.
    """
    self.delayDisplay("Starting newton package test")

    try:
      import newton
      self.delayDisplay("'newton' package is already available.")
    except ImportError:
      self.delayDisplay("Attempting to install 'newton-krylov' from PyPI...")
      import slicer.util
      slicer.util.pip_install('newton-krylov')
      try:
        import newton
        self.delayDisplay("Successfully installed and imported 'newton' package.")
      except ImportError:
        self.delayDisplay("Failed to import 'newton' after installation.", 3000)
        self.fail("Could not import 'newton' package after pip_install.")
    self.delayDisplay('Test passed!')

  def test_MPMSimulation(self):
    """
    Test if 'mpm' simulation can run by copying essential code
    from the mpm.py experiment.
    """
    self.delayDisplay("Starting MPM simulation test")

    try:
      import warp as wp
      import newton
      from newton.solvers import SolverImplicitMPM
      self.delayDisplay("'warp' and 'newton' packages are already available.")
    except ImportError:
      self.delayDisplay("Attempting to install 'warp-lang' and 'newton-krylov' from PyPI...")
      import slicer.util
      try:
        slicer.util.pip_install("warp-lang")
        slicer.util.pip_install("newton-krylov")
        import warp as wp
        import newton
        from newton.solvers import SolverImplicitMPM
        self.delayDisplay("Successfully installed and imported 'warp' and 'newton' packages.")
      except Exception as e:
        self.delayDisplay(f"Failed to install or import required packages: {e}", 3000)
        self.fail(f"Could not install/import packages for MPM test: {e}")

    import slicer

    # Essential code from mpm.py experiment copied here for the test

    class SlicerViewer():
        """Minimal viewer for non-interactive testing"""
        def __init__(self):
            self.running = True
            self.paused = False
        def is_running(self): return self.running
        def is_paused(self): return self.paused
        def set_model(self, model): pass
        def begin_frame(self, sim_time): pass
        def log_points(self, name, points, radii, colors, hidden): pass
        def end_frame(self): pass
        def close(self): pass

    class Example:
        def __init__(self, viewer, options):
            self.fps = 60.0
            self.frame_dt = 1.0 / self.fps
            self.sim_time = 0.0
            self.sim_substeps = 1
            self.sim_dt = self.frame_dt / self.sim_substeps
            self.viewer = viewer

            builder = newton.ModelBuilder()
            sand_particles, snow_particles, mud_particles = self.emit_particles(builder, voxel_size=options.voxel_size)

            builder.add_ground_plane()
            self.model = builder.finalize()

            sand_particles = wp.array(sand_particles, dtype=int, device=self.model.device)
            snow_particles = wp.array(snow_particles, dtype=int, device=self.model.device)
            mud_particles = wp.array(mud_particles, dtype=int, device=self.model.device)

            self.model.particle_ke = 1.0e15
            self.model.particle_kd = 0.0
            self.model.particle_mu = 0.5

            mpm_options = SolverImplicitMPM.Options()
            mpm_options.voxel_size = options.voxel_size
            mpm_options.tolerance = options.tolerance
            mpm_options.max_iterations = options.max_iterations

            mpm_model = SolverImplicitMPM.Model(self.model, mpm_options)

            mpm_model.material_parameters.yield_pressure[snow_particles].fill_(2.0e4)
            mpm_model.material_parameters.yield_stress[snow_particles].fill_(1.0e3)
            mpm_model.material_parameters.tensile_yield_ratio[snow_particles].fill_(0.05)
            mpm_model.material_parameters.friction[snow_particles].fill_(0.1)
            mpm_model.material_parameters.hardening[snow_particles].fill_(10.0)

            mpm_model.material_parameters.yield_pressure[mud_particles].fill_(1.0e10)
            mpm_model.material_parameters.yield_stress[mud_particles].fill_(3.0e2)
            mpm_model.material_parameters.tensile_yield_ratio[mud_particles].fill_(1.0)
            mpm_model.material_parameters.hardening[mud_particles].fill_(2.0)
            mpm_model.material_parameters.friction[mud_particles].fill_(0.0)

            mpm_model.notify_particle_material_changed()

            self.solver = SolverImplicitMPM(mpm_model, mpm_options)

            self.state_0 = self.model.state()
            self.state_1 = self.model.state()

            self.solver.enrich_state(self.state_0)
            self.solver.enrich_state(self.state_1)

            self.viewer.set_model(self.model)

        def simulate(self):
            for _ in range(self.sim_substeps):
                self.state_0.clear_forces()
                self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
                self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0

        def step(self):
            self.simulate()
            self.sim_time += self.frame_dt

        @staticmethod
        def _spawn_particles(builder: newton.ModelBuilder, voxel_size, bounds_lo, bounds_hi, density, flags):
            particles_per_cell = 3
            res = numpy.array(
                numpy.ceil(particles_per_cell * (bounds_hi - bounds_lo) / voxel_size),
                dtype=int,
            )

            cell_size = (bounds_hi - bounds_lo) / res
            cell_volume = numpy.prod(cell_size)
            radius = numpy.max(cell_size) * 0.5
            mass = numpy.prod(cell_volume) * density

            begin_id = len(builder.particle_q)
            builder.add_particle_grid(
                pos=wp.vec3(bounds_lo),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=res[0] + 1,
                dim_y=res[1] + 1,
                dim_z=res[2] + 1,
                cell_x=cell_size[0],
                cell_y=cell_size[1],
                cell_z=cell_size[2],
                mass=mass,
                jitter=2.0 * radius,
                radius_mean=radius,
                flags=flags,
            )

            end_id = len(builder.particle_q)
            return numpy.arange(begin_id, end_id, dtype=int)

        @classmethod
        def emit_particles(cls, builder: newton.ModelBuilder, voxel_size: float):
            cls._spawn_particles(
                builder, voxel_size, bounds_lo=numpy.array([-0.5, -0.5, 0.0]),
                bounds_hi=numpy.array([0.5, 0.5, 0.25]), density=1000.0, flags=0,
            )
            sand_particles = cls._spawn_particles(
                builder, voxel_size, bounds_lo=numpy.array([0.25, -0.5, 0.5]),
                bounds_hi=numpy.array([0.75, 0.5, 0.75]), density=2500.0, flags=newton.ParticleFlags.ACTIVE,
            )
            snow_particles = cls._spawn_particles(
                builder, voxel_size, bounds_lo=numpy.array([-0.75, -0.5, 0.5]),
                bounds_hi=numpy.array([-0.25, 0.5, 0.75]), density=300, flags=newton.ParticleFlags.ACTIVE,
            )
            mud_particles = cls._spawn_particles(
                builder, voxel_size, bounds_lo=numpy.array([-0.5, -0.25, 1.0]),
                bounds_hi=numpy.array([0.5, 0.25, 1.5]), density=1000.0, flags=newton.ParticleFlags.ACTIVE,
            )
            return sand_particles, snow_particles, mud_particles

    class Options():
        max_iterations = 25
        tolerance = 1.0e-6
        voxel_size = 0.15

    try:
      viewer = SlicerViewer()
      options = Options()
      example = Example(viewer, options)
      # run for a few steps to ensure it doesn't crash
      self.delayDisplay("Running simulation for 10 steps...")
      for i in range(10):
        example.step()
        slicer.app.processEvents() # Keep UI responsive
    except Exception as e:
      import traceback
      traceback.print_exc()
      self.delayDisplay(f"Failed to run MPM simulation: {e}", 3000)
      self.fail(f"Failed to run MPM simulation: {e}")

    self.delayDisplay('Test passed!')

  def test_TissueSimulation2(self):
    """ Test a beam made of three elements stacked and glued together
    """

    self.delayDisplay("Starting the Glued Beam test", 100)

    import festiv
    import festiv.structure
    import festiv.element
    import festiv.node
    import festiv.meshing
    import festiv.el_grid
    import festiv.isomap

    import importlib

    # Reload modules to pick up any changes
    importlib.reload(festiv.structure)
    importlib.reload(festiv.element)
    importlib.reload(festiv.node)
    importlib.reload(festiv.meshing)
    importlib.reload(festiv.el_grid)
    importlib.reload(festiv.isomap)

    # --- Setup the Simulation ---

    # 1. Initialize the logic and the main structure
    logic = TissueSimulationLogic()
    s = logic.structure
    iso20 = festiv.isomap.iso20()
    elementSize = 20.0

    # 2. Create the base (bottom) element
    baseElement = festiv.element.element20()
    s._elements.append(baseElement)
    for i in range(20):
      node = festiv.node.node()
      # Position nodes for a 40x40x40 cube centered at (0,0,0)
      node._p = numpy.array(iso20.__unit_nodes__[i]) * elementSize
      s._nodes.append(node)
      baseElement._nodes[i] = node

    # 3. Create the top element, positioned above the base element
    topElement = festiv.element.element20()
    s._elements.append(topElement)
    # The gap between elements should be the size of one element
    z_offset = elementSize * 2 
    for i in range(20):
      node = festiv.node.node()
      node._p = (numpy.array(iso20.__unit_nodes__[i]) * elementSize) + numpy.array([0, 0, z_offset])
      s._nodes.append(node)
      topElement._nodes[i] = node

    # 4. Use glue_faces to create a middle element that connects the base and top
    #    This glues the bottom face (1) of the top element to the top face (0) of the base element
    festiv.meshing.glue_faces(s, topElement, 1, baseElement, 0)

    # 5. Set boundary conditions
    #    - Fix the bottom face of the base element
    for node in baseElement.face_nodes(1):
      node._fixed.fill(1)

    #    - Apply a displacement to a top corner of the top element
    displacedNode = topElement._nodes[0]
    displacedNode._u = numpy.array([10, 10, 10])
    displacedNode._fixed.fill(1)

    # 6. Run the solver
    s.make_K()
    s.apply_bc()
    s.solve()

    # 7. Visualize the result in Slicer
    logic.createModel()
    logic.createNodeControlPoints()
    slicer.tissueLogic = logic

    self.delayDisplay('Glued Beam Test passed!')

  def test_TissueSimulation3(self):
    """ Test a two-element structure with a manually created compatible mesh by sharing nodes.
    """

    self.delayDisplay("Starting the Two-Element test", 100)

    import festiv
    import festiv.structure
    import festiv.element
    import festiv.node
    import festiv.meshing
    import festiv.el_grid
    import festiv.isomap

    import importlib

    # Reload modules to pick up any changes
    importlib.reload(festiv.structure)
    importlib.reload(festiv.element)
    importlib.reload(festiv.node)
    importlib.reload(festiv.meshing)
    importlib.reload(festiv.el_grid)
    importlib.reload(festiv.isomap)

    # --- Setup the Simulation ---
    # Use position-based node sharing: create nodes by physical position
    # and deduplicate via a dict keyed by rounded coordinates.
    # This avoids error-prone face-to-face index mapping entirely.

    # 1. Initialize the logic and the main structure
    logic = TissueSimulationLogic()
    s = logic.structure
    iso20 = festiv.isomap.iso20()
    elementSize = 20.0

    # 2. Create both elements with position-based node sharing
    node_dict = {}  # (round(x), round(y), round(z)) -> node
    for k in range(2):  # 2 elements stacked in z
      center_z = (k - 0.5) * 2 * elementSize  # -20, +20
      element = festiv.element.element20()
      s._elements.append(element)
      for i in range(20):
        ux, uy, uz = iso20.__unit_nodes__[i]
        px = ux * elementSize
        py = uy * elementSize
        pz = uz * elementSize + center_z
        key = (round(px, 6), round(py, 6), round(pz, 6))
        if key not in node_dict:
          node = festiv.node.node()
          node._p = numpy.array([px, py, pz])
          s._nodes.append(node)
          node_dict[key] = node
        element._nodes[i] = node_dict[key]

    # Mark shared faces between elements
    el_node_sets = [set(id(n) for n in el._nodes if n) for el in s._elements]
    for ei, el in enumerate(s._elements):
      for face_idx in range(6):
        face_node_ids = set(id(el._nodes[ni]) for ni in el.__faces__[face_idx][:8] if el._nodes[ni])
        for ej, other_set in enumerate(el_node_sets):
          if ei != ej and face_node_ids.issubset(other_set):
            el._shared_faces[face_idx] = 1
            break

    baseElement = s._elements[0]
    topElement = s._elements[1]

    # 3. Set boundary conditions
    for node in baseElement.face_nodes(1): # Fix bottom face of base element
      node._fixed.fill(1)

    displacedNode = topElement._nodes[0] # Displace a top corner of the top element
    displacedNode._u = numpy.array([10, 10, 10])
    displacedNode._fixed.fill(1)

    # 5. Run the solver and visualize
    logic.run(s)

    self.delayDisplay('Two-Element Test passed!')

  def test_TissueSimulation_Slab(self):
    """ Test a multi-layer slab of tissue.
    This test creates a 2x2x2 grid of elements with different material
    properties to simulate soft and hard tissue layers, adapted from the
    layer generation logic in the original CAPS system.
    """

    self.delayDisplay("Starting the Slab test", 100)

    import festiv

    import importlib

    # Reload modules to pick up any changes
    for module_name in ('structure', 'element', 'node', 'meshing', 'el_grid', 'isomap'):
        importlib.reload(getattr(festiv, module_name))

    # --- Setup the Simulation ---

    # 1. Initialize the logic and structure
    logic = TissueSimulationLogic()
    s = logic.structure

    # 2. Define the slab geometry and material properties for each layer
    grid_dims = (2, 2)  # A 2x2 grid of elements in the XY plane
    element_size = (20.0, 20.0) # Each element is 20x20 units
    layer_defs = [
        {
            'thickness': 20.0,
            'youngs_modulus': 1.e4,  # Softer top layer
            'poissons_ratio': 0.45
        },
        {
            'thickness': 20.0,
            'youngs_modulus': 1.e6,  # Stiffer bottom layer
            'poissons_ratio': 0.3
        }
    ]

    # 3. Create the layered grid
    festiv.meshing.create_layered_grid(s, grid_dims, element_size, layer_defs)

    # 4. Apply boundary conditions
    # Fix all nodes on the bottom-most face of the entire slab
    bottom_z = -sum(layer['thickness'] for layer in layer_defs)
    for node in s._nodes:
        if numpy.isclose(node._p[2], bottom_z):
            node._fixed.fill(1)

    # Displace a single node on the top surface
    top_node = s._nodes[0] # This should be a corner node on the top surface
    top_node._u = numpy.array([10, 10, 10])
    top_node._fixed.fill(1)

    # 5. Run the solver and visualize
    logic.run(s)

    self.delayDisplay('Slab Test passed!')

  def test_LayeredTissueHex(self):
    """Interactive multi-layer tissue simulation using 20-node serendipity hex elements.

    Creates a 4-layer tissue block (liver/muscle/fat/skin) with Neo-Hookean
    hyperelasticity via warp.fem. Bottom face is fixed; a draggable markup
    fiducial on the top surface applies a prescribed displacement BC.

    Self-test checks:
      - Model and fiducials are created in the scene
      - Simulated displacement is non-zero after initial solve
      - Re-solve after programmatic BC change produces different displacements
    """
    self.delayDisplay("Starting LayeredTissueHex test", 100)

    import os, sys
    scriptPath = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "NewtonTissue", "examples", "layered_tissue_hex.py")
    if not os.path.exists(scriptPath):
      self.fail(f"layered_tissue_hex.py not found at {scriptPath}")

    # Import LayeredTissueHex without triggering the entry-point guard
    examplesDir = os.path.dirname(scriptPath)
    srcDir = os.path.join(os.path.dirname(examplesDir), 'src')
    for d in [examplesDir, srcDir]:
      if d not in sys.path:
        sys.path.insert(0, d)

    # Flush stale newton_tissue cache so updated workspace files are picked up
    import importlib, importlib.util
    for _k in [k for k in sys.modules if 'newton_tissue' in k]:
        del sys.modules[_k]

    spec = importlib.util.spec_from_file_location("layered_tissue_hex", scriptPath)
    mod  = importlib.util.module_from_spec(spec)
    mod.slicer = None   # prevents entry-point guard from firing on import
    spec.loader.exec_module(mod)
    LayeredTissueHex = mod.LayeredTissueHex

    # ── Instantiate and run ──────────────────────────────────────────────
    try:
      import warp as wp
    except ImportError:
      self.delayDisplay("warp not available — skipping LayeredTissueHex test", 2000)
      return

    sim = LayeredTissueHex(device="cpu")
    sim.run()
    slicer.app.processEvents()

    # ── Assert model was created ─────────────────────────────────────────
    self.assertIsNotNone(sim.vtk_model,
                         "VTK model node was not created")
    self.assertIsNotNone(sim.fiducial_list,
                         "Fiducial list was not created")
    self.assertIsNotNone(
        slicer.mrmlScene.GetFirstNodeByName('LayeredTissueHex'),
        "LayeredTissueHex model not found in scene")

    # ── Assert non-zero displacement from initial solve ──────────────────
    u_initial = sim.u_field.dof_values.numpy().copy()
    free_dofs = [i for i in range(sim.n_dof) if i not in sim.bc_dofs]
    max_disp = max(numpy.linalg.norm(u_initial[i]) for i in free_dofs)
    self.assertGreater(max_disp, 1e-6,
                       f"Expected non-zero displacement, got max={max_disp}")
    self.delayDisplay(f"  Initial max free-node displacement: {max_disp*1000:.2f} mm")

    # ── Assert re-solve changes displacements ────────────────────────────
    # Use full solve from scratch for reliable comparison
    u_vals = numpy.zeros((sim.n_dof, 3), dtype=numpy.float32)
    u_vals[sim.palp_dof] = [0.0, -0.010, 0.0]
    sim.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
    sim._newton_solve(n_load_steps=4)
    sim.updateModel()
    slicer.app.processEvents()

    u_after = sim.u_field.dof_values.numpy()
    max_disp_after = max(numpy.linalg.norm(u_after[i]) for i in free_dofs)
    self.assertGreater(max_disp_after, max_disp,
                       "Larger BC should produce larger displacements")
    self.delayDisplay(
        f"  After larger BC: max free-node displacement = {max_disp_after*1000:.2f} mm")

    # Restore initial state so interactive simulation is ready after the test
    u_reset = numpy.zeros((sim.n_dof, 3), dtype=numpy.float32)
    u_reset[sim.palp_dof] = [0.0, -0.005, 0.0]
    sim.u_field.dof_values.assign(wp.array(u_reset, dtype=wp.vec3))
    sim._newton_solve(n_load_steps=4)
    sim._last_converged_u = sim.u_field.dof_values.numpy().copy()
    sim._last_converged_corner = sim.u_field.dof_values.numpy()[sim.palp_dof].copy()
    sim.updateModel()
    sim._sync_fiducials()
    slicer.app.processEvents()

    self.delayDisplay('LayeredTissueHex test passed!')

  def test_LayeredAnisotropicTissue(self):
    """Anisotropic layered tissue simulation (HGO fiber term, 20-node hex).

    Parallel to test_LayeredTissueHex but uses AnisotropicMaterial.
    Verifies:
      - Model + fiducials created with AnisotropicMaterial
      - Tissue layer coloring applied
      - Displacement non-zero after initial solve
      - Larger BC produces larger displacements
      - X-push (along fiber) differs from Z-push (transverse) — confirms anisotropy
    """
    self.delayDisplay("Starting LayeredAnisotropicTissue test", 100)

    import os, sys, importlib.util

    scriptPath = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "NewtonTissue", "examples", "layered_aniso_hex.py")
    if not os.path.exists(scriptPath):
      self.fail(f"layered_aniso_hex.py not found at {scriptPath}")

    examplesDir = os.path.dirname(scriptPath)
    srcDir = os.path.join(os.path.dirname(examplesDir), 'src')
    for d in [examplesDir, srcDir]:
      if d not in sys.path:
        sys.path.insert(0, d)

    # Flush stale newton_tissue cache
    import importlib
    for _k in [k for k in sys.modules if 'newton_tissue' in k or 'layered_aniso' in k]:
        del sys.modules[_k]

    spec = importlib.util.spec_from_file_location("layered_aniso_hex", scriptPath)
    mod  = importlib.util.module_from_spec(spec)
    mod.slicer = None
    spec.loader.exec_module(mod)
    LayeredAnisotropicTissueHex = mod.LayeredAnisotropicTissueHex

    try:
      import warp as wp
    except ImportError:
      self.delayDisplay("warp not available — skipping LayeredAnisotropicTissue test", 2000)
      return

    sim = LayeredAnisotropicTissueHex(device="cpu")
    sim.run()
    slicer.app.processEvents()

    self.assertIsNotNone(sim.vtk_model, "VTK model not created")
    self.assertIsNotNone(sim.fiducial_list, "Fiducial list not created")
    self.assertIsNotNone(
        slicer.mrmlScene.GetFirstNodeByName('LayeredAnisotropicTissueHex'),
        "LayeredAnisotropicTissueHex model not in scene")

    # Verify anisotropic material
    from newton_tissue import AnisotropicMaterial
    self.assertIsInstance(sim.tissue_model.material, AnisotropicMaterial,
                          "Material should be AnisotropicMaterial")

    # Verify tissue layer coloring
    dn = sim.vtk_model.GetDisplayNode()
    self.assertEqual(dn.GetScalarVisibility(), 1, "Tissue layer coloring not applied")

    # Verify non-zero displacement
    u_initial = sim.u_field.dof_values.numpy().copy()
    free_dofs = [i for i in range(sim.n_dof) if i not in sim.bc_dofs]
    max_disp = max(numpy.linalg.norm(u_initial[i]) for i in free_dofs)
    self.assertGreater(max_disp, 1e-6, f"Expected non-zero displacement, got {max_disp}")
    self.delayDisplay(f"  Initial max free-node displacement: {max_disp*1000:.2f} mm")

    # Larger BC → larger displacement
    u_vals = numpy.zeros((sim.n_dof, 3), dtype=numpy.float32)
    u_vals[sim.palp_dof] = [0.0, -0.010, 0.0]
    sim.u_field.dof_values.assign(wp.array(u_vals, dtype=wp.vec3))
    sim._newton_solve(n_load_steps=4)
    sim.updateModel()
    slicer.app.processEvents()
    u_after = sim.u_field.dof_values.numpy()
    max_disp_after = max(numpy.linalg.norm(u_after[i]) for i in free_dofs)
    self.assertGreater(max_disp_after, max_disp, "Larger BC should give larger displacement")
    self.delayDisplay(f"  After -10mm BC: {max_disp_after*1000:.2f} mm")

    # Anisotropy check: X-push (along fiber) vs Z-push (transverse) differ
    u_x = numpy.zeros((sim.n_dof, 3), dtype=numpy.float32)
    u_x[sim.palp_dof] = [0.010, 0.0, 0.0]   # push along fiber (X)
    sim.u_field.dof_values.assign(wp.array(u_x, dtype=wp.vec3))
    sim._newton_solve(n_load_steps=4)
    disp_x = max(numpy.linalg.norm(sim.u_field.dof_values.numpy()[i]) for i in free_dofs)

    u_z = numpy.zeros((sim.n_dof, 3), dtype=numpy.float32)
    u_z[sim.palp_dof] = [0.0, 0.0, 0.010]   # push transverse (Z)
    sim.u_field.dof_values.assign(wp.array(u_z, dtype=wp.vec3))
    sim._newton_solve(n_load_steps=4)
    disp_z = max(numpy.linalg.norm(sim.u_field.dof_values.numpy()[i]) for i in free_dofs)

    self.assertNotAlmostEqual(
        disp_x, disp_z, places=4,
        msg="X-push and Z-push should differ due to anisotropy")
    self.delayDisplay(
        f"  Anisotropy confirmed: X-push={disp_x*1000:.2f}mm, "
        f"Z-push={disp_z*1000:.2f}mm  (ratio={disp_x/max(disp_z,1e-12):.2f})")

    # Restore initial state
    u_reset = numpy.zeros((sim.n_dof, 3), dtype=numpy.float32)
    u_reset[sim.palp_dof] = [0.0, -0.005, 0.0]
    sim.u_field.dof_values.assign(wp.array(u_reset, dtype=wp.vec3))
    sim._newton_solve(n_load_steps=4)
    sim._last_converged_u = sim.u_field.dof_values.numpy().copy()
    sim._last_converged_corner = sim.u_field.dof_values.numpy()[sim.palp_dof].copy()
    sim.updateModel()
    sim._sync_fiducials()
    slicer.app.processEvents()

    self.delayDisplay('LayeredAnisotropicTissue test passed!')

  def test_WarpMPM(self):
    """Push-pull palpation realism test for MPM tissue block.

    Simulates pressing a finger into soft tissue (Pillsbury Dough Boy style)
    then releasing it.  Verifies:
      1. Gravity settles free particles downward; fixed inferior face stays put.
      2. Incremental push deforms the palpation region downward.
      3. After pulling back and settling, tissue recovers toward its
         gravity-settled shape (elastic Neo-Hookean recovery).
    """
    self.delayDisplay("Starting WarpMPM push-pull realism test", 200)

    import importlib.util
    if importlib.util.find_spec("warp") is None:
      self.delayDisplay("warp not available — skipping WarpMPM test", 2000)
      return

    import os, sys, importlib.util

    scriptPath = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "NewtonTissue", "examples", "mpm_tissue_block.py")
    if not os.path.exists(scriptPath):
      self.fail(f"mpm_tissue_block.py not found at {scriptPath}")

    examplesDir = os.path.dirname(scriptPath)
    srcDir = os.path.join(os.path.dirname(examplesDir), 'src')
    for d in [examplesDir, srcDir]:
      if d not in sys.path:
        sys.path.insert(0, d)

    for k in [k for k in sys.modules if 'newton_tissue.mpm' in k or 'mpm_tissue_block' in k]:
      del sys.modules[k]

    spec = importlib.util.spec_from_file_location("mpm_tissue_block", scriptPath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    MPMTissueBlock = mod.MPMTissueBlock

    # Stop any previously running simulation loop to prevent GPU interference
    if hasattr(slicer, 'mpmSim') and slicer.mpmSim is not None:
      try:
        slicer.mpmSim.stop_simulation_loop()
      except Exception:
        pass
      slicer.mpmSim = None

    sim = MPMTissueBlock()
    sim.run()
    slicer.app.processEvents()

    free_mask  = ~sim.sim.fixed.numpy().astype(bool)
    fixed_mask =  sim.sim.fixed.numpy().astype(bool)
    n_free  = int(free_mask.sum())
    n_fixed = int(fixed_mask.sum())
    self.delayDisplay(
        f"{sim.sim.n_particles} particles — {n_free} free, {n_fixed} fixed, "
        f"device={sim.device}", 600)

    # --- 1. Gravity settlement checks ---
    pos0 = sim.sim.x0.numpy().copy()         # rest (pre-gravity) positions
    pos_settled = sim.sim.get_positions().copy()   # after 50 warm-up steps

    y_disp_free = pos_settled[free_mask, 1] - pos0[free_mask, 1]
    self.assertLess(y_disp_free.mean(), -1e-6,
                    "Free particles should have settled downward under gravity")

    y_disp_fixed = pos_settled[fixed_mask, 1] - pos0[fixed_mask, 1]
    self.assertAlmostEqual(float(numpy.abs(y_disp_fixed).max()), 0.0, places=6,
                           msg="Inferior fixed particles must not move")
    self.delayDisplay(
        f"Gravity OK — free mean Y: {y_disp_free.mean()*1000:.2f} mm, "
        f"fixed max Y: {abs(y_disp_fixed).max()*1e6:.1f} µm", 600)

    # --- 2. Displacement-controlled push via simulation loop ---
    # Ramp probe sphere 0 → 15 mm into tissue over 60 levels × 150 ms each.
    # The rigid sphere contact pushes particles radially outward, creating a
    # smooth bowl-shaped depression like a real finger pressing into tissue.
    self.delayDisplay("Pressing finger into tissue...", 400)
    import qt as _qt
    push_depth_m = 0.015    # 15 mm total push
    n_ramp       = 60       # displacement levels
    ms_per_level = 150      # ms between increments (sim loop fires freely)
    rest_pos_m   = sim._palp_pos_mm / 1000.0

    for i in range(n_ramp):
      depth = push_depth_m * (i + 1) / n_ramp
      fid_pos = rest_pos_m + numpy.array([0.0, -depth, 0.0])
      sphere_c = sim._sphere_center_for_fiducial(fid_pos)
      sim._contact_sphere = {
        'center': sphere_c,
        'radius': sim._probe_radius,
      }
      sim._idle_ticks = 0
      if not sim._loop_running:
        sim.start_simulation_loop()
      _loop = _qt.QEventLoop()
      _qt.QTimer.singleShot(ms_per_level, _loop.quit)
      _loop.exec_()

    pos_pushed = sim.sim.get_positions().copy()
    dy_push_mm = (pos_pushed[sim._palp_mask, 1].mean()
                  - pos_settled[sim._palp_mask, 1].mean()) * 1000.0
    self.assertLess(dy_push_mm, -1.0,
                    f"Push should move palpation region ≥1 mm downward, got {dy_push_mm:.3f} mm")

    # Check smoothness: particles near (but outside) the contact zone should
    # also have moved somewhat — unlike a cookie-cutter stamp.
    center_xz = numpy.array([0.04, 0.04])  # block centre in xz
    r_xz = numpy.sqrt((pos_pushed[:, 0] - center_xz[0])**2 +
                       (pos_pushed[:, 2] - center_xz[1])**2)
    near_mask = (r_xz > sim._probe_radius) & (r_xz < 2.0 * sim._probe_radius) & free_mask
    if near_mask.sum() > 0:
      dy_near_mm = (pos_pushed[near_mask, 1].mean()
                    - pos_settled[near_mask, 1].mean()) * 1000.0
      self.assertLess(dy_near_mm, -0.1,
                      f"Surrounding tissue should deform too (smooth contact), got {dy_near_mm:.3f} mm")
      self.delayDisplay(
          f"Push: palp region {abs(dy_push_mm):.1f} mm, surround {abs(dy_near_mm):.1f} mm", 800)
    else:
      self.delayDisplay(f"Push: palp region {abs(dy_push_mm):.1f} mm", 800)

    self.delayDisplay(
        f"Finger at {abs(dy_push_mm):.1f} mm depth — releasing.", 1500)

    # --- 3. Finger lifts off gradually (like a real finger) ---
    # Ramp sphere back out over 30 levels so F tracks the withdrawal.
    n_withdraw = 30
    for i in range(n_withdraw):
      frac = 1.0 - (i + 1) / n_withdraw
      depth = push_depth_m * frac
      fid_pos = rest_pos_m + numpy.array([0.0, -depth, 0.0])
      sphere_c = sim._sphere_center_for_fiducial(fid_pos)
      sim._contact_sphere = {
        'center': sphere_c,
        'radius': sim._probe_radius,
      }
      sim._idle_ticks = 0
      if not sim._loop_running:
        sim.start_simulation_loop()
      _loop = _qt.QEventLoop()
      _qt.QTimer.singleShot(ms_per_level, _loop.quit)
      _loop.exec_()

    sim._contact_sphere = None
    sim.recover(n_steps=2000, show_every=5)

    # --- 4. Check elastic recovery ---
    pos_recovered = sim.sim.get_positions().copy()
    residual_mm = (numpy.abs(pos_recovered[free_mask]
                             - pos_settled[free_mask]).max() * 1000.0)
    self.assertLess(residual_mm, 5.0,
                    f"Elastic recovery should leave <5 mm residual, got {residual_mm:.2f} mm")
    self.delayDisplay(
        f"Recovery — max residual displacement: {residual_mm:.2f} mm", 1000)

    slicer.mpmSim = sim
    self.delayDisplay('WarpMPM push-pull test passed!')

  def test_CTHeadMPM(self):
    """CT-driven MPM simulation: particles classified by HU, gravity slider.

    Loads the CTHead dataset and creates an MPM simulation where bone
    particles are fixed, air is skipped, and soft tissue deforms under
    gravity.  A toolbar slider controls gravity magnitude (−2 g … +2 g).
    """
    self.delayDisplay("Starting CTHeadMPM test", 200)

    import importlib.util
    if importlib.util.find_spec("warp") is None:
      self.delayDisplay("warp not available — skipping CTHeadMPM test", 2000)
      return

    import os, sys, importlib.util

    scriptPath = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "NewtonTissue", "examples", "mpm_ct_head.py")
    if not os.path.exists(scriptPath):
      self.fail(f"mpm_ct_head.py not found at {scriptPath}")

    examplesDir = os.path.dirname(scriptPath)
    srcDir = os.path.join(os.path.dirname(examplesDir), 'src')
    for d in [examplesDir, srcDir]:
      if d not in sys.path:
        sys.path.insert(0, d)

    for k in [k for k in sys.modules
              if 'newton_tissue.mpm' in k or 'mpm_ct_head' in k]:
      del sys.modules[k]

    spec = importlib.util.spec_from_file_location("mpm_ct_head", scriptPath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # --- 1. Download / cache CTHead volume ---
    self.delayDisplay("Downloading CTHead (cached after first run)…", 400)
    nrrd_path = mod.download_ct_head()
    self.assertTrue(os.path.exists(nrrd_path),
                    f"CTHead file not found at {nrrd_path}")

    # --- 2. Load into Slicer ---
    self.delayDisplay("Loading CTHead volume…", 400)
    volume_node = slicer.util.loadVolume(nrrd_path)
    self.assertIsNotNone(volume_node, "Failed to load CTHead volume")

    # --- 3. Stop any previous MPM sim ---
    if hasattr(slicer, 'mpmSim') and slicer.mpmSim is not None:
      try:
        slicer.mpmSim.stop_simulation_loop()
        if hasattr(slicer.mpmSim, 'cleanup_toolbar'):
          slicer.mpmSim.cleanup_toolbar()
      except Exception:
        pass
      slicer.mpmSim = None

    # Clean up any previous MPMCTHead model nodes
    nodes = slicer.mrmlScene.GetNodesByName('MPMCTHead')
    for i in range(nodes.GetNumberOfItems()):
      node = nodes.GetItemAsObject(i)
      if node:
        slicer.mrmlScene.RemoveNode(node)

    # --- 4. Create MPM simulation (4 mm grid for practical speed) ---
    self.delayDisplay("Building MPM simulation from CT…", 400)
    sim = mod.MPMCTHead(volume_node, dx_mm=4.0, ppc=2)

    free_mask  = ~sim.sim.fixed.numpy().astype(bool)
    fixed_mask =  sim.sim.fixed.numpy().astype(bool)
    n_free  = int(free_mask.sum())
    n_fixed = int(fixed_mask.sum())
    self.delayDisplay(
        f"{sim.sim.n_particles} particles — {n_free} tissue, {n_fixed} bone, "
        f"device={sim.device}", 600)

    # --- 5. Verify gravity settlement ---
    pos0 = sim.sim.x0.numpy().copy()
    pos_settled = sim.sim.get_positions().copy()

    # Tissue particles should have moved under gravity (−S = −Z in sim)
    z_disp_free = pos_settled[free_mask, 2] - pos0[free_mask, 2]
    self.assertLess(z_disp_free.mean(), -1e-6,
                    "Free tissue should settle downward (−S) under gravity")

    # Bone particles must not move
    bone_disp = numpy.abs(pos_settled[fixed_mask] - pos0[fixed_mask]).max()
    self.assertAlmostEqual(float(bone_disp), 0.0, places=6,
                           msg="Bone particles must stay fixed")

    self.delayDisplay(
        f"Gravity OK — tissue mean Z disp: {z_disp_free.mean()*1000:.2f} mm, "
        f"bone max disp: {bone_disp*1e6:.1f} µm", 600)

    # --- 6. Launch interactive visualisation + gravity slider ---
    # Note: pre-stress (set_prestress) is disabled for now — it creates
    # an F/position mismatch that causes UL mode to diverge.  Proper
    # pre-tension should be done by scaling fiber rest lengths instead.
    sim.run()
    slicer.app.processEvents()

    # --- 7. Add SDF volumes for visual inspection ---
    self._add_sdf_volumes(sim, volume_node)

    # --- 8. Set up curve observer for interactive cutting ---
    # When the user draws an open curve markup, a scalpel cut is
    # automatically applied along that path down to bone depth.
    cuttingPath = os.path.join(examplesDir, "mpm_cutting.py")
    if os.path.exists(cuttingPath):
      for k in [k for k in sys.modules if 'mpm_cutting' in k]:
        del sys.modules[k]
      cut_spec = importlib.util.spec_from_file_location("mpm_cutting", cuttingPath)
      cut_mod  = importlib.util.module_from_spec(cut_spec)
      cut_spec.loader.exec_module(cut_mod)
      sim._curve_observer = cut_mod.CurveObserver(sim, depth_mm=25.0)
      self.delayDisplay("Curve observer active — draw an open curve to cut!", 800)

    slicer.mpmSim = sim
    self.delayDisplay('CTHeadMPM test passed — use gravity slider and draw curves to cut!')

  def _add_sdf_volumes(self, sim, ct_volume_node):
    """Create SDF scalar volumes resampled onto the CT grid for inspection.

    The bone and tissue SDFs are computed on the coarser MPM grid.  This
    method resamples them onto the CT volume grid so the IJKToRAS matches
    exactly, making it easy to overlay on the CT in slice views.
    """
    import vtk
    from scipy.ndimage import map_coordinates

    # CT grid geometry
    ras2ijk = vtk.vtkMatrix4x4()
    ct_volume_node.GetRASToIJKMatrix(ras2ijk)
    ijk2ras = vtk.vtkMatrix4x4()
    ct_volume_node.GetIJKToRASMatrix(ijk2ras)

    ct_arr = slicer.util.arrayFromVolume(ct_volume_node)  # (K, J, I)
    nK, nJ, nI = ct_arr.shape

    # Build mapping: CT IJK → MPM grid IJK
    # CT IJK → RAS: use ijk2ras
    # RAS mm → sim m: (RAS - offset) / 1000
    # sim m → MPM grid float index: pos / dx
    M = numpy.array([[ijk2ras.GetElement(r, c) for c in range(4)]
                      for r in range(4)])
    offset_mm = sim._ras_offset_mm
    dx_m = sim.sim.dx

    # Create index arrays for all CT voxels
    ii, jj, kk = numpy.meshgrid(
        numpy.arange(nI), numpy.arange(nJ), numpy.arange(nK),
        indexing='ij')
    ijk_flat = numpy.stack([ii.ravel(), jj.ravel(), kk.ravel(),
                            numpy.ones(nI * nJ * nK)], axis=1)  # (N, 4)
    ras_flat = (M @ ijk_flat.T).T[:, :3]  # (N, 3)
    sim_m = (ras_flat - offset_mm) / 1000.0
    mpm_ijk = sim_m / dx_m  # float grid indices into the MPM SDF

    for name, sdf_3d in [('BoneSDF', getattr(sim, '_bone_sdf_3d', None)),
                          ('TissueSDF', getattr(sim, '_tissue_sdf_3d', None))]:
      if sdf_3d is None:
        continue
      # Resample MPM SDF onto CT grid using trilinear interpolation
      # map_coordinates expects (3, N) with coordinates in array order (i,j,k)
      coords = numpy.stack([mpm_ijk[:, 0], mpm_ijk[:, 1], mpm_ijk[:, 2]])
      resampled = map_coordinates(sdf_3d, coords, order=1, mode='nearest')
      # Reshape to CT volume shape (KJI) and convert to mm
      ct_sdf = (resampled.reshape(nI, nJ, nK) * 1000.0).astype(numpy.float32)
      ct_sdf_kji = ct_sdf.transpose(2, 1, 0)

      # Remove any old volume with this name
      old = slicer.mrmlScene.GetFirstNodeByName(name)
      if old:
        slicer.mrmlScene.RemoveNode(old)

      vol = slicer.util.addVolumeFromArray(ct_sdf_kji, ijkToRAS=ijk2ras, name=name)
      vol.GetDisplayNode().SetAutoWindowLevel(True)

  def test_ResolutionCompare(self):
    """Compare Low / Medium / High mesh resolution on the same anisotropic tissue.
    Runs resolution_compare.py which creates three side-by-side hex models sharing
    a single palpation fiducial.  Models are solved sequentially on drag.
    """
    self.delayDisplay("Starting Resolution Compare test", 100)

    import os
    scriptPath = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "NewtonTissue", "examples", "resolution_compare.py")
    if not os.path.exists(scriptPath):
      self.fail(f"resolution_compare.py not found at {scriptPath}")

    import runpy
    runpy.run_path(scriptPath, run_name='__main__')

    self.delayDisplay('Resolution Compare test passed!')

  def test_TissueSimulation_Subdivision(self):
    """
    Tests the subdivision of a single 20-node element into eight smaller elements.
    """

    self.delayDisplay("Starting the Subdivision test", 10)

    import festiv
    import festiv.structure
    import festiv.element
    import festiv.node
    import festiv.meshing
    import festiv.el_grid
    import festiv.isomap

    import importlib

    # Reload modules to pick up any changes
    for module_name in ('structure', 'element', 'node', 'meshing', 'el_grid', 'isomap'):
        importlib.reload(getattr(festiv, module_name))

    # --- 1. Create a single parent element (similar to oneelement test) ---
    parent_element = festiv.element.element20()
    iso20 = festiv.isomap.iso20()

    # Create the nodes for the parent element, 40mm on a side
    for i in range(20):
        node = festiv.node.node()
        node._p = numpy.array(iso20.__unit_nodes__[i]) * 20
        parent_element._nodes[i] = node

    # --- 2. Subdivide the parent element into a new 8-element structure ---
    subdivided_structure = festiv.meshing.subdivide_element(parent_element)

    # --- 3. Apply boundary conditions to the new structure ---
    # Fix the bottom face of the entire subdivided block.
    # The bottom-most nodes are those with the minimum Z coordinate.
    min_z = min(node._p[2] for node in subdivided_structure._nodes)
    for node in subdivided_structure._nodes:
        if numpy.isclose(node._p[2], min_z):
            node._fixed.fill(1)

    # Displace a single node on the top surface.
    # Node 26 is the corner at (1,1,1) in the 3x3x3 grid, a top corner of the block.
    top_node = subdivided_structure._nodes[26]
    top_node._u = numpy.array([10, 10, 10])
    top_node._fixed.fill(1)

    # --- 4. Run the solver and visualize ---
    logic = TissueSimulationLogic()
    logic.run(subdivided_structure)

    self.delayDisplay('Subdivision Test passed!')
