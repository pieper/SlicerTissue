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
    scenarios = ("OneElement", "GluedBeam", "TwoElements", "Slab", "Subdivision", "Newton", "MPM")
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
    self.gridder._steps = (3,)*6
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
    elif scenario == "Newton":
      self.test_NewtonPackage()
    elif scenario == "MPM":
      self.test_MPMSimulation()
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
    logic.createNodeControlPoints()

    slicer.tissueLogic = logic

    self.delayDisplay('Test passed!')

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

    # 1. Initialize the logic and the main structure
    logic = TissueSimulationLogic()
    s = logic.structure
    iso20 = festiv.isomap.iso20()
    elementSize = 20.0

    # 2. Create the base (bottom) element and all its nodes
    baseElement = festiv.element.element20()
    s._elements.append(baseElement)
    for i in range(20):
      node = festiv.node.node()
      node._p = numpy.array(iso20.__unit_nodes__[i]) * elementSize
      s._nodes.append(node)
      baseElement._nodes[i] = node

    # 3. Create the top element, sharing nodes with the base element's top face
    topElement = festiv.element.element20()
    s._elements.append(topElement)

    # Map the top face of the base element to the bottom face of the top element
    # This creates the compatible mesh by sharing nodes.
    # NOTE: The node order must be reversed on one face to create a non-inverted element.
    # A simple list reversal is not enough due to the specific ordering of corner and mid-edge nodes.
    # We must map them explicitly.
    # Base Top Face (0) node indices: (0, 11, 3, 10, 2, 9, 1, 8)
    # Top Bottom Face (1) node indices: (4, 12, 5, 13, 6, 14, 7, 15)
    # Correct reversed mapping:
    base_to_top_map = {
        0: 6, 11: 13, 3: 5, 10: 12, 2: 4, 9: 15, 1: 7, 8: 14
    }
    for base_node_idx, top_node_idx in base_to_top_map.items():
        shared_node = baseElement._nodes[base_node_idx]
        topElement._nodes[top_node_idx] = shared_node


    # Create the remaining 12 (non-shared) nodes for the top element
    z_offset = elementSize * 2
    for i in range(20):
      if not topElement._nodes[i]: # If node is not already shared
        node = festiv.node.node()
        node._p = (numpy.array(iso20.__unit_nodes__[i]) * elementSize) + numpy.array([0, 0, z_offset])
        s._nodes.append(node)
        topElement._nodes[i] = node

    # 4. Set boundary conditions
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
