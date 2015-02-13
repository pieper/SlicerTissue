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

    #
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ( ("vtkMRMLScalarVolumeNode"), "" )
    self.inputSelector.addAttribute( "vtkMRMLScalarVolumeNode", "LabelMap", 0 )
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )

    # TODO: make some more interface elements
    # parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

    # reload and run specific tests
    scenarios = ("OneElement",)
    for scenario in scenarios:
      button = qt.QPushButton("Reload and Test %s" % scenario)
      self.reloadAndTestButton.toolTip = "Reload this module and then run the %s self test." % scenario
      parametersFormLayout.addWidget(button)
      #button.connect('clicked()', lambda s=scenario: self.onReloadAndTest(scenario=s))
      button.connect('clicked()', self.onReloadAndTest)

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
    self._updatingNodeFiducials = False
    self.fiducialList = None
    self.model = None

  def createModel(self):
    self.gridder._steps = (4,)*6
    self.gridder.surface_grid()
    # TODO
    # load directly to slicer node
    surfacePath = slicer.app.temporaryPath + 'oneElement.vtk'
    self.gridder.write_grid(surfacePath)
    loaded,self.model = slicer.util.loadModel(surfacePath, returnNode=True)
    displayNode = self.model.GetDisplayNode()
    displayNode.SetBackfaceCulling(0)
    displayNode.SetEdgeVisibility(1)

  def updateModel(self):
    modelPoints = slicer.util.array(self.model.GetID())
    if modelPoints == None:
      # older slicer without direct access to point arrays
      p = self.model.GetPolyData().GetPoints().GetData()
      modelPoints = vtk.util.numpy_support.vtk_to_numpy(p)
    self.gridder.surface_grid()
    modelPoints[:] = numpy.array(self.gridder._points)
    self.model.GetPolyData().GetPoints().GetData().Modified()
    self.model.GetPolyData().GetPoints().Modified()

  def setFiducialListDisplay(self,fiducialList):
    displayNode = fiducialList.GetDisplayNode()
    # TODO: pick appropriate defaults
    # 135,135,84
    displayNode.SetTextScale(2.)
    displayNode.SetGlyphScale(5.)
    displayNode.SetGlyphTypeFromString('Sphere3D')
    displayNode.SetColor((0.6,0.6,0.2))
    displayNode.SetSelectedColor((1,1,0))
    #displayNode.GetAnnotationTextDisplayNode().SetColor((1,1,0))
    displayNode.SetVisibility(True)

  def createNodeFiducials(self,name='N'):
    """Add a fiducial for each node in the structure
    """

    markupsLogic = slicer.modules.markups.logic()
    originalActiveListID = markupsLogic.GetActiveListID()
    slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)

    # make the fiducial list if required
    fiducialListNodeID = markupsLogic.AddNewFiducialNode(name,slicer.mrmlScene)
    self.fiducialList = slicer.util.getNode(fiducialListNodeID)
    self.setFiducialListDisplay(self.fiducialList)

    # make this active so that the fids will be added to it
    markupsLogic.SetActiveListID(self.fiducialList)

    # make a fiducial for each node, indicating fixity
    # - index in fiducial list is equal to node index in _nodes list
    for node in self.structure._nodes:
      pu = node.pu()
      self.fiducialList.AddFiducial(*pu)
      fiducialIndex = self.fiducialList.GetNumberOfFiducials()-1

      self.fiducialList.SetNthFiducialLabel(fiducialIndex, name)
      nodeFixed = node._fixed.max() > 0
      self.fiducialList.SetNthFiducialSelected(fiducialIndex, not nodeFixed)
      self.fiducialList.SetNthMarkupLocked(fiducialIndex, not nodeFixed)

    # observe list for changes
    self.fiducialList.AddObserver( self.fiducialList.PointModifiedEvent, 
      lambda caller,event: self.onFiducialMoved(caller))
    self.fiducialList.AddObserver( self.fiducialList.PointEndInteractionEvent, 
        lambda caller,event: self.onFiducialEndMoving(caller))

    originalActiveList = slicer.util.getNode(originalActiveListID)
    if originalActiveList:
      markupsLogic.SetActiveListID(originalActiveList)
    slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

  def onFiducialMoved(self,fiducialList):
    """Callback when fiducialList's point has been changed."""
    if self._updatingNodeFiducials:
      return
    slicer.mrmlScene.StartState(slicer.mrmlScene.BatchProcessState)
    nodeCount = self.fiducialList.GetNumberOfFiducials()
    for nodeIndex in range(nodeCount):
      node = self.structure._nodes[nodeIndex]
      point = [0,]*3
      self.fiducialList.GetNthFiducialPosition(nodeIndex,point)
      node._u = numpy.array(point) - node._p
    self.updateFromStructure()
    slicer.mrmlScene.EndState(slicer.mrmlScene.BatchProcessState)

  def updateFromStructure(self):
    # update structure (TODO: save decomposed matrix in structure.py)
    self.structure.apply_bc()
    #self.structure.solve()
    self.structure.updateNodes()

    self.updateModel()

    # refresh node fiducials
    self._updatingNodeFiducials = True
    nodeCount = self.fiducialList.GetNumberOfFiducials()
    for nodeIndex in range(nodeCount):
      node = self.structure._nodes[nodeIndex]
      pu = node.pu()
      self.fiducialList.SetNthFiducialPosition(nodeIndex,*pu)
    self._updatingNodeFiducials = False


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
    self.test_TissueSimulation1()

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

    reload(festiv.structure)
    reload(festiv.element)
    reload(festiv.node)
    reload(festiv.meshing)
    reload(festiv.el_grid)

    # make a structure
    logic = TissueSimulationLogic()
    s = logic.structure
    iso20 = festiv.isomap.iso20()

    # add an element
    element = festiv.element.element20()
    s._elements.append(element)

    # create the nodes and make the element 40mm on a side
    for i in xrange(20):
      node = festiv.node.node()
      node._p = numpy.array(iso20.__unit_nodes__[i]) * 20
      s._nodes.append( node )
      element._nodes[i] = node

    # set fixed boundary conditions on the lower face
    for node in element.face_nodes(1):
      node._fixed.fill(1)

    if 0:
      # move all the top nodes up by 5
      for node in element.face_nodes(0):
        node._u.fill(0)
        node._u[2] = 5
        node._fixed.fill(1)

    # grab and move a node at the top corner by a fixed offset
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
    logic.createNodeFiducials()

    slicer.tissueLogic = logic

    self.delayDisplay('Test passed!')
