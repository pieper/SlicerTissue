import numpy
import vtk

slicer.util.pip_install("usd-core")
from pxr import Usd, Sdf, UsdGeom

toMilli = 1000.

filepath = "/Users/pieper/slicer/latest/SlicerTissue/Experiments/warp/example_dem.usd"

stage = Usd.Stage.Open(filepath)

pointInstancerPath = "/root/points"
pointInstancerPrim = stage.GetPrimAtPath(pointInstancerPath)
pointInstancer = UsdGeom.PointInstancer(pointInstancerPrim)
positions = pointInstancer.GetPositionsAttr()
timeSamples = positions.GetTimeSamples()

pointsByTime = {}
for t in timeSamples:
    pointsByTime[t] = numpy.array(positions.Get(t))

vtk_points = vtk.vtkPoints()
vtk_points.SetData(vtk.util.numpy_support.numpy_to_vtk(toMilli * pointsByTime[1]))

polydata = vtk.vtkPolyData()
polydata.SetPoints(vtk_points)
    
vertices = vtk.vtkCellArray()
for i in range(polydata.GetNumberOfPoints()):
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(i)
polydata.SetVerts(vertices)


sphere_source = vtk.vtkSphereSource()
sphere_source.SetRadius(100)
sphere_source.SetThetaResolution(2)
sphere_source.SetPhiResolution(2)


glyph3D = vtk.vtkGlyph3D()
glyph3D.SetSourceConnection(sphere_source.GetOutputPort())
glyph3D.SetInputData(polydata)
    
glyph3D.SetScaleModeToDataScalingOff()
glyph3D.Update()


pointsNode = slicer.mrmlScene.GetFirstNodeByName("points")
if not pointsNode:
    pointsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
    pointsNode.SetName("points")
    pointsNode.CreateDefaultDisplayNodes()

pointsNode.SetAndObserveMesh(glyph3D.GetOutput())

for t in timeSamples:
    vtk_points.SetData(vtk.util.numpy_support.numpy_to_vtk(toMilli * pointsByTime[t]))
    polydata.Modified()
    glyph3D.Update()
    slicer.app.processEvents()

