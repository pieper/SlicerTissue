#!/usr/bin/env python

#
# festiv
# finite element soft tissue interactive visualization
#
# pieper@isomics.com
# copyright 2009 All rights reserved
#
#
# el_grid.py
# - utilities to calculate polygonal grids of element faces
#


import numpy, warnings
import festiv.structure

class gridder:
  """
  gridder()

  class to generate grids for a structure

  Parameters
  ----------

  See Also
  --------

  Acknowledgements
  --------

  Examples
  --------
  >>> g = gridder()


  """

  # for each face:
  #  r_start, r_end, r_dir
  #  s_start, s_end, s_dir
  #  t_start, t_end, t_dir
  __face_increments__ = ( 
          ( -1.0,  1.0,  1.0, # /* top */
            -1.0,  1.0,  1.0,
             1.0,  1.0,  0.0 ),
          (  1.0, -1.0,  1.0, # /* bottom */
            -1.0,  1.0,  1.0,
            -1.0, -1.0,  0.0 ),
          (  1.0, -1.0,  1.0, # /* right */
             1.0,  1.0,  0.0,
            -1.0,  1.0,  1.0 ),
          ( -1.0, -1.0,  0.0, # /* back */
             1.0, -1.0,  1.0,
            -1.0,  1.0,  1.0 ),
          ( -1.0,  1.0,  1.0, # /* left */
            -1.0, -1.0,  0.0,
            -1.0,  1.0,  1.0 ),
          (  1.0,  1.0,  0.0, # /* front */
            -1.0,  1.0,  1.0,
            -1.0,  1.0,  1.0 ))


  # constructor
  def __init__(self, structure, name='surface'):
    # the structure we are gridding
    self._structure = structure
    # name for the outputs
    self._name = name
    # an instance of the isomap interpolator
    self._iso20 = festiv.isomap.iso20()
    # number of steps per face (typically all the same)
    self._steps = (2,2,2,2,2,2)
    # alpha parameter is used when mapping stresses and strains to RGB
    self._alpha = 10
    # containers for the points and polygons
    self._points = []
    self._polygons = []


  def interpolate(self,nodes,array,r,s,t):
    ans = 0.
    for i in range(20):
      if not nodes[i]:
        continue
      ans = ans + array[i] * self._iso20.h(r,s,t,i)
    return ans.tolist()[0][0]


  def surface_grid(self, configuration='displaced'):
    """Fill the points and polygons arrays for the current structure"""
    self._points = []
    self._polygons = []
    x_array = numpy.matrix( numpy.zeros([20,1]) )
    y_array = numpy.matrix( numpy.zeros([20,1]) )
    z_array = numpy.matrix( numpy.zeros([20,1]) )

    for el in self._structure._elements:
      # first create the node point vectors as input for the interpolator
      if configuration == 'displaced':
        el.load_disp_xyz_arrays(x_array, y_array, z_array)
      else: 
        el.load_xyz_arrays(x_array, y_array, z_array)

      # now iterate on faces and add points and polygons
      for face in range(6):
        if el._shared_faces[face]:
          continue
        steps = self._steps[face]
        r_start,r_end,r_dir,s_start,s_end,s_dir,t_start,t_end,t_dir = self.__face_increments__[face]
        r_inc = r_dir * (r_end - r_start)/steps
        s_inc = s_dir * (s_end - s_start)/steps
        t_inc = t_dir * (t_end - t_start)/steps
        if r_inc == 0.:
          r_inc = 1.
          rstep = steps+1
        else:
          rstep = 1
        if s_inc == 0.:
          s_inc = 1.
          sstep = steps+1
        else:
          sstep = 1
        if t_inc == 0.:
          t_inc = 1.
          tstep = steps+1
        else:
          tstep = 1

        # add the points for this face
        pis = [] # point indices for added points
        r = r_start
        for rcount in range(0,steps+1,rstep):
          s = s_start
          for scount in range(0,steps+1,sstep):
            t = t_start
            for tcount in range(0,steps+1,tstep):
              x = self.interpolate(el._nodes,x_array,r,s,t)
              y = self.interpolate(el._nodes,y_array,r,s,t)
              z = self.interpolate(el._nodes,z_array,r,s,t)
              pis.append(len(self._points))
              self._points.append([x,y,z])
              t = t + t_inc
            s = s + s_inc
          r = r + r_inc

        # add the polygons that form this face
        for col in range(steps):
          for row in range(steps):
            ll = (col     * (steps+1)) + row
            ul = (col     * (steps+1)) + row + 1
            ur = ((col+1) * (steps+1)) + row + 1
            lr = ((col+1) * (steps+1)) + row
            self._polygons.append([pis[ll],pis[ul],pis[ur],pis[lr]])
            

  def write_grid(self, fileName):
    f = open(fileName, 'w')
    f.write('# vtk DataFile Version 3.0\n')
    f.write('vtk output\n')
    f.write('ASCII\n')
    f.write('DATASET POLYDATA\n')
    f.write('POINTS %d float\n' % len(self._points))
    for point in self._points:
      f.write( '%g %g %g\n' % (point[0], point[1], point[2]) )
    nPolys = len(self._polygons)
    f.write('POLYGONS %d %d\n' % (nPolys, 5*nPolys))
    for polygon in self._polygons:
      f.write( '4 %d %d %d %d\n' % (polygon[0], polygon[1], polygon[2], polygon[3]) )
    f.close()

def _test():
  import festiv.el_grid
  g = festiv.el_grid.gridder(s)
  g.surface_grid()
  g.write_grid('/tmp/grid.vtk')


if __name__ == '__main__':
    _test()
