#!/usr/bin/env python

#
# festiv
# finite element soft tissue interactive visualization
#
# pieper@isomics.com
# copyright 2009 All rights reserved
#
#
# element.py
# - representation of a 20 node element in a finite element structure
#

import numpy, warnings
import numpy.linalg

import festiv.isomap

class element20:
  """
  element20()

  20 node isoparametric element

  Parameters
  ----------

  See Also
  --------

  Acknowledgements
  --------

  Examples
  --------
  >>> e = element()

  """

  # Gauss Quadrature points and weights.  
  # From the 1987 CRC Handbook, page 462
  __quadpoints__ = ( (  0.7745967, 0.5555556), 
                     (  0.0000000, 0.8888889),
                     ( -0.7745967, 0.5555556) )


  # list of the elements that make up each of the faces of the element
  # note: node list wraps around for use in generating ccw list
  __faces__ = ( ( 0, 11, 3, 10, 2, 9, 1, 8, 0 ),
                ( 4, 12, 5, 13, 6, 14, 7, 15, 4 ),
                ( 0, 8, 1, 17, 5, 12, 4, 16, 0 ),
                ( 1, 9, 2, 18, 6, 13, 5, 17, 1 ),
                ( 2, 10, 3, 19, 7, 14, 6, 18, 2 ),
                ( 0, 16, 4, 15, 7, 19, 3, 11, 0 ) )


  # constructor
  def __init__(self, name=''):
    self._name = name
    # nodes that define this element
    self._nodes = []
    for i in range(20):
      self._nodes.append(False)
    # material properties for this element
    self._youngs_modulus = 1.e6
    self._poissons_ratio = 0.3
    # list of faces of this element that are shared with other elements
    self._shared_faces = numpy.zeros(6)
    # element stiffness matrix
    self._Km = numpy.eye(60)
    # dirty flag is true when this element needs to be recalculated
    self._Km_dirty = True
    # temp variables for calculating Bm
    self._hhat = numpy.matrix( numpy.zeros([3,20]) )
    self._jac = numpy.matrix( numpy.zeros([3,3]) )
    self._jinv = numpy.matrix( numpy.zeros([3,3]) )
    self._gamma = numpy.matrix( numpy.zeros([3,20]) )


  def bbox(self, face=-1):
    """Return the bounding box of the given face or full element in displaced configuration"""
    bbnodes = self._nodes
    if face != -1:
      bbnodes = self.face_nodes(face)
    min = bbnodes[0].pu()
    max = bbnodes[0].pu()
    for node in bbnodes[1:]:
      pu = node.pu()
      for i in range(3):
        if pu[i] < min[i]:
          min[i] = pu[i]
        if pu[i] > max[i]:
          max[i] = pu[i]
    return min, max


  def load_xyz_arrays(self, x_array, y_array, z_array):
    for i in range(20):
      if not self._nodes[i]:
        x_array[i] = y_array[i] = z_array[i] = 0.
      else:
        x_array[i] = self._nodes[i]._p[0]
        y_array[i] = self._nodes[i]._p[1]
        z_array[i] = self._nodes[i]._p[2]

  def load_disp_xyz_arrays(self, x_array, y_array, z_array):
    for i in range(20):
      if not self._nodes[i]:
        x_array[i] = y_array[i] = z_array[i] = 0.
      else:
        x_array[i] = self._nodes[i].pu()[0]
        y_array[i] = self._nodes[i].pu()[1]
        z_array[i] = self._nodes[i].pu()[2]


  def fill_C(self, real_C):
    """Fill the stress-strain relation tensor based on material properties"""
    y = self._youngs_modulus
    p = self._poissons_ratio
    C11 = ( y * (1-p) ) / ( (1 + p) * (1 - 2*p) )
    C12 = ( y *    p  ) / ( (1 + p) * (1 - 2*p) )
    C44 = y / (2 + 2*p)

    real_C[0,0] = C11
    real_C[1,1] =    C11
    real_C[2,2] =       C11
    real_C[3,3] =          C44
    real_C[4,4] =             C44
    real_C[5,5] =                C44

    real_C[0,1] =    C12
    real_C[0,2] =       C12
    real_C[1,2] =       C12
    real_C[1,0] = C12
    real_C[2,0] = C12
    real_C[2,1] =    C12


  def calculate_J(self, x_array, y_array, z_array, r, s, t, jac, jinv):
    """find jacobian, inverse, and determinant"""
    iso20 = festiv.isomap.iso20()
    for i in range(20):
      if self._nodes[i]:
        dhdr = iso20.dhdr(r,s,t,i)
        dhds = iso20.dhds(r,s,t,i)
        dhdt = iso20.dhdt(r,s,t,i)

        jac[0,0] += x_array[i]*dhdr
        jac[1,0] += x_array[i]*dhds
        jac[2,0] += x_array[i]*dhdt

        jac[0,1] += y_array[i]*dhdr
        jac[1,1] += y_array[i]*dhds
        jac[2,1] += y_array[i]*dhdt

        jac[0,2] += z_array[i]*dhdr
        jac[1,2] += z_array[i]*dhds
        jac[2,2] += z_array[i]*dhdt

    jinv[:] = numpy.linalg.inv(jac)
    detj = numpy.linalg.det(jac)
    return detj

  def calculate_H_T(self, x_array, y_array, z_array, r, s, t, H_T):
    """Calculate the H_T element displacement matrix for the current element at the point r,s,t"""
    iso20 = festiv.isomap.iso20()
    for i in range(20):
      H_T[   i,0] = iso20.h(r,s,t,i)
      H_T[20+i,1] = iso20.h(r,s,t,i)
      H_T[40+i,2] = iso20.h(r,s,t,i)


  def calculate_B(self, x_array, y_array, z_array, r, s, t, Bm):
    """Calculate the B matrix for the current element at the point r,s,t; return determinant of jacobian at r,s,t"""

    # reset temp variables
    self._hhat[:] = 0.
    self._jac[:] = 0.
    self._jinv[:] = 0.
    self._gamma[:] = 0.
    Bm[:] = 0.
    iso20 = festiv.isomap.iso20()

    # fill hhat
    for i in range(20):
      self._hhat[0,i] = iso20.dhdr(r,s,t,i)
      self._hhat[1,i] = iso20.dhds(r,s,t,i)
      self._hhat[2,i] = iso20.dhdt(r,s,t,i)

    detj = self.calculate_J(x_array, y_array, z_array, r, s, t, self._jac, self._jinv)
    
    self._gamma = self._jinv * self._hhat

    # now fill Bm
    for i in range(20):
      Bm[0,     i] = self._gamma[0,i] 
      Bm[1,20 + i] = self._gamma[1,i] 
      Bm[2,40 + i] = self._gamma[2,i] 

      Bm[3,     i] = self._gamma[1,i] 
      Bm[3,20 + i] = self._gamma[0,i] 

      Bm[4,20 + i] = self._gamma[2,i] 
      Bm[4,40 + i] = self._gamma[1,i] 

      Bm[5,     i] = self._gamma[2,i] 
      Bm[5,40 + i] = self._gamma[0,i] 

    return detj


  def calculate_stiffness(self):
    """Calculate the stiffness matrix for the current element"""
    if not self._Km_dirty:
      return

    self._Km = numpy.matrix( numpy.zeros([60,60]) )
    x_array = numpy.matrix( numpy.zeros([20,1]) )
    y_array = numpy.matrix( numpy.zeros([20,1]) )
    z_array = numpy.matrix( numpy.zeros([20,1]) )
    self.load_xyz_arrays( x_array, y_array, z_array )

    kmtmp = numpy.matrix( numpy.zeros([60,60]) )
    real_B = numpy.matrix( numpy.zeros([6,60]) )
    real_C = numpy.matrix( numpy.zeros([6,6]) )

    #TODO: set active nodes based on node list
    
    self.fill_C(real_C)
    for r in range(3):
      for s in range(3):
        for t in range(3):
          qp = self.__quadpoints__
          detj = self.calculate_B( x_array, y_array, z_array, qp[r][0], qp[s][0], qp[t][0], real_B )
          kmtmp = real_B.T * real_C * real_B
          kmtmp *= detj * qp[r][1] * qp[s][1] * qp[t][1] 
          self._Km += kmtmp


  def calculate_gravity(self, g):
    x_array = numpy.matrix( numpy.zeros([20,1]) )
    y_array = numpy.matrix( numpy.zeros([20,1]) )
    z_array = numpy.matrix( numpy.zeros([20,1]) )
    self.load_xyz_arrays( x_array, y_array, z_array )
    H_T = numpy.matrix( numpy.zeros([60,3]) )
    Re_g = numpy.matrix( numpy.zeros([60,1]) )

    # calculate the nodal equivelent forces for the gravity load
    # by integrating of the quadrature points
    for r in range(3):
      for s in range(3):
        for t in range(3):
          self._jac[:] = 0.
          self._jinv[:] = 0.
          qp = self.__quadpoints__
          detj = self.calculate_J(x_array, y_array, z_array, qp[r][0], qp[s][0], qp[t][0], self._jac, self._jinv)
          self.calculate_H_T(x_array, y_array, z_array, qp[r][0], qp[s][0], qp[t][0], H_T)
          Re_g_tmp = H_T * g.T
          Re_g += detj * qp[r][1] * qp[s][1] * qp[2][1] * Re_g_tmp # should multiply by mass density rho(r,s,t)
    # copy the node loads back into the nodes
    for i in range(20):
      if self._nodes[i]:
        for dof in range(3):
          self._nodes[i]._r[dof] = Re_g[20*dof + i]


  def face_nodes(self, face):
    nodes = []
    for nodei in self.__faces__[face][0:8]:  # account for wraparound
      nodes.append(self._nodes[nodei])
    return nodes


  def move_face(self, face, displacement):
    for node in self.face_nodes(face):
      self.nodes[nodei]._p += displacement


def _test():
  import festiv.element


if __name__ == '__main__':
    _test()
