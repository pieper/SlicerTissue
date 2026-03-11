#!/usr/bin/env python

#
# festiv
# finite element soft tissue interactive visualization
#
# pieper@isomics.com
# copyright 2009 All rights reserved
#
#
# structure.py
# - representation of a finite element structure
#

import numpy, warnings

import festiv.isomap
import festiv.node
import festiv.element

class structure:
  """
  structure()

  A mechanical structure modeled as nodes and elements.
  Currently handles only 20 node isoparametric elements.

  Parameters
  ----------

  See Also
  --------

  Acknowledgements
  --------

  Examples
  --------
  >>> s = structure()

  """

  # constructor
  def __init__(self,name=''):
    self._name = name
    # nodes that define this structure
    self._nodes = []
    # elements that define this structure
    self._elements = []
    # flag that structure matrices need to be recalculated
    self._dirty = True
    # flag that structure fixity at the nodes is dirty and some matrix rearranging is required
    self._dirty_fixity = False
    # overall stiffness matrix for this structure
    self._K = numpy.matrix([])
    # decomposed stiffness matrix
    self._this_K = numpy.matrix([])
    # current displacements
    self._U = numpy.matrix([])
    # current loads
    self._R = numpy.matrix([])
    # column swap vector
    self._IX = numpy.matrix([])
    # number of degrees of freedom in the current structure
    self._N = 0


  def establish_variables(self):
    """Initialize variables for solving the current configuration of the structure"""
    if not self._dirty:
      return
    N = 3 * self._nodes.__len__()
    self._N = N
    self._K = numpy.matrix( numpy.zeros([N,N]) )
    self._this_K = numpy.matrix( numpy.zeros([N,N]) )
    self._U = numpy.matrix( numpy.zeros([N,1]) )
    self._R = numpy.matrix( numpy.zeros([N,1]) )
    self._IX = numpy.matrix( numpy.zeros([N,1]) )
    i = 0
    for node in self._nodes:
      node._node_list_index = i
      i = i + 1
    self._dirty = False


  def add_Km_to_K(self, element):
    """Copy element stiffness matrix into structure's global stiffness matrix""" 
    nsize = self._nodes.__len__()
    for ncount in range(20):
      if not element._nodes[ncount]:
        continue
      nli = element._nodes[ncount]._node_list_index
      for i in range(20):
        if not element._nodes[i]:
          continue
        other_n = element._nodes[i]
        other_nli = other_n._node_list_index

        # Correctly map local element DOFs to global DOFs
        for dof_row in range(3):
          global_row = dof_row * nsize + nli
          local_row_offset = dof_row * 20
          for dof_col in range(3):
            global_col = dof_col * nsize + other_nli
            local_col_offset = dof_col * 20
            self._K[global_row, global_col] += element._Km[local_row_offset + ncount, local_col_offset + i]


  def make_K(self):
    """create the global stiffness matrix""" 
    self.establish_variables()
    self._K.fill(0)
    for element in self._elements:
      element.calculate_stiffness()
      self.add_Km_to_K(element)


  def apply_gravity(self, g):
    """apply a gravity field uniformly over the structure.
       node equivalent loads are calculated per element."""
    for element in self._elements:
      element.calculate_gravity(g)


  def apply_bc(self):
    """Apply the loading and displacement boundary condtions to the current K matrix"""
    self.establish_variables()
    nsize = len(self._nodes)
    ncount = 0
    for node in self._nodes:
      for dof in range(3):
        i = nsize*dof + ncount
        if not node._fixed[dof]:
          # not fixed: apply load to right hand side vector
          self._R[i] = node._r[dof]
        else:
          # is fixed: apply displacement and set corresponding equations to identity
          self._R[i] = node._u[dof]
          self._K[i].fill(0)
          self._K[i,i] = 1
      # TODO: apply suture constraints
      ncount = ncount + 1

  def solve(self):
    """Solve for the displacements and copy them back to the nodes"""
    self.establish_variables()
    self._U = numpy.linalg.solve(self._K, self._R)
    #self._Kinv = numpy.linalg.inv(self._K)
    self.updateNodes()

  def updateNodes(self):
    #self._U = self._Kinv * self._R
    nsize = len(self._nodes)
    ncount = 0
    for node in self._nodes:
      for dof in range(3):
        if not node._fixed[dof]:
          i = nsize*dof + ncount
          node._u[dof] = self._U[i]
      ncount = ncount + 1

def _test():
  import festiv.structure


if __name__ == '__main__':
    _test()
