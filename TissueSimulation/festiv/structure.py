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


  def add_Km_to_K(self, element):
    """Copy element stiffness matrix into structure's global stiffness matrix""" 
    nsize = self._nodes.__len__()
    for ncount in xrange(20):
      if not element._nodes[ncount]:
        continue
      nli = element._nodes[ncount]._node_list_index
      for i in xrange(20):
        if not element._nodes[i]:
          continue
        other_n = element._nodes[i]
        other_nli = other_n._node_list_index

        # copy x row
        self._K[ 0       + nli, 0       + other_nli ] += element._Km[ 0  + ncount, 0  + i ]
        self._K[ 0       + nli, nsize   + other_nli ] += element._Km[ 0  + ncount, 20 + i ]
        self._K[ 0       + nli, 2*nsize + other_nli ] += element._Km[ 0  + ncount, 40 + i ]

        # copy y row
        self._K[ nsize   + nli, 0       + other_nli ] += element._Km[ 20 + ncount, 0  + i ]
        self._K[ nsize   + nli, nsize   + other_nli ] += element._Km[ 20 + ncount, 20 + i ]
        self._K[ nsize   + nli, 2*nsize + other_nli ] += element._Km[ 20 + ncount, 40 + i ]

        # copy z row
        self._K[ 2*nsize + nli, 0       + other_nli ] += element._Km[ 40 + ncount, 0  + i ]
        self._K[ 2*nsize + nli, nsize   + other_nli ] += element._Km[ 40 + ncount, 20 + i ]
        self._K[ 2*nsize + nli, 2*nsize + other_nli ] += element._Km[ 40 + ncount, 40 + i ]


  def make_K(self):
    """create the global stiffness matrix""" 
    self._dirty = True
    self.establish_variables()
    for element in self._elements:
      element.calculate_stiffness()
      self.add_Km_to_K(element)
    self._dirty = False


  def apply_gravity(self, g):
    """apply a gravity field uniformly over the structure.
       node equivalent loads are calculated per element."""
    for element in self._elements:
      element.calculate_gravity(g)


  def apply_bc(self):
    """Apply the loading and displacement boundary condtions to the current K matrix"""
    nsize = len(self._nodes)
    ncount = 0
    for node in self._nodes:
      for dof in xrange(3):
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
    self._U = numpy.linalg.solve(self._K, self._R)
    self._Kinv = numpy.linalg.inv(self._K)
    # self._U = numpy.linalg.solve(self._K, self._R)
    self.updateNodes()

  def updateNodes(self):
    self._U = self._Kinv * self._R
    nsize = len(self._nodes)
    ncount = 0
    for node in self._nodes:
      for dof in xrange(3):
        if not node._fixed[dof]:
          i = nsize*dof + ncount
          node._u[dof] = self._U[i]
      ncount = ncount + 1

def _test():
  import festiv.structure


if __name__ == '__main__':
    _test()
