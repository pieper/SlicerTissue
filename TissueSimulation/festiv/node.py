#!/usr/bin/env python

#
# festiv
# finite element soft tissue interactive visualization
#
# pieper@isomics.com
# copyright 2009 All rights reserved
#
#
# node.py
# - representation of a node in a finite element structure
#

import numpy, warnings

class node:
  """
  node()

  3D Node in a structure

  Parameters
  ----------

  See Also
  --------

  Acknowledgements
  --------

  Examples
  --------
  >>> n = node()

  """

  # constructor
  def __init__(self,name=''):
    self._name = name
    # initial position of the node
    self._p = numpy.zeros(3);
    # current displacement of the node
    self._u = numpy.zeros(3);
    # current load on the node
    self._r = numpy.zeros(3);
    # fixed state of the 3 dofs for the node
    self._fixed = numpy.zeros(3);
    # elements this node is part of
    self._elements = []
    # node that this node is constrained to with a suture constraint
    self._sutured_to = []
    # where this node is located in the current structure's node list
    self._node_list_index = []

  def pu(self):
    """current position of the node (original position p plus displacement u)"""
    return self._p + self._u

def _test():
  import festiv.node
  n = festiv.node.node()
  n._p = numpy.array([1, 0, 0])
  n._u = numpy.array([0, 1, 0])
  assert all(n.pu().__eq__(numpy.array([1,1,0])))


if __name__ == '__main__':
    _test()
