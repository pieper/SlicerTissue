#!/usr/bin/env python

#
# festiv
# finite element soft tissue interactive visualization
#
# pieper@isomics.com
# copyright 2009 All rights reserved
#
#
# meshing.py
# - utilties for creating meshes for a structure
#

import numpy, warnings

import festiv.structure
import festiv.element
import festiv.node

def glue_faces(structure, from_el, from_face, to_el, to_face):
  """Put a new element between two existing elements and have it share the face nodes with
     the specified elements.
     from_face on from_el becomes new element's face 0.
     to_face on to_el becomes new element's face 1.
     intermediate nodes are interpolated
     """

  # add the glue element
  new_el = festiv.element.element20()
  structure._elements.append(new_el)

  faces = new_el.__faces__
  # link to the shared nodes - need to go opposite direction 
  for i in xrange(8):
    from_node = from_el._nodes[ faces[from_face][8-i] ]  # uses wraparound
    new_el._nodes[ faces[0][i] ] = from_node
    to_node = to_el._nodes[ faces[to_face][8-i] ]  # uses wraparound
    new_el._nodes[ faces[1][i] ] = to_node

  # position middle nodes between corresponding face nodes
  for i in xrange(4):
    node = festiv.node.node()
    new_el._nodes[16+i] = node
    p0 = new_el._nodes[i]._p
    p1 = new_el._nodes[i+4]._p
    node._p = (p0 + p1) / 2.
    structure._nodes.append( node )

  from_el._shared_faces[from_face] = 1
  to_el._shared_faces[to_face] = 1
  new_el._shared_faces[0] = 1
  new_el._shared_faces[1] = 1

