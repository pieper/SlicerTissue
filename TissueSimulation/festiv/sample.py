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


import festiv
import festiv.structure
import festiv.element
import festiv.el_grid
import festiv.node
import festiv.meshing

reload(festiv.structure)
reload(festiv.element)
reload(festiv.node)
reload(festiv.meshing)

# make a structure
s = festiv.structure.structure()
iso20 = festiv.isomap.iso20()

# add an element
element = festiv.element.element20()
s._elements.append(element)

# create the nodes and put them at the default location
for i in xrange(20):
  node = festiv.node.node()
  node._p = iso20.__unit_nodes__[i]
  s._nodes.append( node )
  element._nodes[i] = node

# set fixed boundary conditions on the lower face
for node in element.face_nodes(1):
  node._fixed.fill(1)

# make another element, offset in z
element = festiv.element.element20()
s._elements.append(element)
for i in xrange(20):
  node = festiv.node.node()
  node._p = iso20.__unit_nodes__[i] + numpy.array([0, 0, 4])
  s._nodes.append( node )
  element._nodes[i] = node

# now put an element between the other two that shares its faces
festiv.meshing.glue_faces(s, s._elements[1], 1, s._elements[0], 0)

experiment = 'gravity'

if experiment == 'cornerpull':
  # grab and move a node at the top corner by a fixed offset
  node = s._elements[1]._nodes[0]
  node._u = numpy.array([1,1,1])
  node._fixed.fill(1)

elif experiment == 'toppull':
  off = 5
  # move all the top nodes up by offset
  for node in s._elements[1].face_nodes(0):
    node._u.fill(0)
    node._u[1] = off
    node._fixed.fill(1)

elif experiment == 'gravity':
  g = numpy.matrix( [1e4, 0, 0] )
  s.apply_gravity(g)


# create the stiffness matrix
s.make_K()
s.apply_bc()
s.solve()

g = festiv.el_grid.gridder(s)
g._steps = (8,8,8,8,8,8)
g.surface_grid()
g.write_grid('/Users/pieper/data/caps/glue.vtk')

