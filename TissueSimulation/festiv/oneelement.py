#!/usr/bin/env python

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

if 0:
  # move all the top nodes up by 5
  for node in element.face_nodes(0):
    node._u.fill(0)
    node._u[2] = 5
    node._fixed.fill(1)

# grab and move a node at the top corner by a fixed offset
node = s._elements[0]._nodes[0]
node._u = numpy.array([1,1,1])
node._fixed.fill(1)


# create the stiffness matrix
s.make_K()
s.apply_bc()
s.solve()

g = festiv.el_grid.gridder(s)
g._steps = (8,8,8,8,8,8)
g.surface_grid()
g.write_grid('/data/caps/one.vtk')

# write cvs files of the node locations

f = open('/Users/pieper/tmp/pu.fcsv', 'w')
f.write('# columns = label,x,y,z,sel,vis\n')
n = 0
for node in s._nodes:
  pu = node.pu()
  f.write('pu%d, %g, %g, %g, 1, 1\n' % (n, pu[0], pu[1], pu[2]) )
  n = n + 1
f.close()

f = open('/Users/pieper/tmp/p.fcsv', 'w')
f.write('# columns = label,x,y,z,sel,vis\n')
n = 0
for node in s._nodes:
  p = node._p
  f.write('p%d, %g, %g, %g, 1, 1\n' % (n, p[0], p[1], p[2]) )
  n = n + 1
f.close()



