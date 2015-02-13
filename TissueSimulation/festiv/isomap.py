#!/usr/bin/env python

#
# festiv
# finite element soft tissue interactive visualization
#
# pieper@isomics.com
# copyright 2009 All rights reserved
#
#
# isomap.py
# - core routines for 20 node isoparametric elements
# - based on Bathe-style interpolation
#

import numpy, warnings

class iso20:
  """
  iso20()

  core routines for manipulating a 20 node element
  (tri-parabolic)

  See Bathe 1982 page 201 figure 5.6

  Parameters
  ----------

  See Also
  --------

  Acknowledgements
  --------
  Thanks to David T. Chen for an earlier version of the interpolation function code in C.

  Examples
  --------
  >>> i = iso20()
  >>> i.h(0,0,0,1)
  -0.25

  """

  # home positions of nodes in isoparametric space
  __unit_nodes__ = (
    ( 1,  1,  1), (-1,  1,  1), (-1, -1,  1), ( 1, -1,  1),
    ( 1,  1, -1), (-1,  1, -1), (-1, -1, -1), ( 1, -1, -1),
    ( 0,  1,  1), (-1,  0,  1), ( 0, -1,  1), ( 1,  0,  1),
    ( 0,  1, -1), (-1,  0, -1), ( 0, -1, -1), ( 1,  0, -1),
    ( 1,  1,  0), (-1,  1,  0), (-1, -1,  0), ( 1, -1,  0)
  )

  # constructor
  def __init__(self):
    # active nodes are ones that correspond to real nodes of the element
    # (i.e. you can leave out nodes for compatibility by setting values here to 0)
    self.__activeNodes__ = numpy.ones(20);

  # interpolation core functions G and dG

  def G(self, beta, beta_i):
    """Implementation of the G interpolation"""
    if beta_i == 0:
      return 1. - (beta**2)
    elif beta_i == 1:
      return .5 * (1 + beta)
    elif beta_i == -1:
      return .5 * (1 - beta)
    else:
      warnings.warn("G: bad beta_i %g" % beta_i)
      return 0.

  def dG(self, beta, beta_i):
    """Implementation of the dG interpolation"""
    if beta_i == 0:
      return -2 * beta
    elif beta_i == 1:
      return .5
    elif beta_i == -1:
      return -.5
    else:
      warnings.warn("G: bad beta_i %g" % beta_i)
      return 0.


  # interpolation helper functions g at each node i 
  # plus derivatives with respect to r, s, and t
  
  def g(self, r, s, t, i):
    """g function for node i at r,s,t"""
    if i > 19 or i < 0:
      warnings.warn("g: bad i %d" % i)
      return 0.
    if not self.__activeNodes__[i]:
      return 0.
    return (self.G(r, self.__unit_nodes__[i][0]) *
            self.G(s, self.__unit_nodes__[i][1]) *
            self.G(t, self.__unit_nodes__[i][2]) )

  def dgdr(self, r, s, t, i):
    """derivative of g with respect to r for node i at r,s,t"""
    if i > 19 or i < 0:
      warnings.warn("dgdr: bad i %d" % i)
      return 0
    if not self.__activeNodes__[i]:
      return 0.
    return (self.dG(r, self.__unit_nodes__[i][0]) *
            self. G(s, self.__unit_nodes__[i][1]) *
            self. G(t, self.__unit_nodes__[i][2]) )

  def dgds(self, r, s, t, i):
    """derivative of g with respect to s for node i at r,s,t"""
    if i > 19 or i < 0:
      warnings.warn("dgds: bad i %d" % i)
      return 0
    if not self.__activeNodes__[i]:
      return 0.
    return (self. G(r, self.__unit_nodes__[i][0]) *
            self.dG(s, self.__unit_nodes__[i][1]) *
            self. G(t, self.__unit_nodes__[i][2]) )

  def dgdt(self, r, s, t, i):
    """derivative of g with respect to t for node i at r,s,t"""
    if i > 19 or i < 0:
      warnings.warn("dgdt: bad i %d" % i)
      return 0
    if not self.__activeNodes__[i]:
      return 0.
    return (self. G(r, self.__unit_nodes__[i][0]) *
            self. G(s, self.__unit_nodes__[i][1]) *
            self.dG(t, self.__unit_nodes__[i][2]) )


  # blending function

  def blend(self, r, s, t, i, g):
    "blending function at r,s,t for node i of function g"
    if i > 19 or i < 0:
      warnings.warn("dgdt: bad i %d" % i)
      return 0
    ans = g(r,s,t,i)
    if i == 0:
      ans += -.5 * ( g(r,s,t, 8) + g(r,s,t,11) + g(r,s,t,16) )
    elif i == 1:
      ans += -.5 * ( g(r,s,t, 8) + g(r,s,t, 9) + g(r,s,t,17) )
    elif i == 2:
      ans += -.5 * ( g(r,s,t, 9) + g(r,s,t,10) + g(r,s,t,18) )
    elif i == 3:
      ans += -.5 * ( g(r,s,t,10) + g(r,s,t,11) + g(r,s,t,19) )
    elif i == 4:
      ans += -.5 * ( g(r,s,t,12) + g(r,s,t,15) + g(r,s,t,16) )
    elif i == 5:
      ans += -.5 * ( g(r,s,t,12) + g(r,s,t,13) + g(r,s,t,17) )
    elif i == 6:
      ans += -.5 * ( g(r,s,t,13) + g(r,s,t,14) + g(r,s,t,18) )
    elif i == 7:
      ans += -.5 * ( g(r,s,t,14) + g(r,s,t,15) + g(r,s,t,19) )
    return ans
    
  def h(self, r,s,t,i):
    # evaluation of interpolation function i at r,s,t
    return self.blend(r,s,t,i,self.g)

  def dhdr(self, r,s,t,i):
    # evaluation of derivative of interpolation function i at r,s,t with respect to r
    return self.blend(r,s,t,i,self.dgdr)

  def dhds(self, r,s,t,i):
    # evaluation of derivative of interpolation function i at r,s,t with respect to s
    return self.blend(r,s,t,i,self.dgds)

  def dhdt(self, r,s,t,i):
    # evaluation of derivative of interpolation function i at r,s,t with respect to t
    return self.blend(r,s,t,i,self.dgdt)


def _test():
  i = iso20()
  # known values at center of cube
  for n in xrange(8):
    assert i.h(0,0,0, n) == -.25
  for n in xrange(9,20):
    assert i.h(0,0,0, n) == .25
  # interpolation function n is 1 at node n's coordinates 
  # and 0 at all other interpolation functions are 0
  for n in xrange(0,20):
    r,s,t = i.__unit_nodes__[n]
    assert i.h(r,s,t, n) == 1.
    for nn in xrange(0,20):
      if nn != n:
        assert i.h(r,s,t, nn) == 0.

if __name__ == '__main__':
    _test()
