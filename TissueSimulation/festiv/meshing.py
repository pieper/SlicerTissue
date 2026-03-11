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
import festiv.isomap

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

  # Map from_el's face to new_el's face 0
  from_face_indices = from_el.__faces__[from_face][:8]
  new_el_face0_indices = new_el.__faces__[0][:8]
  for i in range(8):
      shared_node = from_el._nodes[from_face_indices[i]]
      new_el._nodes[new_el_face0_indices[i]] = shared_node

  # Map to_el's face to new_el's face 1, with reversed mapping for compatibility
  # This uses the same explicit mapping logic from the successful two-element test.
  to_face_map = { 0: 6, 11: 13, 3: 5, 10: 12, 2: 4, 9: 15, 1: 7, 8: 14 }
  for to_el_idx, new_el_idx in to_face_map.items():
      shared_node = to_el._nodes[to_el_idx]
      new_el._nodes[new_el_idx] = shared_node

  # position middle nodes between corresponding face nodes
  # These are nodes 16, 17, 18, 19, which connect corner nodes (0,4), (1,5), (2,6), (3,7)
  for i in range(4): 
    node = festiv.node.node()
    p0 = new_el._nodes[i]._p      # Position of corner node on face 0
    p1 = new_el._nodes[i+4]._p    # Position of corresponding corner node on face 1
    node._p = (p0 + p1) / 2.
    new_el._nodes[16+i] = node    # Assign new node to the correct middle-layer slot
    structure._nodes.append( node )

  structure._dirty = True
  from_el._shared_faces[from_face] = 1
  to_el._shared_faces[to_face] = 1
  new_el._shared_faces[0] = 1
  new_el._shared_faces[1] = 1

def create_layered_grid(structure, grid_dims, element_size, layer_defs):
    """
    Creates a multi-layered grid of 20-node hexahedral elements.

    This function builds a slab of tissue by stacking layers of finite elements.
    It first generates a 2D grid of nodes and then extrudes it downwards for
    each layer defined in layer_defs.

    Args:
        structure (festiv.structure.structure): The structure to add elements and nodes to.
        grid_dims (tuple): A tuple (dim_x, dim_y) for the number of elements in the grid.
        element_size (numpy.array): A numpy array [size_x, size_y] for the dimensions of each element in the grid plane.
        layer_defs (list): A list of dictionaries, where each dictionary defines a layer
                           with 'thickness', 'youngs_modulus', and 'poissons_ratio'.
    """
    dim_x, dim_y = grid_dims
    size_x, size_y = element_size
    iso20 = festiv.isomap.iso20()

    # --- Node Generation ---
    # A 3D grid of nodes that will be used to define the elements.
    # Dimensions are (dim_x*2+1, dim_y*2+1, num_layers*2+1)
    # We use 2*dim+1 because 20-node elements have nodes at corners and edge midpoints.
    num_layers = len(layer_defs)
    node_grid = numpy.full((dim_x * 2 + 1, dim_y * 2 + 1, num_layers * 2 + 1), None, dtype=object)

    z = 0.0  # Start at z=0 for the top surface
    for l_idx in range(num_layers + 1):
        z_level_nodes = 2 * l_idx
        z_level_mid_nodes = 2 * l_idx - 1

        # Create nodes for the current horizontal plane (z-level)
        for j in range(dim_y * 2 + 1):
            for i in range(dim_x * 2 + 1):
                # Only create nodes for the top layer or if they haven't been created by the layer below
                if l_idx == 0 or node_grid[i, j, z_level_nodes] is None:
                    node = festiv.node.node()
                    node._p = numpy.array([i * size_x / 2.0, j * size_y / 2.0, z])
                    structure._nodes.append(node)
                    node_grid[i, j, z_level_nodes] = node

        if l_idx > 0:
            # Create mid-layer nodes between the current z-level and the one above
            prev_z = z + layer_defs[l_idx-1]['thickness']
            mid_z = (z + prev_z) / 2.0
            for j in range(dim_y * 2 + 1):
                for i in range(dim_x * 2 + 1):
                    node = festiv.node.node()
                    node._p = numpy.array([i * size_x / 2.0, j * size_y / 2.0, mid_z])
                    structure._nodes.append(node)
                    node_grid[i, j, z_level_mid_nodes] = node

        # Update z for the next layer's bottom plane
        if l_idx < num_layers:
            z -= layer_defs[l_idx]['thickness']

    # --- Element Generation ---
    for l_idx, layer_info in enumerate(layer_defs):
        for j in range(dim_y):
            for i in range(dim_x):
                element = festiv.element.element20()
                element._youngs_modulus = layer_info['youngs_modulus']
                element._poissons_ratio = layer_info['poissons_ratio']
                structure._elements.append(element)

                # --- Faithful translation of the C code's element creation logic ---
                # The C code iterates through a 2D polygon's vertices and edges to build
                # the 20 nodes of the 3D hexahedral element. We simulate that here.

                # Base indices for the element's corner in the global node_grid
                i0, j0, z0 = i * 2, j * 2, l_idx * 2

                # The 4 vertices of the quad in the XY plane (in CCW order)
                # This corresponds to the b_polygon_get_nth_vert loop in the C code.
                quad_verts = [(i0, j0), (i0+2, j0), (i0+2, j0+2), (i0, j0+2)]

                # The C code iterates through polygon vertices *backwards* to get the
                # correct node ordering for the element faces. We replicate that here.
                
                # Top face corners (Nodes 0-3)
                for idx, (vx, vy) in enumerate(reversed(quad_verts)):
                    element._nodes[idx] = node_grid[vx, vy, z0+2]

                # Bottom face corners (Nodes 4-7)
                for idx, (vx, vy) in enumerate(reversed(quad_verts)):
                    element._nodes[4+idx] = node_grid[vx, vy, z0]

                # Mid-edge nodes on Top Face (Nodes 8-11)
                element._nodes[8]  = node_grid[i0+1, j0+2, z0+2] # between 0 and 1
                element._nodes[9]  = node_grid[i0,   j0+1, z0+2] # between 1 and 2
                element._nodes[10] = node_grid[i0+1, j0,   z0+2] # between 2 and 3
                element._nodes[11] = node_grid[i0+2, j0+1, z0+2] # between 3 and 0

                # Mid-edge nodes on Bottom Face (Nodes 12-15)
                element._nodes[12] = node_grid[i0+1, j0+2, z0]   # between 4 and 5
                element._nodes[13] = node_grid[i0,   j0+1, z0]   # between 5 and 6
                element._nodes[14] = node_grid[i0+1, j0,   z0]   # between 6 and 7
                element._nodes[15] = node_grid[i0+2, j0+1, z0]   # between 7 and 4

                # Mid-edge nodes connecting Top and Bottom Faces (Nodes 16-19)
                # These correspond to the 'center_nodes' in the C code.
                for idx, (vx, vy) in enumerate(reversed(quad_verts)):
                    element._nodes[16+idx] = node_grid[vx, vy, z0+1]

    structure._dirty = True

def subdivide_element(element_to_subdivide):
    """
    Subdivides a 20-node element into eight smaller, compatible 20-node elements.

    This function creates a new structure containing the eight sub-elements.
    New nodes are created at the center of the original element, the center of each
    face, and the center of each edge. The positions of these new nodes are
    interpolated from the original element's nodes.

    Args:
        element_to_subdivide (festiv.element.element20): The element to subdivide.

    Returns:
        festiv.structure.structure: A new structure containing the 8 sub-elements.
    """
    new_structure = festiv.structure.structure()
    iso20_interp = festiv.isomap.iso20()

    # A 3x3x3 grid of node locations in isoparametric space (-1 to 1).
    # This gives 27 unique node positions for the subdivided mesh.
    node_grid = numpy.full((3, 3, 3), None, dtype=object)

    # Create all 27 nodes for the new mesh.
    # Their positions are interpolated from the parent element's shape functions.
    for i_idx, r in enumerate([-1, 0, 1]):
        for j_idx, s in enumerate([-1, 0, 1]):
            for k_idx, t in enumerate([-1, 0, 1]):
                new_node = festiv.node.node()
                new_pos = numpy.zeros(3)

                # Interpolate position using parent element's shape functions
                for parent_node_idx, parent_node in enumerate(element_to_subdivide._nodes):
                    if parent_node:
                        h = iso20_interp.h(r, s, t, parent_node_idx)
                        new_pos += parent_node._p * h

                new_node._p = new_pos
                new_structure._nodes.append(new_node)
                # Map from isoparametric coords {-1,0,1} to grid indices {0,1,2}
                node_grid[i_idx, j_idx, k_idx] = new_node

    # Create the 8 new hexahedral elements, one for each octant of the parent
    for i_el in range(2):  # Octant index along r-axis
        for j_el in range(2):  # Octant index along s-axis
            for k_el in range(2):  # Octant index along t-axis
                new_element = festiv.element.element20()
                new_element._youngs_modulus = element_to_subdivide._youngs_modulus
                new_element._poissons_ratio = element_to_subdivide._poissons_ratio
                new_structure._elements.append(new_element)

                # Base indices for this sub-element in the 3x3x3 node grid
                # These are the starting indices (0 or 1) for the octant.
                # We use these to select the correct 2x2x2 sub-cube of nodes from the 3x3x3 grid.
                i0, j0, k0 = i_el, j_el, k_el

                # Map the 20 nodes for this sub-element from the node_grid.
                # The node ordering must match the standard 'festiv.isomap.iso20' definition.

                # Corner nodes (0-7)
                new_element._nodes[0] = node_grid[i0 + 1, j0 + 1, k0 + 1]
                new_element._nodes[1] = node_grid[i0,     j0 + 1, k0 + 1]
                new_element._nodes[2] = node_grid[i0,     j0,     k0 + 1]
                new_element._nodes[3] = node_grid[i0 + 1, j0,     k0 + 1]
                new_element._nodes[4] = node_grid[i0 + 1, j0 + 1, k0]
                new_element._nodes[5] = node_grid[i0,     j0 + 1, k0]
                new_element._nodes[6] = node_grid[i0,     j0,     k0]
                new_element._nodes[7] = node_grid[i0 + 1, j0,     k0]

                # Mid-edge nodes (8-19) - This is the corrected mapping.
                # Top face mid-edges (nodes 8-11) - z = k0+1
                new_element._nodes[8]  = node_grid[i0,     j0 + 1, k0 + 1] # Mid-point of local edge 1-0
                new_element._nodes[9]  = node_grid[i0,     j0,     k0 + 1] # Mid-point of local edge 2-1
                new_element._nodes[10] = node_grid[i0 + 1, j0,     k0 + 1] # Mid-point of local edge 3-2
                new_element._nodes[11] = node_grid[i0 + 1, j0 + 1, k0 + 1] # Mid-point of local edge 0-3

                # Bottom face mid-edges (nodes 12-15) - z = k0
                new_element._nodes[12] = node_grid[i0,     j0 + 1, k0] # Mid-point of local edge 5-4
                new_element._nodes[13] = node_grid[i0,     j0,     k0] # Mid-point of local edge 6-5
                new_element._nodes[14] = node_grid[i0 + 1, j0,     k0] # Mid-point of local edge 7-6
                new_element._nodes[15] = node_grid[i0 + 1, j0 + 1, k0] # Mid-point of local edge 4-7

                # Vertical mid-edges (nodes 16-19) - z = k0
                new_element._nodes[16] = node_grid[i0 + 1, j0 + 1, k0] # Mid-point of local edge 0-4
                new_element._nodes[17] = node_grid[i0,     j0 + 1, k0] # Mid-point of local edge 1-5
                new_element._nodes[18] = node_grid[i0,     j0,     k0] # Mid-point of local edge 2-6
                new_element._nodes[19] = node_grid[i0 + 1, j0,     k0] # Mid-point of local edge 3-7

    new_structure._dirty = True
    return new_structure
