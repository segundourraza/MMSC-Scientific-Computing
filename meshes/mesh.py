import pygmsh, meshio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# polygon vertices in CCW order (z coordinate zero for 2D)
poly_pts = [
    [0.0, 0.0, 0.0],
    [1.5, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.2, 0.0],
]

poly_pts = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
]


with pygmsh.geo.Geometry() as geom:
    # create polygon; mesh_size is characteristic element size
    poly = geom.add_polygon(poly_pts, mesh_size=0.08)
    mesh = geom.generate_mesh()

# mesh is a meshio.Mesh object
# get node coordinates
points = mesh.points  # (N,3) array; z=0 for 2D
# get triangle connectivity robustly:
tri_cells = None
for cell_block in mesh.cells:
    if cell_block.type == "triangle":
        tri_cells = cell_block.data
        break

print("nodes:", points.shape)
print("triangles:", tri_cells.shape)

triang = tri.Triangulation(points[:, 0],
                           points[:, 1],
                           tri_cells)

plt.figure()
plt.plot(points[:,0], points[:,1], '.k', ms = 6)
plt.triplot(triang, linewidth=0.6)
plt.gca().set_aspect("equal")
plt.title("FEM Mesh")
plt.show()

# meshio.write("meshes/polygon_mesh.vtk", mesh)
