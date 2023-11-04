import meshio

mesh_3d = meshio.read('z31-cube-mesh/cube_4ele_w_bc.msh')
mesh_2d = meshio.read('z32-square-mesh/square_w_tag.msh')
print(mesh_2d)