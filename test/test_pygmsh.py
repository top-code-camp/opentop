import pygmsh

with pygmsh.geo.Geometry() as geom:
    poly = geom.add_polygon(
        [
            [0.0, 0.0],
            [1.0, -0.2],
            [1.1, 1.2],
            [0.1, 0.7],
        ],
        mesh_size=0.1,
    )
    
    #geom.extrude(poly, [0.0, 0.3, 1.0], num_layers=5)
    
    mesh = geom.generate_mesh()
    
    mesh.write("test.vtk")
    
    pygmsh.write("test.msh")