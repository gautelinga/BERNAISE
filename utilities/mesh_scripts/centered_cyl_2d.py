""" periodic_porous script. """
import numpy as np
from generate_mesh import MESHES_DIR, store_mesh_HDF5, line_points, \
    rad_points, round_trip_connect, numpy_to_dolfin, numpy_to_dolfin_old
from utilities.plot import plot_edges, plot_faces, plt
from meshpy import triangle as tri
from common import info
import os
import dolfin as df
from mesh_scripts.periodic_porous_2d import discretize_loop
import math


def description(**kwargs):
    info("")


def method(Lx=4., Ly=4.,
           rad=0.25, dx=0.05, dx_outer=0.5,
           show=False, **kwargs):
    x_min, x_max = -Lx/2, Lx/2
    y_min, y_max = -Ly/2, Ly/2

    obstacle = (0., 0.)

    segments = [((x_min, y_min), (x_max, y_min)),
                ((x_max, y_min), (x_max, y_max)),
                ((x_max, y_max), (x_min, y_max)),
                ((x_min, y_max), (x_min, y_min))]
    segments = dict(segments)
    pt_start = (x_min, y_min)

    if show:
        for x_a, x_b in segments.items():
            x = np.array([x_a[0], x_b[0]])
            y = np.array([x_a[1], x_b[1]])
            plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1],
                       scale_units='xy', angles='xy', scale=1)        
        plt.show()

    pts = discretize_loop(pt_start, [], [], segments, dx_outer)[::-1]
    edges = round_trip_connect(0, len(pts)-1)

    pts_obstacle = rad_points(obstacle, rad, dx)[1:]
    edges_obstacle = round_trip_connect(len(pts), len(pts)+len(pts_obstacle)-1)

    pts.extend(pts_obstacle)
    edges.extend(edges_obstacle)

    if show:
        plot_edges(pts, edges)

    mi = tri.MeshInfo()
    mi.set_points(pts)
    mi.set_facets(edges)
    mi.set_holes([obstacle])

    max_area = 1000 * 0.5*dx**2

    def needs_refinement(vertices, area):
        vert_origin, vert_destination, vert_apex = vertices
        bary_x = (vert_origin.x + vert_destination.x + vert_apex.x) / 3
        bary_y = (vert_origin.y + vert_destination.y + vert_apex.y) / 3

        r = math.sqrt(bary_x ** 2 + bary_y ** 2)
        R = min(Lx, Ly)/2 #np.sqrt((Lx/2)**2+(Ly/2)**2)
        A_outer = 0.5*dx_outer**2
        A = 0.5*dx**2
        max_area = A + (A_outer - A)* (math.fabs((r - rad)/(R-rad)))**2
        return area > max_area

    mesh = tri.build(mi, min_angle=25,
                     refinement_func=needs_refinement,
                     allow_boundary_steiner=False)

    coords = np.array(mesh.points)
    faces = np.array(mesh.elements)

    pp = [tuple(point) for point in mesh.points]
    info("Number of points:     {}".format(len(pp)))
    info("Number unique points: {}".format(len(set(pp))))

    if show:
        plot_faces(coords, faces)

    msh = numpy_to_dolfin(coords, faces)

    mesh_path = os.path.join(MESHES_DIR,
                             "centered_cyl_Lx{}_Ly{}_r{}_dx{}".format(
                             Lx, Ly, rad, dx))
    store_mesh_HDF5(msh, mesh_path)

    obstacles_path = os.path.join(
        MESHES_DIR, "centered_cyl_Lx{}_Ly{}_r{}_dx{}.dat".format(Lx, Ly, rad, dx))

    all_obstacles = [obstacle]

    if len(all_obstacles):
        np.savetxt(obstacles_path,
                   np.hstack((all_obstacles,
                              np.ones((len(all_obstacles), 1))*rad)))
