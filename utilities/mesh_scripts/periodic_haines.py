""" periodic_porous script. """
import numpy as np
from generate_mesh import MESHES_DIR, store_mesh_HDF5, line_points, \
    rad_points, round_trip_connect, numpy_to_dolfin, numpy_to_dolfin_old
from .periodic_porous_2d import classify_obstacles, compute_intersections, \
    draw_curves, get_curve_intersection_points, construct_segments, discretize_loop
from utilities.plot import plot_edges, plot_faces, plt
from meshpy import triangle as tri
from common import info
import os
import dolfin as df


def description(**kwargs):
    info("")


def place_obstacles(Lx, Ly, Nx, Ny, perturb):
    pts = []
    # Nx = int(Lx/(2*R))
    R = Lx / (2*Nx)
    for iy in range(Ny):
        x0 = R * (iy % 2) - Lx/2
        y0 = 0
        for ix in range(Nx):
            # eps = perturb if iy==1 and ix >
            eps = np.random.normal(0, 1, 2)
            eps *= perturb/np.linalg.norm(eps)
            pts.append([x0 + ix * 2*R + eps[0], y0 + (iy - (Ny-1)/2) * R * np.sqrt(3) + eps[1]])
    pts = np.array(pts)

    pts = pts[pts[:, 0].argsort(), :]
    obstacles = [tuple(row) for row in pts]
    return obstacles, R


def method(Lx=10., Ly=20., Nx=7, Ny=15, rad=0.5, dx=0.05, perturb=0.0, seed=123, show=False, **kwargs):
    x_min, x_max = -Lx/2, Lx/2
    y_min, y_max = -Ly/2, Ly/2

    np.random.seed(seed)
    obstacles, R = place_obstacles(Lx, Ly, Nx, Ny, perturb)
    num_obstacles = len(obstacles)

    xx = np.array(obstacles)
    plt.scatter(xx[:, 0], xx[:, 1])
    plt.show()

    interior_obstacles, _, obst = classify_obstacles(
        obstacles, rad, x_min, x_max, y_min, y_max)

    theta_low, theta_high = compute_intersections(
        obst, rad, x_min, x_max, y_min, y_max)


    curves = draw_curves(obst, theta_low, theta_high, rad, dx)
    if len(curves) > 0:
        curve_start, curve_stop = get_curve_intersection_points(
            curves, x_min, x_max, y_min, y_max)

        segments = construct_segments(curve_start, curve_stop,
                                      x_min, x_max, y_min, y_max)
        pt_start = curves[0][0]
    else:
        curve_start = []
        curve_stop = []
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
        for curve in curves:
            x = np.array(curve)
            plt.quiver(x[:-1, 0], x[:-1, 1],
                       x[1:, 0]-x[:-1, 0],
                       x[1:, 1]-x[:-1, 1],
                       scale_units='xy', angles='xy', scale=1)

    pts = discretize_loop(pt_start, curve_start,
                          curves, segments, dx)[::-1]

    plt.show()

    edges = round_trip_connect(0, len(pts)-1)

    for interior_obstacle in interior_obstacles:
        pts_obstacle = rad_points(interior_obstacle, rad, dx)[1:]
        edges_obstacle = round_trip_connect(len(pts),
                                            len(pts)+len(pts_obstacle)-1)

        pts.extend(pts_obstacle)
        edges.extend(edges_obstacle)

    if show:
        plot_edges(pts, edges)

    mi = tri.MeshInfo()
    mi.set_points(pts)
    mi.set_facets(edges)
    mi.set_holes(interior_obstacles)

    max_area = 0.5*dx**2

    mesh = tri.build(mi, max_volume=max_area, min_angle=25,
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
                             "cylarr_Lx{}_Ly{}_r{}_dx{}".format(
                                 Lx, Ly, rad, dx))
    store_mesh_HDF5(msh, mesh_path)

    obstacles_path = os.path.join(
        MESHES_DIR,
        "periodic_porous_Lx{}_Ly{}_r{}_dx{}.dat".format(
            Lx, Ly, rad, dx))

    if len(obst) and len(interior_obstacles):
        all_obstacles = np.vstack((np.array(obst),
                                   np.array(interior_obstacles)))
    elif len(interior_obstacles):
        all_obstacles = interior_obstacles
    else:
        all_obstacles = []

    if len(all_obstacles):
        np.savetxt(obstacles_path,
                   np.hstack((all_obstacles,
                              np.ones((len(all_obstacles), 1))*rad)))
