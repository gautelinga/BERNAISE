""" periodic_porous script. """
import numpy as np
from generate_mesh import MESHES_DIR, store_mesh_HDF5, line_points, \
    rad_points, round_trip_connect, numpy_to_dolfin, numpy_to_dolfin_old
from utilities.plot import plot_edges, plot_faces, plt
from meshpy import triangle as tri
from common import info
import os
import dolfin as df
import math
from .periodic_porous_2d import place_obstacles, correct_obstacles, classify_obstacles, compute_intersections, draw_curves, \
    get_curve_intersection_points, construct_segments, discretize_loop, cppcode


def description(**kwargs):
    info("Periodic porous refined.")


def method(Lx=6., Ly=4., pad_x=0., pad_y=0., num_obstacles=25,
           rad=0.25, R=0.3, dx=0.02, dx_outer=0.5, scale_outer=1., seed=123,
           show=False, Nx=1000, Ny=1000, **kwargs):
    x_min, x_max = -Lx/2, Lx/2
    y_min, y_max = -Ly/2, Ly/2

    obstacles = cppcode.place_obstacles(num_obstacles, Lx, Ly, R, pad_x=pad_x, pad_y=pad_y, seed=seed)
    # np.random.seed(seed)
    # obstacles = place_obstacles(num_obstacles, Lx_inner, Ly_inner, R, Ly_outer=Ly, Lx_outer=Lx)
    if len(obstacles) < num_obstacles:
        print("Could not fit {} obstacles. Could only fit {} obstacles.".format(num_obstacles, len(obstacles)))
    num_obstacles = len(obstacles)

    obstacles = correct_obstacles(obstacles, rad, x_min, x_max, y_min, y_max)

    interior_obstacles, exterior_obstacles, obst = classify_obstacles(
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
                          curves, segments, dx_outer)[::-1]

    if show:
        plt.show()

    edges = round_trip_connect(0, len(pts)-1)

    max_area = 0.5*dx**2

    XX, YY = np.meshgrid(np.linspace(-Lx/2, Lx/2, Nx, endpoint=False), np.linspace(-Ly/2, Ly/2, Ny, endpoint=False))
    dist = 10 * np.ones_like(XX)
    for xx, yy in obstacles:
        r = np.sqrt(np.minimum(abs(XX-xx), Lx-abs(XX-xx))**2 + np.minimum(abs(YY-yy), Ly-abs(YY-yy))**2)
        dist[r < dist] = r[r < dist]

    if show:
        plt.pcolormesh(XX, YY, dist)
        plt.show()

    #L = min(Lx, Ly)/2 #np.sqrt((Lx/2)**2+(Ly/2)**2)
    A_outer = 0.5*dx_outer**2
    A = 0.5*dx**2
    scalef = (dist - rad)/scale_outer
    scalef[scalef > 1.] = 1.
    scalef = -2*scalef**3 + 3*scalef**2
    if show:
        plt.pcolormesh(XX, YY, scalef)
        plt.show()

    max_area = A + (A_outer - A) * scalef
    ddx = Lx / Nx
    ddy = Ly / Ny

    count = 1
    while count:
        edges_out = list(edges)
        pts_out = list(pts)
        count = 0
        for iedge, (i, j) in enumerate(edges):
            xi = np.array(pts[i])
            xj = np.array(pts[j])
            xmid = 0.5*(xi + xj)
            m = round((np.remainder(xmid[0] + 0*Lx/2, Lx) - Lx/2) / ddx)
            n = round((np.remainder(xmid[1] + 0*Ly/2, Ly) - Ly/2) / ddy)
            lij = np.linalg.norm(xi - xj)
            if lij**2 > 2*max_area[n, m]:
                pts_out.append(list(xmid))
                edges_out[iedge] = (i, len(pts_out)-1)
                edges_out.append((len(pts_out)-1, j))
                count += 1
        edges = list(edges_out)
        pts = list(pts_out)

    if show:
        plot_edges(pts, edges)
        plt.show()

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

    def needs_refinement(vertices, area):
        vert_origin, vert_destination, vert_apex = vertices
        x = [(vert_origin.x + vert_destination.x + vert_apex.x) / 3,
             (vert_origin.y + vert_destination.y + vert_apex.y) / 3]

        i = round((x[0] - Lx/2) / ddx)
        j = round((x[1] - Ly/2) / ddy)
        max_area_ij = max_area[j, i]
        return area > max_area_ij

    mesh = tri.build(mi, # max_volume=max_area, 
                     min_angle=25,
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
                             "periodic_porous_Lx{}_Ly{}_r{}_R{}_N{}_dx{}".format(
                                 Lx, Ly, rad, R, num_obstacles, dx))
    store_mesh_HDF5(msh, mesh_path)

    porosity = 1 - len(obstacles) * np.pi * rad**2 / (Lx * Ly)

    print("porosity={}".format(porosity))

    obstacles_path = os.path.join(
        MESHES_DIR,
        "periodic_porous_Lx{}_Ly{}_r{}_R{}_N{}_dx{}.dat".format(
            Lx, Ly, rad, R, num_obstacles, dx))

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
