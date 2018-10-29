""" periodic_porous script. """
import numpy as np
from generate_mesh import MESHES_DIR, store_mesh_HDF5, line_points, \
    rad_points, round_trip_connect, numpy_to_dolfin
from utilities.plot import plot_edges, plot_faces
from meshpy import triangle as tri
from common import info
import os
import dolfin as df
import matplotlib.pyplot as plt


def description(**kwargs):
    info("")


def method(Lx=6., Ly=4., Lx_inner=4., num_obstacles=32,
           rad=0.2, R=0.3, dx=0.05, seed=121, show=False, **kwargs):
    N = int(np.ceil(Lx/dx))

    x_min, x_max = -Lx/2, Lx/2
    y_min, y_max = -Ly/2, Ly/2

    y = np.linspace(y_min, y_max, N).flatten()

    pts = np.zeros((num_obstacles, 2))
    diam2 = 4*R**2

    np.random.seed(seed)

    for i in range(num_obstacles):
        while True:
            pt = (np.random.rand(2)-0.5) * np.array([Lx_inner, Ly])
            if i == 0:
                break
            dist = pts[:i, :] - np.outer(np.ones(i), pt)
            for j in range(len(dist)):
                if abs(dist[j, 1]) > Ly/2:
                    dist[j, 1] = abs(dist[j, 1])-Ly
            dist2 = dist[:, 0]**2 + dist[:, 1]**2
            if all(dist2 > diam2):
                break
        pts[i, :] = pt

    pts = pts[pts[:, 0].argsort(), :]

    obstacles = [tuple(row) for row in pts]

    line_segments_top = []
    line_segments_btm = []

    x_prev = x_min

    curve_segments_top = []
    curve_segments_btm = []

    interior_obstacles = []
    exterior_obstacles = []

    for x_c in obstacles:
        # Close to the top of the domain
        if x_c[1] > y_max-rad:
            # identify intersection
            theta = np.arcsin((y_max-x_c[1])/rad)
            rx = rad*np.cos(theta)
            x_left = x_c[0]-rx
            x_right = x_c[0]+rx

            line_segments_top.append(line_points((x_prev, y_max),
                                                 (x_left, y_max), dx))
            line_segments_btm.append(line_points((x_prev, y_min),
                                                 (x_left, y_min), dx))
            curve_btm = rad_points((x_c[0], x_c[1]-Ly), rad, dx,
                                   theta_start=np.pi-theta,
                                   theta_stop=theta)[1:-1]
            curve_top = rad_points(x_c, rad, dx,
                                   theta_start=np.pi-theta,
                                   theta_stop=2*np.pi+theta)[1:-1]
            curve_segments_btm.append(curve_btm)
            curve_segments_top.append(curve_top)

            x_prev = x_right

            exterior_obstacles.append(x_c)
            exterior_obstacles.append((x_c[0], x_c[1]-Ly))
        # Close to the bottom of the domain
        elif x_c[1] < y_min+rad:
            # identify intersection
            theta = np.arcsin((-y_min+x_c[1])/rad)
            rx = rad*np.cos(theta)
            x_left = x_c[0]-rx
            x_right = x_c[0]+rx

            line_segments_top.append(line_points((x_prev, y_max),
                                                 (x_left, y_max), dx))
            line_segments_btm.append(line_points((x_prev, y_min),
                                                 (x_left, y_min), dx))
            curve_btm = rad_points(x_c, rad, dx,
                                   theta_start=np.pi+theta,
                                   theta_stop=-theta)[1:-1]
            curve_top = rad_points((x_c[0], x_c[1]+Ly), rad, dx,
                                   theta_start=np.pi+theta,
                                   theta_stop=2*np.pi-theta)[1:-1]
            curve_segments_btm.append(curve_btm)
            curve_segments_top.append(curve_top)

            x_prev = x_right

            exterior_obstacles.append(x_c)
            exterior_obstacles.append((x_c[0], x_c[1]+Ly))
        else:
            interior_obstacles.append(x_c)

    line_segments_top.append(line_points((x_prev, y_max),
                                         (x_max, y_max), dx))
    line_segments_btm.append(line_points((x_prev, y_min),
                                         (x_max, y_min), dx))

    assert(len(line_segments_top) == len(curve_segments_top)+1)
    assert(len(line_segments_btm) == len(curve_segments_btm)+1)

    pts_top = list(line_segments_top[0])
    for i in range(len(curve_segments_top)):
        pts_top.extend(curve_segments_top[i])
        pts_top.extend(line_segments_top[i+1])
    pts_top = pts_top[::-1]

    pts_btm = list(line_segments_btm[0])
    for i in range(len(curve_segments_btm)):
        pts_btm.extend(curve_segments_btm[i])
        pts_btm.extend(line_segments_btm[i+1])

    y_side = y[1:-1]
    pts_right = list(zip(x_max*np.ones(N-2), y_side))
    pts_left = list(zip(x_min*np.ones(N-2), y_side[::-1]))

    pts = pts_btm + pts_right + pts_top + pts_left
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

    # pp = [tuple(point) for point in mesh.points]
    # print "Number of points:", len(pp)
    # print "Number unique points:", len(set(pp))

    if show:
        plot_faces(coords, faces)

    msh = numpy_to_dolfin(coords, faces)

    mesh_path = os.path.join(
        MESHES_DIR,
        "periodic_porous_Lx{}_Ly{}_rad{}_N{}_dx{}".format(
            Lx, Ly, rad, num_obstacles, dx))
    store_mesh_HDF5(msh, mesh_path)

    obstacles_path = os.path.join(
        MESHES_DIR,
        "periodic_porous_Lx{}_Ly{}_rad{}_N{}_dx{}.dat".format(
            Lx, Ly, rad, num_obstacles, dx))

    if len(exterior_obstacles) > 0 and len(interior_obstacles) > 0:
        all_obstacles = np.vstack((np.array(exterior_obstacles),
                                   np.array(interior_obstacles)))
    elif len(exterior_obstacles) > 0:
        all_obstacles = np.array(exterior_obstacles)
    else:
        all_obstacles = np.array(interior_obstacles)
    np.savetxt(obstacles_path,
               np.hstack((all_obstacles,
                          np.ones((len(all_obstacles), 1))*rad)))

    if show:
        df.plot(msh)
        plt.show()
