__author__ = "Asger J. S Bolet <asgerbolet@gmail.com>"
__date__ = "2017-04-28"
__copyright__ = "Copyright (C) 2017 " + __author__
__license__ = "MIT"

''' "StoreMeshHDF5(mesh, meshpath)",
"StraightCapilar(res, height, length, usemshr)" and 
"BarbellCapilar(res, diameter, length)" 
'''
import dolfin as df
import mshr as mshr
from mpi4py import MPI
import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import meshpy.triangle as tri
from matplotlib.collections import LineCollection

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MESHES_DIR = "../meshes/"


def store_mesh_HDF5(mesh, meshpath):
    '''
    Function that stores generated mesh in both "HDMF5"
    (.h5) format and in "XDMF" (.XMDF) format.
    '''
    meshpathhdf5 = meshpath + ".h5"
    hdf5 = df.HDF5File(mesh.mpi_comm(), meshpathhdf5, "w")
    if rank == 0:
        print "Storing the mesh in " + MESHES_DIR
    hdf5.write(mesh, "mesh")
    hdf5.close()
    meshpathxdmf = meshpath + "_xdmf.xdmf"
    xdmff1 = df.XDMFFile(mesh.mpi_comm(), meshpathxdmf)
    xdmff1.write(mesh)
    if rank == 0:
        print 'Done.'


def straight_capilar(res=10, height=1, length=5, use_mshr=False):
    '''
    Function That Generates a mesh for a straight capilar,
    defualt meshing method is dolfin's "RectangleMesh" but have option for mshr.
    Note: Should be run form "BERNAISE/utilies/" in order to work.
    Note: The generarted mesh is storred in "BERNAISE/meshes/".
    '''
    if use_mshr:  # use mshr for the generation
        if rank == 0:
            print "Generating mesh using the mshr tool"
        # Define coners of Rectangle
        a = df.Point(0, 0)
        b = df.Point(height, length)
        domain = mshr.Rectangle(a, b)
        mesh = mshr.generate_mesh(domain, res)
        meshpath = os.path.join(MESHES_DIR,
                                "StraightCapilarMshr_h" + str(height) + "_l" +
                                str(length) + "_res" + str(res))
        if rank == 0:
            print "Done."
    else:  # use the Dolfin built-in function
        if rank == 0:
            print "Genrating mesh using the Dolfin built-in function."
        # Define coners of rectangle/capilar
        a = df.Point(0, 0)
        b = df.Point(height, length)
        # Setting the reselution
        if height <= length:
            num_points_height = res
            num_points_length = res*int(length/height)
        else:
            num_points_height = res*int(height/length)
            num_points_length = res
        mesh = df.RectangleMesh(a, b, num_points_height, num_points_length)
        meshpath = os.path.join(MESHES_DIR,
                                "StraightCapilarDolfin_h" +
                                str(height) + "_l" +
                                str(length) + "_res" + str(res))
    store_mesh_HDF5(mesh, meshpath)


def barbell_capilar(res=50, diameter=1., length=5.):
    '''
    Function That Generates a mesh for a barbell capilar,
    Meshing method is mshr.
    Note: Should be run form "BERNAISE/utilies/" in order to work.
    Note: The generarted mesh is storred in "BERNAISE/meshes/".
    '''
    if rank == 0:
        print "Generating mesh using the mshr tool."

    inletdiameter = diameter*5.
    inletlength = diameter*4.

    # Define coners of "capilar"
    a = df.Point(-diameter/2., -length/2-inletlength/2.)
    b = df.Point(diameter/2., length/2+inletlength/2.)
    capilar = mshr.Rectangle(a, b)
    # Define coners of "leftbell
    c = df.Point(-inletdiameter/2., -length/2-inletlength)
    d = df.Point(inletdiameter/2., -length/2)
    leftbell = mshr.Rectangle(c, d)
    # Define coners of "rightbell"
    e = df.Point(-inletdiameter/2., length/2)
    f = df.Point(inletdiameter/2., length/2+inletlength)
    rightbell = mshr.Rectangle(e, f)

    domain = capilar + leftbell + rightbell
    mesh = mshr.generate_mesh(domain, res)
    meshpath = os.path.join(MESHES_DIR,
                            "BarbellCapilarDolfin_d" + str(diameter) + "_l" +
                            str(length) + "_res" + str(res))
    store_mesh_HDF5(mesh, meshpath)

def AddVerticalBoundaryVertices(l,x,y,n,dr):
    """
    A function to make a 'polygon'-srightline in order to make a periodig domain.
    l is eges in the polygon, x is the x coordinat for the line, y is the lengt of the line,
    n is the number for verticies and dr 
    """
    for i in range(1,n):
        if dr == -1: # Values go from (near) y to 0 with n points
            val = y * float(n - i)/float(n)
        elif dr == 1: # values go from (near) 0 to y with n points
            val = y * float(i)/float(n)
        l.append( df.Point(x, val) )

def roundet_barbell_capilar(L=6., H=2., R=0.3, n_segments=40, res=180): 
    """
    Generates barbell capilar with roundet eges.
    """
    if rank == 0:
        print "Generating mesh of roudet barbell capilar"
    
    pt_1 = df.Point(0., 0.)
    pt_1star = df.Point(1., 0.)
    pt_1starstar = df.Point(L/(2*res), 0.)
    pt_2 = df.Point(L, H)
    pt_2star = df.Point(L-1., H)
    pt_2starstar = df.Point(L-L/(2*res), H)
    pt_3 = df.Point(1.,H)
    pt_3star = df.Point(0, H)
    pt_3starstar = df.Point(L/(2*res), H)
    pt_4 = df.Point(L-1.,0)
    pt_4star = df.Point(L,0)
    pt_4starstar = df.Point(L-L/(2*res), 0)
    pt_5 = df.Point(1.,R)
    pt_6 = df.Point(1.,H-R)
    pt_7 = df.Point(L-1.,R)
    pt_8 = df.Point(L-1.,H-R)
    pt_9 = df.Point(1.+2*R,R)
    pt_10 = df.Point(1.+2*R,H-R)
    pt_11 = df.Point(L-2*R-1,R)
    pt_12 = df.Point(L-2*R-1,H-R)
    pt_13 = df.Point(1.+2*R,H-2*R)
    pt_14 = df.Point(L-2*R-1,2*R)
    
    inlet_polygon = [pt_1]
    inlet_polygon.append(pt_1starstar)
    AddVerticalBoundaryVertices(inlet_polygon,L/res,H,res,1)
    inlet_polygon.append(pt_3starstar)
    inlet_polygon.append(pt_3star)
    AddVerticalBoundaryVertices(inlet_polygon,0.0,H,res,-1)
    inlet_polygon.append(pt_1)

    outlet_polygon = [pt_4starstar]
    outlet_polygon.append(pt_4star)
    AddVerticalBoundaryVertices(outlet_polygon,L,H,res,1)
    outlet_polygon.append(pt_2)
    outlet_polygon.append(pt_2starstar)
    AddVerticalBoundaryVertices(outlet_polygon,L-L/res,H,res,-1)
    outlet_polygon.append(pt_4starstar)

    inlet1 = mshr.Polygon(inlet_polygon)
    inlet2 = mshr.Rectangle(pt_1starstar,pt_3)
    outlet1 = mshr.Polygon(outlet_polygon)
    outlet2 = mshr.Rectangle(pt_4,pt_2starstar)
    channel = mshr.Rectangle(pt_5,pt_8)
    pos_cir_1 = mshr.Circle(pt_5,R,segments=n_segments)
    pos_cir_2 = mshr.Circle(pt_6,R,segments=n_segments)
    pos_cir_3 = mshr.Circle(pt_7,R,segments=n_segments)
    pos_cir_4 = mshr.Circle(pt_8,R,segments=n_segments)
    neg_cir_1 = mshr.Circle(pt_9,R,segments=n_segments)
    neg_cir_2 = mshr.Circle(pt_10,R,segments=n_segments)
    neg_cir_3 = mshr.Circle(pt_11,R,segments=n_segments)
    neg_cir_4 = mshr.Circle(pt_12,R,segments=n_segments)
    neg_reg_1 = mshr.Rectangle(pt_13,pt_12)
    neg_reg_2 = mshr.Rectangle(pt_9,pt_14)

    domain = inlet1 + inlet2 + outlet1 + outlet2 + channel + pos_cir_1 + pos_cir_2 + pos_cir_3 + pos_cir_4 - neg_cir_1 - neg_cir_2 - neg_cir_3 - neg_cir_4 - neg_reg_1 - neg_reg_2 

    mesh = mshr.generate_mesh(domain, res)
    
    mesh_path = os.path.join(MESHES_DIR,
                             "roundet_barbell_res" + str(res))
    store_mesh_HDF5(mesh, mesh_path)
    df.plot(mesh)
    df.interactive()

def snausen_mesh(L=3., H=1., R=0.3, n_segments=40, res=60):
    """
    Generates mesh of Snausen/Snoevsen.
    """
    if rank == 0:
        print "Generating mesh of Snoevsen."

    # Define points:
    pt_A = df.Point(0., 0.)
    pt_B = df.Point(L, H)
    pt_C = df.Point(L/2-2*R, 0.)
    pt_D = df.Point(L/2+2*R, 0.)
    pt_E = df.Point(L/2-2*R, -R)
    pt_F = df.Point(L/2+2*R, -R)
    pt_G = df.Point(L/2, -R)

    tube = mshr.Rectangle(pt_A, pt_B)
    add_rect = mshr.Rectangle(pt_E, pt_D)
    neg_circ_L = mshr.Circle(pt_E, R, segments=n_segments)
    neg_circ_R = mshr.Circle(pt_F, R, segments=n_segments)
    pos_circ = mshr.Circle(pt_G, R, segments=n_segments)

    domain = tube + add_rect - neg_circ_L - neg_circ_R + pos_circ

    mesh = mshr.generate_mesh(domain, res)

    # check that mesh is periodic along x
    boun = df.BoundaryMesh(mesh, "exterior")
    y = boun.coordinates()[:, 1]
    y = y[y < 1.]
    y = y[y > 0.]
    n_y = len(y)
    n_y_unique = len(np.unique(y))
    print n_y
    print n_y_unique
    assert(n_y == 2*n_y_unique)

    mesh_path = os.path.join(MESHES_DIR,
                             "snoevsen_res" + str(res))
    store_mesh_HDF5(mesh, mesh_path)

    #df.plot(mesh)
    #df.interactive()


def porous_mesh(Lx=4., Ly=4., rad=0.2, R=0.3, N=24, n_segments=40, res=80):

    if rank == 0:
        print "Generating porous mesh"

    # x = np.random.rand(N, 2)

    diam2 = 4*R**2

    pts = np.zeros((N, 2))
    for i in range(N):
        while True:
            pt = (np.random.rand(2)-0.5) * np.array([Lx-2*R, Ly-2*R])
            if i == 0:
                break
            dist = pts[:i, :] - np.outer(np.ones(i), pt)
            dist2 = dist[:, 0]**2 + dist[:, 1]**2
            if all(dist2 > diam2):
                break
        pts[i, :] = pt

    rect = mshr.Rectangle(df.Point(-Lx/2, -Ly/2), df.Point(Lx/2, Ly/2))
    domain = rect
    for i in range(N):
        domain -= mshr.Circle(df.Point(pts[i, 0], pts[i, 1]),
                              rad, segments=n_segments)

    mesh = mshr.generate_mesh(domain, res)

    df.plot(mesh)
    df.interactive()


def round_trip_connect(start, end):
    return [(i, i+1) for i in range(start, end)] + [(end, start)]


def rad_points(x_c, rad, dx, theta_start=0., theta_stop=2*np.pi):
    if theta_stop > theta_start:
        arc_length = (theta_stop-theta_start)*rad
    else:
        arc_length = (-theta_stop+theta_start)*rad
    return [(rad * np.cos(theta) + x_c[0], rad * np.sin(theta) + x_c[1])
            for theta in np.linspace(theta_start, theta_stop,
                                     int(np.ceil(arc_length/dx)))]


def line_points(x_left, x_right, dx):
    N = int(np.ceil(np.sqrt((x_right[0]-x_left[0])**2 +
                            (x_right[1]-x_left[1])**2)/dx))
    return zip(np.linspace(x_left[0], x_right[0], N).flatten(),
               np.linspace(x_left[1], x_right[1], N).flatten())


def periodic_porous_mesh(Lx=4., Ly=4., num_obstacles=12,
                         rad=0.3, R=0.45, dx=0.1):
    N = int(np.ceil(Lx/dx))

    x_min, x_max = -Lx/2, Lx/2
    y_min, y_max = -Ly/2, Ly/2

    y = np.linspace(y_min, y_max, N).flatten()

    pts = np.zeros((num_obstacles, 2))
    diam2 = 4*R**2
    for i in range(num_obstacles):
        while True:
            pt = (np.random.rand(2)-0.5) * np.array([Lx-2*R, Ly])
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

    obstacles = [(-Lx/4., y_max-rad/2),
                 (Lx/4., y_min+rad/2),
                 (0., 0.),
                 (Lx/4, 0.),
                 (-Lx/4, 0.)]

    obstacles = [tuple(row) for row in pts]

    print obstacles

    line_segments_top = []
    line_segments_btm = []

    x_prev = x_min

    curve_segments_top = []
    curve_segments_btm = []

    interior_obstacles = []

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
        else:
            interior_obstacles.append(x_c)

    line_segments_top.append(line_points((x_prev, y_max),
                                         (x_max, y_max), dx))
    line_segments_btm.append(line_points((x_prev, y_min),
                                         (x_max, y_min), dx))

    assert(len(line_segments_top) == len(curve_segments_top)+1)
    assert(len(line_segments_btm) == len(curve_segments_btm)+1)

    pts_top = line_segments_top[0]
    for i in range(len(curve_segments_top)):
        pts_top.extend(curve_segments_top[i])
        pts_top.extend(line_segments_top[i+1])
    pts_top = pts_top[::-1]

    pts_btm = line_segments_btm[0]
    for i in range(len(curve_segments_btm)):
        pts_btm.extend(curve_segments_btm[i])
        pts_btm.extend(line_segments_btm[i+1])

    y_side = y[1:-1]
    pts_right = zip(x_max*np.ones(N-2), y_side)
    pts_left = zip(x_min*np.ones(N-2), y_side[::-1])

    pts = pts_btm + pts_right + pts_top + pts_left
    edges = round_trip_connect(0, len(pts)-1)

    for interior_obstacle in interior_obstacles:
        pts_obstacle = rad_points(interior_obstacle, rad, dx)
        edges_obstacle = round_trip_connect(len(pts),
                                            len(pts)+len(pts_obstacle)-1)

        pts.extend(pts_obstacle)
        edges.extend(edges_obstacle)

    nppts = np.array(pts)
    npedges = np.array(edges)
    lc = LineCollection(nppts[npedges])
    fig = plt.figure()
    plt.gca().add_collection(lc)
    plt.xlim(nppts[:, 0].min(), nppts[:, 0].max())
    plt.ylim(nppts[:, 1].min(), nppts[:, 1].max())
    plt.plot(nppts[:, 0], nppts[:, 1], 'ro')
    plt.show()

    mi = tri.MeshInfo()
    mi.set_points(pts)
    mi.set_facets(edges)
    mi.set_holes(interior_obstacles)

    max_area = 0.5*dx**2

    mesh = tri.build(mi, max_volume=max_area, min_angle=25,
                     allow_boundary_steiner=False)

    coords = np.array(mesh.points)
    faces = np.array(mesh.elements)

    #fig = plt.figure()
    #plt.triplot(coords[:, 0], coords[:, 1], faces, 'go-')
    #plt.show()

    #msh = df.UnitSquareMesh(10, 10)
    #msh.coordinates().array() = coords
    #msh.cells().array() = faces

    with open("tmp.xml", "w") as f:
        f.write("""
<?xml version="1.0" encoding="UTF-8"?>
<dolfin xmlns:dolfin="http://www.fenics.org/dolfin/">
    <mesh celltype="triangle" dim="2">
        <vertices size="%d">""" % len(mesh.points))

        for i, pt in enumerate(mesh.points):
            f.write('<vertex index="%d" x="%g" y="%g"/>' % (
                i, pt[0], pt[1]))

        f.write("""
        </vertices>
        <cells size="%d">
        """ % len(mesh.elements))

        for i, element in enumerate(mesh.elements):
            f.write('<triangle index="%d" v0="%d" v1="%d" v2="%d"/>' % (
                i, element[0], element[1], element[2]))

        f.write("""
            </cells>
          </mesh>
        </dolfin>
        """)

    msh = df.Mesh("tmp.xml")

    df.plot(msh)
    df.interactive()


def main():
    #straight_capilar()
    #barbell_capilar()
    #snausen_mesh()
    #porous_mesh()
    #periodic_porous_mesh()
    roundet_barbell_capilar()

if __name__ == "__main__":
    main()
 
