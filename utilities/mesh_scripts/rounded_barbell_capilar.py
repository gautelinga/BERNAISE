""" rounded_barbell_capilar script. """
import dolfin as df
from common import info
from generate_mesh import MESHES_DIR, store_mesh_HDF5, \
    add_vertical_boundary_vertices
import mshr
import os


def description(**kwargs):
    info("")


def method(L=6., H=2., R=0.3, n_segments=40, res=120, **kwargs):
    """
    Generates barbell capilar with rounded edges.
    """
    info("Generating mesh of rounded barbell capilar")
    
    pt_1 = df.Point(0., 0.)
    pt_1star = df.Point(1., 0.)
    pt_1starstar = df.Point(L/(2*res), 0.)
    pt_2 = df.Point(L, H)
    pt_2star = df.Point(L-1., H)
    pt_2starstar = df.Point(L-L/(2*res), H)
    pt_3 = df.Point(1., H)
    pt_3star = df.Point(0, H)
    pt_3starstar = df.Point(L/(2*res), H)
    pt_4 = df.Point(L-1., 0)
    pt_4star = df.Point(L, 0)
    pt_4starstar = df.Point(L-L/(2*res), 0)
    pt_5 = df.Point(1., R)
    pt_6 = df.Point(1., H-R)
    pt_7 = df.Point(L-1., R)
    pt_8 = df.Point(L-1., H-R)
    pt_9 = df.Point(1.+2*R, R)
    pt_10 = df.Point(1.+2*R, H-R)
    pt_11 = df.Point(L-2*R-1, R)
    pt_12 = df.Point(L-2*R-1, H-R)
    pt_13 = df.Point(1.+2*R, H-2*R)
    pt_14 = df.Point(L-2*R-1, 2*R)

    inlet_polygon = [pt_1]
    inlet_polygon.append(pt_1starstar)
    add_vertical_boundary_vertices(inlet_polygon, L/res, H, res, 1)
    inlet_polygon.append(pt_3starstar)
    inlet_polygon.append(pt_3star)
    add_vertical_boundary_vertices(inlet_polygon, 0.0, H, res, -1)
    inlet_polygon.append(pt_1)

    outlet_polygon = [pt_4starstar]
    outlet_polygon.append(pt_4star)
    add_vertical_boundary_vertices(outlet_polygon, L, H, res, 1)
    outlet_polygon.append(pt_2)
    outlet_polygon.append(pt_2starstar)
    add_vertical_boundary_vertices(outlet_polygon, L-L/res, H, res, -1)
    outlet_polygon.append(pt_4starstar)

    inlet1 = mshr.Polygon(inlet_polygon)
    inlet2 = mshr.Rectangle(pt_1starstar, pt_3)
    outlet1 = mshr.Polygon(outlet_polygon)
    outlet2 = mshr.Rectangle(pt_4, pt_2starstar)
    channel = mshr.Rectangle(pt_5, pt_8)
    pos_cir_1 = mshr.Circle(pt_5, R, segments=n_segments)
    pos_cir_2 = mshr.Circle(pt_6, R, segments=n_segments)
    pos_cir_3 = mshr.Circle(pt_7, R, segments=n_segments)
    pos_cir_4 = mshr.Circle(pt_8, R, segments=n_segments)
    neg_cir_1 = mshr.Circle(pt_9, R, segments=n_segments)
    neg_cir_2 = mshr.Circle(pt_10, R, segments=n_segments)
    neg_cir_3 = mshr.Circle(pt_11, R, segments=n_segments)
    neg_cir_4 = mshr.Circle(pt_12, R, segments=n_segments)
    neg_reg_1 = mshr.Rectangle(pt_13, pt_12)
    neg_reg_2 = mshr.Rectangle(pt_9, pt_14)

    domain = inlet1 + inlet2 + outlet1 + outlet2 + \
        channel + pos_cir_1 + pos_cir_2 + pos_cir_3 + \
        pos_cir_4 - neg_cir_1 - neg_cir_2 - neg_cir_3 - \
        neg_cir_4 - neg_reg_1 - neg_reg_2

    mesh = mshr.generate_mesh(domain, res)

    mesh_path = os.path.join(MESHES_DIR,
                             "roundet_barbell_res" + str(res))
    store_mesh_HDF5(mesh, mesh_path)
    df.plot(mesh)
    df.interactive()
