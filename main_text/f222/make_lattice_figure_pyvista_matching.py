from hams import *
import lattice_figure_functions as pplocal
from itertools import product

import pyvista as pv

try:
    from PIL import Image
except ImportError:
    Image = None



def plot_clipped_unit_cell(plotter, lattice_matrix, remove_condition,
                           edge_color='white', linewidth=2, halo=True,
                           exclude_top_corner=True,
                           use_tubes=False, tube_radius=0.01, tube_sides=24):
    """
    Plot wireframe of unit cell with an octant removed.

    Parameters
    ----------
    plotter : pyvista.Plotter
    lattice_matrix : (3,3) array-like
        Rows are direct lattice vectors a, b, c
    remove_condition : callable
        Function f(a, b, c) -> bool, returning True for points to REMOVE.
        Coordinates are in fractional (lattice) units, range (0,1).
        e.g. lambda a,b,c: (a>0.5) & (b>0.5) & (c<0.5)
    exclude_top_corner : bool
        If True, excludes the 3 edges connected to the corner (0, 0, 1).
    """
    L = np.array(lattice_matrix)
    segments = []

    def frac_to_cart(fa, fb, fc):
        return fa * L[0] + fb * L[1] + fc * L[2]

    def plot_edge(p1, p2):
        segments.append((np.asarray(p1), np.asarray(p2)))

    N = 500

    # The 3 edges connected to (0, 0, 1) in fractional coords
    top_corner_edges = {
        (tuple([0, 0, 1]), tuple([1, 0, 1])),  # along a
        (tuple([0, 0, 1]), tuple([0, 1, 1])),  # along b
        (tuple([0, 0, 0]), tuple([0, 0, 1])),  # along c
    }

    # All 12 edges of the unit cell
    edges_frac = []
    for fixed in product([0, 1], repeat=2):
        edges_frac.append(
            (np.array([0, fixed[0], fixed[1]]),
             np.array([1, fixed[0], fixed[1]]))
        )
        edges_frac.append(
            (np.array([fixed[0], 0, fixed[1]]),
             np.array([fixed[0], 1, fixed[1]]))
        )
        edges_frac.append(
            (np.array([fixed[0], fixed[1], 0]),
             np.array([fixed[0], fixed[1], 1]))
        )

    # Deduplicate and optionally exclude top corner edges
    seen = set()
    unique_edges = []
    for e in edges_frac:
        key = (tuple(e[0]), tuple(e[1]))
        if key in seen:
            continue
        seen.add(key)
        if exclude_top_corner and key in top_corner_edges:
            continue
        unique_edges.append(e)

    # For each original edge, find and plot the kept segments
    for start_f, end_f in unique_edges:
        t = np.linspace(0, 1, N)
        pts_f = np.outer(1 - t, start_f) + np.outer(t, end_f)
        kept = ~remove_condition(pts_f[:, 0], pts_f[:, 1], pts_f[:, 2])

        segment = []
        for i, (keep, pf) in enumerate(zip(kept, pts_f)):
            if keep:
                segment.append(frac_to_cart(*pf))
            elif segment:
                if len(segment) > 1:
                    plot_edge(segment[0], segment[-1])
                segment = []
        if len(segment) > 1:
            plot_edge(segment[0], segment[-1])

   

    pplocal.add_line_segments(
        plotter,
        segments,
        color=edge_color,
        line_width=linewidth,
        use_tubes=use_tubes,
        tube_radius=tube_radius,
        tube_sides=tube_sides,
    )
    



tube_sides = 24
use_sphere_glyphs = True
use_line_tubes = True


save_dpi = pplocal.save_dpi
show_interactive = False


plotter = pv.Plotter(off_screen=not show_interactive)
plotter.set_background('white')




(a1,a2,a3), (b1,b2,b3) = f222_lat_vecs()

novel_octant_segments = []

faces_shifts =   [np.zeros(3),a1+a3,a2+a3,(a1+a2+a3)*0.5,(a1+a2+a3)*0.5,(a1+a2+a3)*0.5]
faces_spanners = [[a1,a2],[a2,-a3],[a1,-a3],[0.5*a1,-0.5*a3],[0.5*a1,0.5*a2],[0.5*a2,-0.5*a3]]

for ii in [3,4,5] : 
    v1,v2 = faces_spanners[ii]
    base = faces_shifts[ii]
    
    novel_octant_segments.append((base,base + v1))
    novel_octant_segments.append((base,base + v2))
    novel_octant_segments.append((base + v2,base + v2 + v1))
    novel_octant_segments.append((base + v1,base + v2 + v1))


a_mat = np.stack([a1,a2,a3])



# plotter.hide_axes()          # removes axis lines, ticks, and labels
# plotter.show_axes()

original_view_dict = {"elev" : 45,
                    "azim" : -135,
                    "roll" : 0}

elev=105

shift=0.5*np.array([0,0,1])

num_uc = 3



plot_all = True
save_path = f"./figs/main_text_figure_3_b_lattice_figure_plot_matching.png"



for ii in range(2) :
    for jj in range(2) :
        for kk in range(2) :

            vec_tmp = ii*a1 + jj*a2 + kk*a3
            if not (((ii!=0) and (jj!=0) and (kk!=1)) or ((ii!=1) and (jj!=1) and (kk!=0))) : 
                pplocal.add_points(plotter, [vec_tmp], color=pplocal.sphere_color_4a,
                use_glyphs=use_sphere_glyphs, sphere_radius=pplocal.sphere_radius)
            
            

pplocal.add_points(plotter, [np.array([0.5,0.5,0.5])], color=pplocal.sphere_color_4b,
            use_glyphs=use_sphere_glyphs, sphere_radius=pplocal.sphere_radius)

cut = lambda a, b, c: (a > 0.5) & (b > 0.5) & (c < 0.5)

pplocal.add_line_segments(plotter,novel_octant_segments,pplocal.octant_color,use_tubes=True,tube_radius=pplocal.tube_radius_graphic)

plot_clipped_unit_cell(
    plotter,
    a_mat,
    cut,
    edge_color=pplocal.primitive_uc_color,
    linewidth=5,
    use_tubes=use_line_tubes,
    tube_radius=pplocal.tube_radius_graphic,
    tube_sides=tube_sides,
)


# pplocal.add_rhombus_surface(plotter,a1,a2,base=a3,opacity=pplocal.face_opacity)
# pplocal.add_rhombus_surface(plotter,a1,a3,base=a2,opacity=pplocal.face_opacity)

pplocal.add_rhombus_surface(plotter,a3*0.5,a2*0.5,base=a1,opacity=pplocal.face_opacity)
pplocal.add_rhombus_surface(plotter,a3*0.5,a2*0.5,base=a1+0.5*(a3 + a2),opacity=pplocal.face_opacity)
pplocal.add_rhombus_surface(plotter,-a3*0.5,a2*0.5,base=a1+a3,opacity=pplocal.face_opacity)

pplocal.add_rhombus_surface(plotter,a3*0.5,a1*0.5,base=a2,opacity=pplocal.face_opacity)
pplocal.add_rhombus_surface(plotter,a3*0.5,a1*0.5,base=a2+0.5*(a3 + a1),opacity=pplocal.face_opacity)
pplocal.add_rhombus_surface(plotter,-a3*0.5,a1*0.5,base=a2+a3,opacity=pplocal.face_opacity)

pplocal.add_rhombus_surface(plotter,a1*0.5,a2*0.5,base=np.zeros(3),opacity=pplocal.face_opacity)
pplocal.add_rhombus_surface(plotter,a1*0.5,a2*0.5,base=0.5*a1,opacity=pplocal.face_opacity)
pplocal.add_rhombus_surface(plotter,a1*0.5,a2*0.5,base=0.5*a2,opacity=pplocal.face_opacity)

pplocal.add_rhombus_surface(plotter,a3*0.5,a2*0.5,base=0.5*(a1+a2),opacity=pplocal.face_opacity,color='g')
pplocal.add_rhombus_surface(plotter,a3*0.5,a1*0.5,base=0.5*(a1+a2),opacity=pplocal.face_opacity,color='g')
pplocal.add_rhombus_surface(plotter,a1*0.5,a2*0.5,base=0.5*(a1+a2+a3),opacity=pplocal.face_opacity,color='g')

pplocal.set_camera_position(plotter,pplocal.camera_position_LIF)
pplocal.add_lighting(plotter)


if show_interactive:
    plotter.show()
    print(plotter.camera_position)
else:
    plotter.show(auto_close=False)
    print(plotter.camera_position)
    pplocal.screenshot_plotter(plotter,save_path,savepath_dpi=save_dpi,autocrop_params=pplocal.autocrop_base)
    plotter.close()



