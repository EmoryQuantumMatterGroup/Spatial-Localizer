from hams import *

from itertools import product

import pyvista as pv

try:
    from PIL import Image
except ImportError:
    Image = None

import lattice_figure_functions as pplocal




save_dpi=pplocal.save_dpi
use_sphere_glyphs = True
use_line_tubes = True
tube_sides = 24



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
save_path = f"./figs/main_text_figure_3_a_lattice_figure_plot_all={plot_all}.png"



if True : ## plotting scatter points
    conventional_corners = [0*a1,a1+a2-a3,a1-a2+a3,-a1+a2+a3,a1+a2+a3,a3*2,a1*2,a2*2]
    conventional_faces = [a1,a2,a3,a1+a2,a2+a3,a1+a3]
    conventional_edges = []
    for ii in [0,1] :
        for jj in [0,1] :

            tmp_vec = np.array([ii,jj,0.5])

            for i_roll in range(3) :

                conventional_edges.append(np.roll(tmp_vec,i_roll))


    tol = 0.1

    for vec_tmp in conventional_corners :
        pplocal.add_points(plotter, [vec_tmp], color=pplocal.sphere_color_4a,
                use_glyphs=use_sphere_glyphs, sphere_radius=pplocal.sphere_radius)

        vec_tmp2 = vec_tmp+shift

        if not ((vec_tmp2[0]>1+tol) or (vec_tmp2[1]>1+tol) or (vec_tmp2[2]>1+tol)):

            pplocal.add_points(plotter, [vec_tmp2], color=pplocal.sphere_color_4b,
                    use_glyphs=use_sphere_glyphs, sphere_radius=pplocal.sphere_radius)

    for vec_tmp in conventional_faces :
        pplocal.add_points(plotter, [vec_tmp], color=pplocal.sphere_color_4a,
                use_glyphs=use_sphere_glyphs, sphere_radius=pplocal.sphere_radius)
        vec_tmp2 = vec_tmp+shift

        if not ((vec_tmp2[0]>1+tol) or (vec_tmp2[1]>1+tol) or (vec_tmp2[2]>1+tol)):

            pplocal.add_points(plotter, [vec_tmp2], color=pplocal.sphere_color_4b,
                    use_glyphs=use_sphere_glyphs, sphere_radius=pplocal.sphere_radius)

    for vec_tmp in conventional_edges :
        pplocal.add_points(plotter, [vec_tmp], color=pplocal.sphere_color_4b,
                use_glyphs=use_sphere_glyphs, sphere_radius=pplocal.sphere_radius)


cut = lambda a, b, c: (a > 0.5) & (b > 0.5) & (c < 0.5)


x,y,z = np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])

conventional_lat_mat = np.stack([x,y,z],axis=0)

pplocal.plot_unit_cell_wireframe(
    plotter,
    conventional_lat_mat,
    use_tubes=use_line_tubes,
    tube_radius=pplocal.tube_radius_graphic,
    tube_sides=tube_sides,c2=pplocal.conventional_uc_color
)

pplocal.plot_unit_cell_wireframe(
    plotter,
    a_mat,
    use_tubes=use_line_tubes,
    tube_radius=pplocal.tube_radius_graphic,
    tube_sides=tube_sides,c2=pplocal.primitive_uc_color
)

# pplocal.add_line_segments(plotter,novel_octant_segments,pplocal.octant_color,use_tubes=True,tube_radius=pplocal.tube_radius_graphic)


pplocal.set_camera_position(plotter,pplocal.camera_position_conventional)

pplocal.add_lighting(plotter)
print(plotter.camera_position)


pplocal.add_rhombus_surface(plotter,a1,a2,opacity=pplocal.face_opacity)
pplocal.add_rhombus_surface(plotter,a1,a3,opacity=pplocal.face_opacity)
pplocal.add_rhombus_surface(plotter,a3,a2,opacity=pplocal.face_opacity)
pplocal.add_rhombus_surface(plotter,a1,a2,base=a3,opacity=pplocal.face_opacity)
pplocal.add_rhombus_surface(plotter,a1,a3,base=a2,opacity=pplocal.face_opacity)
pplocal.add_rhombus_surface(plotter,a3,a2,base=a1,opacity=pplocal.face_opacity)


if show_interactive:
    plotter.show()
    print(plotter.camera_position)
else:
    plotter.show(auto_close=False)
    print(plotter.camera_position)
    pplocal.screenshot_plotter(plotter,save_path,savepath_dpi=save_dpi,autocrop_params=pplocal.autocrop_base)
    plotter.close()
