import sys
import os
from itertools import product
import lattice_figure_functions as pplocal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

from hams import *

sys.path.insert(0,"./../../")

import toolkit_local.hdf5 as hdf

try:
    import pyvista as pv
except ImportError as exc:
    raise ImportError("plot_LIF_pyvista.py requires pyvista") from exc

try:
    from PIL import Image
except ImportError:
    Image = None


def parse_bool(value):
    return str(value).lower() in ["1","true","yes","y","on"]


kres = int(sys.argv[1])
scan_res = int(sys.argv[2])
phase = str(sys.argv[3])


show_interactive = False


fname = f"./h5_files_new/prep_phase={phase}_kres={kres}_scan_res={scan_res}.h5"
save_folder = f"./figs/"


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
        tube_radius=pplocal.tube_radius_LIF,
        tube_sides=tube_sides,
    )


def make_face_polydata(face_index):
    rs_scan = hdf.load_hdf5_to_numpy(fname,f"/prep/scan_meshes_dict/{face_index}/rs_scan")
    xs_mesh = hdf.load_hdf5_to_numpy(fname,f"/prep/scan_meshes_dict/{face_index}/xs_mesh")
    ys_mesh = hdf.load_hdf5_to_numpy(fname,f"/prep/scan_meshes_dict/{face_index}/ys_mesh")
    gaps = hdf.load_hdf5_to_numpy(fname,f"/{face_index}")

    if rs_scan.shape[:2] != gaps.shape:
        raise ValueError(f"face {face_index} rs_scan shape {rs_scan.shape[:2]} does not match gaps shape {gaps.shape}")

    nx,ny = gaps.shape
    points = rs_scan.reshape(-1,3)
    quads = []
    lif = []

    for ix in range(nx-1):
        for iy in range(ny-1):
            x_cell = 0.25*(xs_mesh[ix,iy] + xs_mesh[ix+1,iy] + xs_mesh[ix+1,iy+1] + xs_mesh[ix,iy+1])
            y_cell = 0.25*(ys_mesh[ix,iy] + ys_mesh[ix+1,iy] + ys_mesh[ix+1,iy+1] + ys_mesh[ix,iy+1])

            keep_cell = True
            if face_index < 3:
                keep_cell = not ((x_cell >= 0.5) and (y_cell >= 0.5))

            if not keep_cell:
                continue

            p00 = ix*ny + iy
            p10 = (ix+1)*ny + iy
            p11 = (ix+1)*ny + (iy+1)
            p01 = ix*ny + (iy+1)

            quads.extend([4,p00,p10,p11,p01])
            lif.append(0.25*(gaps[ix,iy] + gaps[ix+1,iy] + gaps[ix+1,iy+1] + gaps[ix,iy+1]))

    mesh = pv.PolyData(points,np.asarray(quads,dtype=np.int64))
    mesh.cell_data["lif_raw"] = np.asarray(lif,dtype=float)

    print(f"face {face_index} vertices {rs_scan.shape[:2]} cells {mesh.n_cells}")

    return mesh


def meshes_norm(meshes):
    all_lif = np.concatenate([
        mesh.cell_data["lif_raw"]
        for mesh in meshes
        if mesh.n_cells > 0
    ])

    finite_lif = all_lif[np.isfinite(all_lif)]
    positive_lif = finite_lif[finite_lif > 0]

    if len(positive_lif) == 0:
        raise ValueError("all plotted face cell values are non-positive or non-finite; cannot use log color scale")


    vmin = max(np.min(positive_lif),1e-10)
    vmax = 1.0

    for mesh in meshes:
        lif_normalized = mesh.cell_data["lif_raw"]
        lif_normalized = np.where(lif_normalized > 0,lif_normalized,vmin)
        mesh.cell_data["lif"] = lif_normalized

    return [vmin,vmax]


def save_matplotlib_colorbar(
    savepath,
    cmap="magma_r",
    clim=None,
    log_scale=True,
    label=r"$\mu(\mathbf{r})$",
    figsize=(0.45, 3.0),
    dpi=300,
    fontsize=20,
):
    
    # def activate_latex(font_size=18,legend_font_size=12) : 

    #     plt.rcParams.update({
    #         "text.usetex": True,
    #         "font.family":"Computer Modern Roman"
    #     })
    #     plt.rcParams.update({'font.size': font_size})
    #     print(type(plt.rcParams['text.latex.preamble']))
    #     plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \usepackage{physics} \newcommand{\rv}{{\bf r}}'
    #     print(plt.rcParams['text.latex.preamble'])
    #     plt.rc("legend",fontsize=legend_font_size)
        
    # activate_latex(30)
    
    if clim is None:
        raise ValueError("clim is required to build the matplotlib colorbar")

    if log_scale:
        norm = mcolors.LogNorm(vmin=clim[0], vmax=clim[1])
    else:
        norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])

    fig, ax = plt.subplots(figsize=figsize)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax)

    if log_scale:
        cbar.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        cbar.ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    cbar.ax.tick_params(labelsize=fontsize, which="both")
    cbar.set_label(label, fontsize=fontsize,rotation=270,labelpad=20)
    fig.savefig(savepath, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Matplotlib colorbar saved -> {savepath}")


os.makedirs(save_folder,exist_ok=True)

a_mat = hdf.load_hdf5_to_numpy(fname,"/prep/lat_vecs_direct")
meshes = [make_face_polydata(face_index) for face_index in range(6)]
clim_raw_data_bounds = meshes_norm(meshes)
clim = (1e-1,1)
lif_cmap = "magma_r"
lif_log_scale = True


save_dpi = pplocal.save_dpi

plotter = pv.Plotter(off_screen=not show_interactive)
plotter.set_background("white")

for mesh_index,mesh in enumerate(meshes):
    if mesh.n_cells == 0:
        continue

    plotter.add_mesh(
        mesh,
        scalars="lif",
        preference="cell",
        cmap=lif_cmap,
        clim=clim,
        log_scale=lif_log_scale,
        show_edges=False,
        smooth_shading=False,
        lighting=False,
        show_scalar_bar=False,
        scalar_bar_args={
            "title": "LIF (normalized)",
            "vertical": True,
        },
    )

cut = lambda a,b,c: (a > 0.5) & (b > 0.5) & (c < 0.5)

plot_clipped_unit_cell(
    plotter,
    a_mat,
    cut,
    edge_color=pplocal.primitive_uc_color,
    use_tubes=True,
    tube_radius=pplocal.tube_radius_LIF,
    tube_sides=24,
)

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


pplocal.add_line_segments(plotter,novel_octant_segments,pplocal.octant_color,use_tubes=True,tube_radius=pplocal.tube_radius_LIF)
pplocal.set_camera_position(plotter,pplocal.camera_position_LIF)
pplocal.add_lighting(plotter)

subfig_letter = 'd' if phase == 'topo' else 'c'

save_path = save_folder + f"main_text_figure_3_{subfig_letter}_lif_phase={phase}.png"
colorbar_path = save_folder + f"main_text_figure_3_cd_lif_phase={phase}_colorbar.png"

if show_interactive:
    plotter.show()
    print(plotter.camera_position)
else:
    plotter.show(auto_close=False)
    print(plotter.camera_position)
    pplocal.screenshot_plotter(plotter,save_path,savepath_dpi=save_dpi,autocrop_params=pplocal.autocrop_base)
    plotter.close()
save_matplotlib_colorbar(
    colorbar_path,
    cmap=lif_cmap,
    clim=clim,
    log_scale=lif_log_scale,
    dpi=save_dpi,
)
