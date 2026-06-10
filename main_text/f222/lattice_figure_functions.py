import pyvista as pv
import numpy as np

conventional_uc_color = 'k'
primitive_uc_color = 'grey'
octant_color = 'g'
sphere_color_4a='b'
sphere_color_4b='r'
sphere_radius = 0.05
tube_radius_graphic = 0.005
tube_radius_LIF = 0.005

face_opacity=0.4

save_dpi=2400
base_dpi=300
screenshot_window_size = None


autocrop_base = {
    "enabled": True,
    "padding": 20,
    "background": None,
    "fallback_background": "white",
    "tolerance": 8,
}

camera_position_LIF = [(1.2353727903846918, 1.235372790384692, 4.3812364373217),
 (0.5000000074505806, 0.5000000074505806, 0.5000000223517418),
 (0.0, 0.0, -1.0)]

camera_position_conventional = [(-3.1725334285445155, -0.7682354042174562, 1.5245094028549742),
 (0.5000000074505806, 0.5000000074505806, 0.5000000223517418),
 (0.23735195008513907, 0.09375523017491674, 0.9668888294967682)]
# [(-3.0747741299907987, -1.1669445490127643, 1.2666994945094014),
#  (0.5000000074505806, 0.5000000074505806, 0.5000000223517418),
#  (0.0, 0.0, 1.0)]

def set_camera_position(plotter, camera_position=None):
    if camera_position is not None:
        plotter.camera_position = camera_position


def dpi_to_screenshot_scale(dpi, dpi_base=base_dpi):
    return max(1, round(float(dpi) / float(dpi_base)))


def autocrop_png(png_path, crop_params=None):
    crop_params = crop_params or {}
    if not crop_params.get("enabled", False):
        return

    try:
        from PIL import Image, ImageColor
    except ImportError:
        print("Pillow not installed; autocrop skipped.")
        return

    img = Image.open(png_path).convert("RGB")
    arr = np.asarray(img)
    background = crop_params.get("background") or crop_params.get("fallback_background", "black")

    try:
        bg_rgb = np.array(ImageColor.getrgb(background), dtype=np.int16)
    except ValueError:
        bg_rgb = arr[0, 0].astype(np.int16)

    diff = np.abs(arr.astype(np.int16) - bg_rgb)
    mask = np.any(diff > int(crop_params.get("tolerance", 8)), axis=2)

    if not np.any(mask):
        print("Autocrop skipped; no non-background pixels found.")
        return

    ys, xs = np.where(mask)
    pad = int(crop_params.get("padding", 20))
    left = max(int(xs.min()) - pad, 0)
    right = min(int(xs.max()) + pad + 1, img.width)
    top = max(int(ys.min()) - pad, 0)
    bottom = min(int(ys.max()) + pad + 1, img.height)

    img.crop((left, top, right, bottom)).save(png_path)
    print(f"Autocropped -> {png_path}")


def screenshot_plotter(plotter, savepath, savepath_dpi=300, autocrop_params=None):
    png_path = savepath if savepath.lower().endswith(".png") else savepath + ".png"
    quality_scale = dpi_to_screenshot_scale(savepath_dpi)
    plotter.screenshot(
        png_path,
        window_size=screenshot_window_size,
        scale=quality_scale,
    )
    print(f"Screenshot saved -> {png_path} (scale={quality_scale})")
    autocrop_png(png_path, autocrop_params)

    if savepath.lower().endswith(".pdf"):
        try:
            from PIL import Image
            img = Image.open(png_path)
            img.save(savepath, "PDF", resolution=savepath_dpi)
            print(f"PDF saved -> {savepath}")
        except ImportError:
            print("Pillow not installed; PDF conversion skipped (PNG saved).")


def close_plotter(
    plotter,
    interactive=True,
    savepath=None,
    savepath_dpi=300,
    print_camera=False,
    autocrop_params=None,
):
    def _on_close(p):
        if print_camera:
            print("camera_position =", p.camera_position)
        if savepath is not None:
            screenshot_plotter(p, savepath, savepath_dpi, autocrop_params)

    if savepath is not None or print_camera:
        if interactive:
            plotter.show(before_close_callback=_on_close)
        else:
            plotter.show(auto_close=False)
            if print_camera:
                print("camera_position =", plotter.camera_position)
            if savepath is not None:
                screenshot_plotter(plotter, savepath, savepath_dpi, autocrop_params)
            plotter.close()
    elif interactive:
        plotter.show()


def make_line_polydata(points, segments):
    points = np.asarray(points,dtype=float)
    if len(points) == 0 or len(segments) == 0:
        return pv.PolyData(np.empty((0,3)))

    lines = np.asarray(
        [[2,int(ii),int(jj)] for ii,jj in segments],
        dtype=np.int64,
    ).ravel()

    return pv.PolyData(points,lines=lines)


def make_polyline_polydata(polylines):
    points = []
    lines = []

    for polyline in polylines:
        if len(polyline) < 2:
            continue

        start = len(points)
        points.extend(polyline)
        point_ids = list(range(start,start + len(polyline)))
        lines.extend([len(point_ids),*point_ids])

    if len(points) == 0:
        return pv.PolyData(np.empty((0,3)))

    return pv.PolyData(np.asarray(points,dtype=float),lines=np.asarray(lines))


def add_rhombus_surface(
    plotter,
    v1,
    v2,
    base=np.zeros(3),
    color="grey",
    opacity=0.35,
    show_edges=False,
    edge_color="black",
    edge_width=2,
):
    v1 = np.asarray(v1,dtype=float)
    v2 = np.asarray(v2,dtype=float)
    base = np.asarray(base,dtype=float)

    if v1.shape != (3,):
        raise ValueError("v1 must be a 3D vector")
    if v2.shape != (3,):
        raise ValueError("v2 must be a 3D vector")
    if base.shape != (3,):
        raise ValueError("base must be a 3D vector")

    corners = np.array([
        base,
        base + v1,
        base + v1 + v2,
        base + v2,
    ])
    faces = np.array([4,0,1,2,3],dtype=np.int64)
    mesh = pv.PolyData(corners,faces)

    plotter.add_mesh(
        mesh,
        color=color,
        opacity=opacity,
        lighting=True,
        ambient=0.55,
        diffuse=0.45,
        specular=0.0,
    )

    if show_edges:
        add_line_segments(
            plotter,
            [
                (corners[0],corners[1]),
                (corners[1],corners[2]),
                (corners[2],corners[3]),
                (corners[3],corners[0]),
            ],
            color=edge_color,
            line_width=edge_width,
        )


def segments_to_polylines(segments, decimals=12):
    if len(segments) == 0:
        return []

    def key(point):
        return tuple(np.round(np.asarray(point,dtype=float),decimals=decimals))

    points_by_key = {}
    adjacency = {}
    edges = set()

    for p1,p2 in segments:
        k1,k2 = key(p1),key(p2)
        if k1 == k2:
            continue

        points_by_key.setdefault(k1,np.asarray(p1,dtype=float))
        points_by_key.setdefault(k2,np.asarray(p2,dtype=float))
        adjacency.setdefault(k1,set()).add(k2)
        adjacency.setdefault(k2,set()).add(k1)
        edges.add(frozenset((k1,k2)))

    visited = set()
    polylines = []

    def walk(start, next_key):
        path = [start]
        prev = start
        current = next_key
        visited.add(frozenset((start,next_key)))

        while True:
            path.append(current)
            candidates = [
                candidate for candidate in adjacency[current]
                if candidate != prev and frozenset((current,candidate)) not in visited
            ]
            if len(candidates) != 1:
                break

            prev,current = current,candidates[0]
            visited.add(frozenset((prev,current)))

        return [points_by_key[item] for item in path]

    start_keys = [
        item for item,neighbors in adjacency.items()
        if len(neighbors) != 2
    ]

    for start in start_keys:
        for neighbor in adjacency[start]:
            edge = frozenset((start,neighbor))
            if edge not in visited:
                polylines.append(walk(start,neighbor))

    for edge in edges:
        if edge in visited:
            continue

        start,neighbor = tuple(edge)
        polylines.append(walk(start,neighbor))

    return polylines


def add_line_segments(plotter, segments, color='white', line_width=2,
                      use_tubes=False, tube_radius=0.01, tube_sides=24):
    if len(segments) == 0:
        return

    if use_tubes:
        mesh = make_polyline_polydata(segments_to_polylines(segments))
        if mesh.n_points == 0:
            return

        mesh = mesh.tube(radius=tube_radius,n_sides=tube_sides,capping=True)
        plotter.add_mesh(
            mesh,
            color=color,
            smooth_shading=True,
            lighting=True,
            ambient=0.55,
            diffuse=0.45,
            specular=0.0,
        )
        return

    points = []
    line_indices = []
    for p1,p2 in segments:
        idx = len(points)
        points.extend([p1,p2])
        line_indices.append((idx,idx+1))

    mesh = make_line_polydata(points,line_indices)
    plotter.add_mesh(
        mesh,
        color=color,
        line_width=line_width,
        lighting=True,
        ambient=0.55,
        diffuse=0.45,
        specular=0.0,
    )


def add_points(plotter, points, color='white', point_size=16,
               use_glyphs=False, sphere_radius=0.045):
    """
    Add Cartesian points to a PyVista plotter as sphere-like markers.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return

    points = np.atleast_2d(points)
    if use_glyphs:
        sphere = pv.Sphere(
            radius=sphere_radius,
            theta_resolution=48,
            phi_resolution=48,
        )
        glyphs = pv.PolyData(points).glyph(geom=sphere, scale=False, orient=False)
        plotter.add_mesh(
            glyphs,
            color=color,
            smooth_shading=True,
            lighting=True,
            ambient=0.12,
            diffuse=0.72,
            specular=0.8,
            specular_power=90,
        )
        return

    plotter.add_points(
        points,
        color=color,
        point_size=point_size,
        render_points_as_spheres=True,
        lighting=True,
        ambient=0.25,
        diffuse=0.75,
        specular=0.35,
        specular_power=25,
    )


def add_lighting(plotter):
    """
    Add a restrained studio-light setup for the lattice markers and wireframe.
    """
    bounds = plotter.bounds
    center = np.array([
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    ])
    span = np.array([
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    ])
    scale = max(np.linalg.norm(span), 1.0)

    camera_position = np.asarray(plotter.camera.position)
    camera_direction = camera_position - center
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    up = np.asarray(plotter.camera.up)
    up = up / np.linalg.norm(up)
    side = np.cross(camera_direction, up)
    side = side / np.linalg.norm(side)

    plotter.remove_all_lights()

    lights = [
        # pv.Light(
        #     position=center + scale * (10 * camera_direction),
        #     focal_point=center,
        #     color='white',
        #     intensity=1,#0.85,
        # ),
        pv.Light(
            position=center + scale * (1.4 * camera_direction + 0.8 * up + 0.5 * side),
            focal_point=center,
            color='white',
            intensity=1,#0.85,
        ),
        pv.Light(
            position=center + scale * (0.8 * camera_direction + 0.3 * up - 1.2 * side),
            focal_point=center,
            color='white',
            intensity=0.35,
        ),
        pv.Light(
            position=center + scale * (1.8 * up - 0.4 * camera_direction),
            focal_point=center,
            color='white',
            intensity=0.25,
        ),
    ]

    for light in lights:
        plotter.add_light(light)



def plot_unit_cell_wireframe(plotter, lattice_matrix,c1='white',c2='black',base=np.zeros(3),lw=2,
                             use_tubes=False, tube_radius=0.01, tube_sides=24):
    """
    Plot a wireframe of the unit cell defined by the direct lattice vectors.

    Parameters
    ----------
    plotter : pyvista.Plotter
    lattice_matrix : (3,3) array-like
        Rows are the direct lattice vectors, i.e.
        a = lattice_matrix[0], b = lattice_matrix[1], c = lattice_matrix[2]
    """
    L = np.array(lattice_matrix)
    a, b, c = L[0], L[1], L[2]

    # 8 corners of the unit cell: all combinations of 0/1 coefficients
    corners = np.array([
        i*a + j*b + k*c + base
        for i in (0, 1)
        for j in (0, 1)
        for k in (0, 1)
    ])  # shape (8, 3)

    # Each edge connects two corners that differ in exactly one lattice direction
    edges = [
        (i, j)
        for i in range(8)
        for j in range(i+1, 8)
        if bin(i ^ j).count('1') == 1   # differ in exactly one bit
    ]

    mesh = make_line_polydata(corners, edges)
    if use_tubes:
        mesh = mesh.tube(radius=tube_radius, n_sides=tube_sides, capping=True)
        plotter.add_mesh(
            mesh,
            color=c2,
            smooth_shading=True,
            lighting=True,
            ambient=0.55,
            diffuse=0.45,
            specular=0.0,
        )
        return

    plotter.add_mesh(
        mesh,
        color=c2,
        line_width=lw,
        lighting=True,
        ambient=0.55,
        diffuse=0.45,
        specular=0.0,
    )



