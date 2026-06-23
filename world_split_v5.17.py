import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import trimap_beta as tm
import math
import cv2
import numpy as np
import ctypes
import random
from read_config import read_config
from trackers import get_trackers_from_file
from color_picking import find_blob_centers
import sys
import os
import tqdm


MAP_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
ACTIVE_MAP = None

MAP_PROFILES = {
    "map_1": {
        "height_path": os.path.join("maps", "map_1.png"),
        "color_path": os.path.join("colors", "col_1.png"),
        "margin": 8,
        "map_scale": 14.0,
        "blur_sigma": 3.0,
    },
    "map_2": {
        "height_path": os.path.join("maps", "map_2.png"),
        "color_path": os.path.join("colors", "col_2.png"),
        "margin": 8,
        "map_scale": 20.0,
        "blur_sigma": 0.0,
    },
    "map_3": {
        "height_path": os.path.join("maps", "map_3.jpg"),
        "color_path": os.path.join("colors", "col_3.png"),
        "margin": 10,
        "map_scale": 15.0,
        "blur_sigma": 4.0,
    },
    "map_4": {
        "height_path": os.path.join("maps", "map_4.png"),
        "color_path": os.path.join("colors", "col_4.png"),
        "margin": 4,
        "map_scale": 150.0,
        "blur_sigma": 2.0,
    },
}

DEFAULT_MAP_PROFILE = {
    "margin": 8,
    "map_scale": 20.0,
    "blur_sigma": 0.0,
}


def select_map_path(maps_dir="maps"):
    map_files = sorted(
        name for name in os.listdir(maps_dir)
        if os.path.isfile(os.path.join(maps_dir, name))
        and os.path.splitext(name)[1].lower() in MAP_EXTENSIONS
    )
    if not map_files:
        print(f"No map files found in '{maps_dir}'.")
        sys.exit(1)

    print("\nSelect a map:\n")
    for i, filename in enumerate(map_files, start=1):
        print(f"{i}. {os.path.splitext(filename)[0]}")

    while True:
        choice = input("\nEnter selection: ").strip()
        try:
            index = int(choice)
        except ValueError:
            print(f"Invalid selection. Enter a number from 1 to {len(map_files)}.")
            continue
        if 1 <= index <= len(map_files):
            return os.path.join(maps_dir, map_files[index - 1])
        print(f"Invalid selection. Enter a number from 1 to {len(map_files)}.")

def select_matching_color_map(map_path, colors_dir="colors"):
    """
    For maps/map_N.ext, try colors/col_N with a supported image extension.
    Return the path if found, otherwise None.
    """
    map_stem = os.path.splitext(os.path.basename(map_path))[0]  # e.g. map_2

    if map_stem.startswith("map_"):
        suffix = map_stem[len("map_"):]                         # e.g. 2
        color_stem = f"col_{suffix}"                            # e.g. col_2
    else:
        color_stem = map_stem

    for extension in MAP_EXTENSIONS:
        candidate = os.path.join(colors_dir, color_stem + extension)
        if os.path.isfile(candidate):
            return candidate

    return None


def select_map_profile():
    selected_path = select_map_path()
    map_name = os.path.splitext(os.path.basename(selected_path))[0]

    if map_name in MAP_PROFILES:
        profile = dict(MAP_PROFILES[map_name])
    else:
        profile = dict(DEFAULT_MAP_PROFILE)
        profile["height_path"] = selected_path
        profile["color_path"] = select_matching_color_map(selected_path)

    profile["name"] = map_name
    profile["tri_path"] = os.path.join("generated", f"{map_name}.tri")

    color_path = profile.get("color_path")
    if color_path and not os.path.isfile(color_path):
        profile["color_path"] = None

    return profile

def has_minimum_pnp_points(count):
    if count < 4:
        print(
            "PnP requires at least 4 real 2D-3D correspondences; "
            f"currently have {count}."
        )
        return False
    return True


def clear_picking_state():
    global pending_left_world_point, picked_points, picked_correspondences
    global pnp_result
    pending_left_world_point = None
    picked_points = []
    picked_correspondences = []
    pnp_result = None


def focus_pygame_window():
    if sys.platform != "win32":
        return False
    try:
        hwnd = pygame.display.get_wm_info().get("window")
        if not hwnd:
            return False

        user32 = ctypes.windll.user32
        SW_RESTORE = 9
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        user32.SetFocus(hwnd)
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Clipping planes
# ---------------------------------------------------------------------------
NEAR =   0.1
FAR  = 5000.0

# ---------------------------------------------------------------------------
# Tracker rendering
# ---------------------------------------------------------------------------
TRACKER_RADIUS = 2   # world-space radius of each tracker sphere

# ---------------------------------------------------------------------------
# Pyramid VBO  (built once, instanced per camera position)
# ---------------------------------------------------------------------------
_PYRAMID_TIP  = (0.0, 0.0, 1.5)
_PYRAMID_BASE = [
    (-0.5, -0.5, 0.5),
    ( 0.5, -0.5, 0.5),
    ( 0.5,  0.5, 0.5),
    (-0.5,  0.5, 0.5),
]
_PYRAMID_SCALE = 3


def build_pyramid_vbo():
    tip  = _PYRAMID_TIP
    base = _PYRAMID_BASE
    verts = []
    for i in range(4):
        verts += [0.0, 0.0, 0.0,  tip[0],  tip[1],  tip[2]]
        verts += [0.0, 0.0, 1.0,  base[i][0],       base[i][1],       base[i][2]]
        verts += [0.0, 0.0, 1.0,  base[(i+1)%4][0], base[(i+1)%4][1], base[(i+1)%4][2]]
    for idx in [0, 1, 2, 0, 2, 3]:
        verts += [0.0, 0.0, 1.0,  base[idx][0], base[idx][1], base[idx][2]]
    data = np.array(verts, dtype=np.float32)
    vbo  = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vbo, len(verts) // 6


def draw_pyramid_vbo(vbo, vertex_count, tint):
    """Draw one pyramid (O(1), local-space geometry) with the given RGB tint."""
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glEnableClientState(GL_COLOR_ARRAY)
    glEnableClientState(GL_VERTEX_ARRAY)
    glColorPointer(3, GL_FLOAT, 6*4, ctypes.c_void_p(0))
    glVertexPointer(3, GL_FLOAT, 6*4, ctypes.c_void_p(3*4))
    glDisableClientState(GL_COLOR_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    tip  = _PYRAMID_TIP
    base = _PYRAMID_BASE
    r, g, b = tint
    glBegin(GL_TRIANGLES)
    for i in range(4):
        glColor3f(0, 0, 0);  glVertex3fv(tip)
        glColor3f(r, g, b);  glVertex3fv(base[i])
        glColor3f(r, g, b);  glVertex3fv(base[(i+1)%4])
    glEnd()
    glBegin(GL_QUADS)
    glColor3f(r, g, b)
    for bv in base:
        glVertex3fv(bv)
    glEnd()
    glDisableClientState(GL_VERTEX_ARRAY)


# ---------------------------------------------------------------------------
# Terrain VBO
# ---------------------------------------------------------------------------
def build_terrain_vbo(
        tri_path, height_path, image, margin, map_scale, blur_sigma,
        color_map=None):
    import image_to_tris
    image_to_tris.main(
        image_path=height_path,
        margin=margin,
        map_scale=map_scale,
        output_path=tri_path,
        blur_sigma=blur_sigma,
    )
    tris = list(tm.read_tri_map(tri_path))
    vertex_count = len(tris) * 3
    data = np.empty(vertex_count * 6, dtype=np.float32)
    idx = 0
    for tri in tris:
        for v in (tri.v1, tri.v2, tri.v3):
            if color_map is not None:
                try:
                    bgr = color_map[int(v.z) * margin, int(v.x) * margin]
                except IndexError:
                    bgr = (0, 0, 0)  # default to black if out of bounds
            else:
                bgr = image[int(v.z) * margin, int(v.x) * margin]
            data[idx]   = bgr[2] / 255.0
            data[idx+1] = bgr[1] / 255.0
            data[idx+2] = bgr[0] / 255.0
            data[idx+3] = v.x
            data[idx+4] = v.y
            data[idx+5] = v.z
            idx += 6
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vbo, vertex_count


def draw_terrain_vbo(vbo, vertex_count):
    stride = 6 * 4
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glEnableClientState(GL_COLOR_ARRAY)
    glEnableClientState(GL_VERTEX_ARRAY)
    glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))
    glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(3*4))
    glDrawArrays(GL_TRIANGLES, 0, vertex_count)
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_COLOR_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, 0)


# ---------------------------------------------------------------------------
# Resize / Init
# ---------------------------------------------------------------------------
def resize(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def init():
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)


# ---------------------------------------------------------------------------
# Background / decorative
# ---------------------------------------------------------------------------
def draw_gradient_background():
    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix(); glLoadIdentity(); glOrtho(-1, 1, -1, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix(); glLoadIdentity()
    glBegin(GL_QUADS)
    glColor3f(0.2, 0.2, 0.2); glVertex2f(-1,  1); glVertex2f( 1,  1)
    glColor3f(0.6, 0.6, 0.6); glVertex2f( 1, -1); glVertex2f(-1, -1)
    glEnd()
    glPopMatrix()
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)


def draw_seperator_line():
    width, height = pygame.display.get_surface().get_size()
    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix(); glLoadIdentity(); glOrtho(-1, 1, -1, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix(); glLoadIdentity()
    glViewport(0, 0, width, height)
    glBegin(GL_LINES)
    glColor3f(0, 0, 0)
    glVertex2f(0, -height); glVertex2f(0, height)
    glEnd()
    glPopMatrix()
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


# ---------------------------------------------------------------------------
# Screen-space text (PyGame font -> OpenGL texture, cached per string)
# ---------------------------------------------------------------------------
_text_font = None
_text_cache = {}   # key -> (tex_id, w, h)


def _get_text_font():
    global _text_font
    if _text_font is None:
        if not pygame.font.get_init():
            pygame.font.init()
        _text_font = pygame.font.SysFont("Consolas", 20)
    return _text_font


def _get_text_texture(text, color, outline, outline_w):
    """Build (and cache) an RGBA texture for a string with an optional outline."""
    key = (text, color, outline, outline_w)
    cached = _text_cache.get(key)
    if cached is not None:
        return cached

    font = _get_text_font()
    base = font.render(text, True, color)
    if outline_w <= 0:
        surf = base
    else:
        w, h = base.get_size()
        surf = pygame.Surface((w + 2 * outline_w, h + 2 * outline_w), pygame.SRCALPHA)
        outline_surf = font.render(text, True, outline)
        for dx in range(-outline_w, outline_w + 1):
            for dy in range(-outline_w, outline_w + 1):
                if dx or dy:
                    surf.blit(outline_surf, (outline_w + dx, outline_w + dy))
        surf.blit(base, (outline_w, outline_w))

    w, h = surf.get_size()
    data = pygame.image.tostring(surf, "RGBA", True)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data)
    glBindTexture(GL_TEXTURE_2D, 0)

    # Bound the cache so transient strings (changing every frame) can't leak
    # textures without limit.
    if len(_text_cache) > 64:
        old_tex, _, _ = _text_cache.pop(next(iter(_text_cache)))
        glDeleteTextures([old_tex])
    _text_cache[key] = (tex, w, h)
    return tex, w, h


def draw_text_2d(text, x, y, color=(255, 255, 255), outline=(0, 0, 0),
                 outline_w=2):
    """Draw a string at pixel (x, y) measured from the TOP-LEFT of the window."""
    tex, w, h = _get_text_texture(text, color, outline, outline_w)
    width, height = pygame.display.get_surface().get_size()

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, width, height, 0, -1, 1)          # pixel coords, y-down
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()

    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)
    glColor4f(1, 1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(x,     y)
    glTexCoord2f(1, 1); glVertex2f(x + w, y)
    glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
    glTexCoord2f(0, 0); glVertex2f(x,     y + h)
    glEnd()
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


# ---------------------------------------------------------------------------
# Camera / picking helpers
# ---------------------------------------------------------------------------
def gl_rotation_matrix(rx, ry, rz):
    """World->eye rotation for the render convention Ry(ry)*Rx(rx)*Rz(rz)."""
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def rotation_geodesic_error(R_est, rx, ry, rz):
    """Minimal angle (deg) between an estimated world->eye rotation R_est and
    the actual camera given by render Euler angles (rx, ry, rz).

    Representation-independent: immune to the equivalent-Euler-triplet
    ambiguity that a per-angle difference suffers from (e.g. when roll != 0).
    """
    R_act = gl_rotation_matrix(rx, ry, rz)
    c = (np.trace(np.asarray(R_est).T @ R_act) - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def render_euler_from_pnp_R(est_R):
    """Convert an OpenCV PnP world->cam rotation into render-convention Euler
    angles (rx, ry, rz) for Ry(ry)*Rx(rx)*Rz(rz).

    The OpenCV frame is related to OpenGL by D = diag(1,-1,-1) (y/z flipped),
    so the GL world->eye rotation is D*est_R. Extracting Euler angles from that
    yields the SAME convention as the actual camera, so a pyramid drawn with the
    standard Ry(-ry)*Rx(-rx)*Rz(-rz) recipe points the correct way (no flip).
    """
    R = np.diag([1.0, -1.0, -1.0]) @ np.asarray(est_R, dtype=np.float64)
    ry = math.atan2(R[0, 2], R[2, 2])
    rx = math.asin(max(-1.0, min(1.0, -R[1, 2])))
    rz = math.atan2(R[1, 0], R[1, 1])
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def gl_rotation_from_view_matrix(view_mat):
    """Extract the 3x3 world->eye rotation from a column-major GL modelview
    matrix (list of 16) or a 4x4 array."""
    M = np.asarray(view_mat, dtype=np.float64)
    if M.shape == (16,):
        M = M.reshape(4, 4).T          # column-major -> row-major
    elif M.shape == (4, 4):
        pass
    else:
        M = M.reshape(4, 4)
    return M[:3, :3]


def setup_right_view_matrices(width, height):
    glViewport(width // 2, 0, width // 2, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width / 2) / height, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glRotatef(r_y2, 0, 1, 0)
    glRotatef(r_x2, 1, 0, 0)
    glRotatef(r_z2, 0, 0, 1)
    glTranslatef(c_x2, c_y2, c_z2)


def setup_left_view_matrices(width, height):
    glViewport(0, 0, width // 2, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width / 2) / height, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glRotatef(r_y, 0, 1, 0)
    glRotatef(r_x, 1, 0, 0)
    glRotatef(r_z, 0, 0, 1)
    glTranslatef(c_x, c_y, c_z)


def get_world_coords(mouse_x, mouse_y, view="right"):
    width, height = pygame.display.get_surface().get_size()
    if view == "left":
        setup_left_view_matrices(width, height)
    else:
        setup_right_view_matrices(width, height)
    viewport    = glGetIntegerv(GL_VIEWPORT)
    real_y      = height - mouse_y
    depth       = glReadPixels(mouse_x, real_y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
    depth_value = depth[0][0]
    if depth_value >= 1.0:
        return None
    modelview  = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    world_x, world_y, world_z = gluUnProject(
        mouse_x, real_y, depth_value, modelview, projection, viewport)
    return world_x, world_y, world_z


# ---------------------------------------------------------------------------
# PnP camera estimation
# ---------------------------------------------------------------------------
def build_camera_intrinsics(view_w, view_h):
    """
    Reconstruct the OpenCV K matrix that matches
    gluPerspective(45, view_w/view_h, NEAR, FAR).
    """
    fov_y_rad = math.radians(45)
    fy = (view_h / 2.0) / math.tan(fov_y_rad / 2.0)
    fx = fy
    cx = view_w / 2.0
    cy = view_h / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


def _points_are_coplanar(pts3d, tol=0.05):
    """
    Return True if all 3D points lie on (approximately) the same plane.
    Uses PCA: if the smallest singular value is < tol × largest, they're coplanar.
    Terrain points are nearly always coplanar, so this will almost always be True.
    """
    if len(pts3d) < 3:
        return True
    centred = pts3d - pts3d.mean(axis=0)
    _, s, _ = np.linalg.svd(centred)
    return s[-1] < tol * s[0]


def _try_solvers(pts3d, pts2d, K, dist, solvers):
    """
    Try a list of (flag, label) solver pairs in order.
    Returns (rvec, tvec) from the first one that succeeds, or (None, None).
    """
    for flag, label in solvers:
        try:
            ok, rvec, tvec = cv2.solvePnP(pts3d, pts2d, K, dist, flags=flag)
            if ok:
                print(f"Solver '{label}' succeeded.")
                return rvec, tvec
        except cv2.error as e:
            print(f"Solver '{label}' failed: {e.msg.splitlines()[0]}")
    return None, None


def solve_pnp(picked_correspondences, view_w, view_h, actual_cam=None):
    """
    Estimate camera pose from N >= 4 2D-3D correspondences.
    Returns (success, cam_pos_world, (rx_deg, ry_deg, rz_deg), R, tvec)
    or      (False, None, None, None, None) on failure.

    Terrain points are nearly always coplanar (flat mesh), so SQPNP is avoided
    and EPNP / IPPE are preferred - they handle planar configurations correctly.

    4+ points: RANSAC(EPNP) then direct EPNP fallback, then LM refinement.
    """
    n = len(picked_correspondences)
    if not has_minimum_pnp_points(n):
        return False, None, None, None, None

    K    = build_camera_intrinsics(view_w, view_h)
    dist = np.zeros((4, 1))

    pts3d = np.array([[w[0], w[1], w[2]] for (_, w) in picked_correspondences],
                     dtype=np.float64)
    pts2d = np.array([[p[0], p[1]] for (p, _) in picked_correspondences],
                     dtype=np.float64)

    coplanar = _points_are_coplanar(pts3d)
    if coplanar:
        print("Points appear coplanar (expected for terrain) - using planar solvers.")

    # 4+ points: RANSAC for robustness
    ransac_flag = cv2.SOLVEPNP_EPNP   # EPNP is robust to planar configs
    try:
        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            pts3d, pts2d, K, dist, flags=ransac_flag)
    except cv2.error:
        ok = False
    if not ok:
        solvers = [(cv2.SOLVEPNP_EPNP, "EPNP"),
                   (cv2.SOLVEPNP_IPPE, "IPPE")]
        rvec, tvec = _try_solvers(pts3d, pts2d, K, dist, solvers)
    if rvec is None:
        print("All solvers failed.")
        return False, None, None, None, None
    # Optional LM refinement - skip silently if it fails
    try:
        cv2.solvePnPRefineLM(pts3d, pts2d, K, dist, rvec, tvec)
    except cv2.error:
        pass

    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - pts2d, axis=1)))
    print(f"Reprojection error: {err:.2f}px")

    R, _ = cv2.Rodrigues(rvec)
    cam_pos = (-R.T @ tvec).flatten()
    

    ry_rad = math.atan2( R[0, 2],  R[2, 2])
    rx_rad = math.asin(max(-1.0, min(1.0, -R[1, 2])))
    rz_rad = math.atan2( R[1, 0],  R[1, 1])
    euler  = (math.degrees(rx_rad), math.degrees(ry_rad), math.degrees(rz_rad))

    print(f"PnP pos={cam_pos}  euler(deg)={euler}")

    # If no actual camera is provided, default to the RIGHT camera.
    # This keeps Picking Mode behavior unchanged.
    if actual_cam is None:
        actual_cam = (c_x2, c_y2, c_z2, r_x2, r_y2, r_z2)

    ax, ay, az, arx, ary, arz = actual_cam

    print(f"actual cam pos={(ax, ay, az)}  euler(deg)={(arx, ary, arz)}")

    pos_error = 0
    pos_error += math.pow(cam_pos[0] + ax, 2)
    pos_error += math.pow(cam_pos[1] + ay, 2)
    pos_error += math.pow(cam_pos[2] + az, 2)
    pos_error = math.sqrt(pos_error)

    rot_error = rotation_geodesic_error(
        np.diag([1.0, -1.0, -1.0]) @ R, arx, ary, arz)

    print(f"PnP position error vs actual camera: {pos_error} units")
    print(f"PnP rotation error vs actual camera: {rot_error} degrees")
    return True, cam_pos, euler, R, tvec


def solve_pnp_trackers(tracker_2d_3d_pairs, view_w, view_h):
    """
    Estimate camera pose from tracker 2D-3D correspondences.

    Returns (cam_pos_array, euler_deg_tuple) on success, or None on failure.
    Never modifies any global state or display.
    """
    n = len(tracker_2d_3d_pairs)
    print(f"\n--- Tracker PnP ({n} point{'s' if n != 1 else ''}) ---")

    if n == 0:
        print("No tracker pairs – skipping PnP.")
        return None
    if not has_minimum_pnp_points(n):
        return None

    K    = build_camera_intrinsics(view_w, view_h)
    dist = np.zeros((4, 1))

    pts3d = np.array([[w[0], w[1], w[2]] for (_, w) in tracker_2d_3d_pairs],
                     dtype=np.float64)
    pts2d = np.array([[p[0], p[1]] for (p, _) in tracker_2d_3d_pairs],
                     dtype=np.float64)


    coplanar = _points_are_coplanar(pts3d)
    if coplanar:
        print("Points appear coplanar - using planar solvers.")

    rvec = None
    try:
        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            pts3d, pts2d, K, dist, flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        ok = False
    if not ok:
        rvec, tvec = _try_solvers(
            pts3d, pts2d, K, dist,
            [(cv2.SOLVEPNP_EPNP, "EPNP"), (cv2.SOLVEPNP_IPPE, "IPPE")])
    if rvec is not None:
        try:
            cv2.solvePnPRefineLM(pts3d, pts2d, K, dist, rvec, tvec)
        except cv2.error:
            pass

    if rvec is None:
        print("All solvers failed.")
        return None

    # Reprojection error
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    reproj_err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - pts2d, axis=1)))

    # Camera position and orientation in world space
    R, _    = cv2.Rodrigues(rvec)
    cam_pos = (-R.T @ tvec).flatten()
    cam_pos = [-c for c in cam_pos]  # negate to match OpenGL convention

    ry_rad = math.atan2( R[0, 2],  R[2, 2])
    rx_rad = math.asin(max(-1.0, min(1.0, -R[1, 2])))
    rz_rad = math.atan2( R[1, 0],  R[1, 1])
    euler  = (math.degrees(rx_rad), math.degrees(ry_rad), math.degrees(rz_rad))
    #euler = tuple(-angle for angle in euler)  # negate to match OpenGL convention

    # Compare against the actual LEFT camera
    pos_err  = math.sqrt(
        (cam_pos[0] - c_x) ** 2 +
        (cam_pos[1] - c_y) ** 2 +
        (cam_pos[2] - c_z) ** 2
    )
    
    rot_err  = rotation_geodesic_error(
        np.diag([1.0, -1.0, -1.0]) @ R, r_x, r_y, r_z)

    print(f"  Reprojection error : {reproj_err:.2f} px")
    print(f"  Estimated position : {cam_pos}")
    print(f"  Estimated euler    : {euler} deg")
    print(f"  Actual left cam pos: ({c_x:.2f}, {c_y:.2f}, {c_z:.2f})")
    print(f"  Actual left cam rot: ({r_x:.2f}, {r_y:.2f}, {r_z:.2f}) deg")
    print(f"  Position error     : {pos_err:.3f} units")
    print(f"  Rotation error     : {rot_err:.3f} deg")
    print("--- end tracker PnP ---\n")

    return cam_pos, euler, R, tvec


# ---------------------------------------------------------------------------
# Feature-matching pose estimation (SIFT detection + FLANN matching)
# ---------------------------------------------------------------------------
# Startup learns a database of SIFT descriptors, each tagged with the 3D world
# point it corresponds to, by rendering the terrain from several reference
# viewpoints and unprojecting every keypoint against that view's depth buffer.
# At estimation time the query screenshot is SIFT-matched (FLANN + Lowe ratio
# test) against this database to recover 2D-3D correspondences, which the
# existing solve_pnp() turns into a camera pose.
#
# This is entirely self-contained to feature mode and does not touch the
# trackers/picking estimation paths.

_FEATURE_DB = {
    "descriptors": None,   # (N, 128) float32 stacked SIFT descriptors
    "points3d":    None,   # (N, 3)   float32 world points, parallel to descriptors
    "sift":        None,   # cv2.SIFT detector instance
    "flann":       None,   # cv2.FlannBasedMatcher instance
    "ready":       False,
}


def _make_sift():
    """Create a SIFT detector, raising a clear error if unavailable."""
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "cv2.SIFT_create not available - install opencv-contrib-python "
            "or a recent opencv-python (>=4.4).")
    return cv2.SIFT_create()


def _make_flann():
    """FLANN matcher configured for SIFT's float descriptors (KD-tree)."""
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def _terrain_world_bounds():
    """
    Compute the terrain's world-space bounding box from the actual mesh data
    (the .tri vertices), returning (min_xyz, max_xyz, center) as numpy arrays.

    Vertices are produced by image_to_tris as (x/margin, height, z/margin), so
    this reflects the real extent of the world rather than image dimensions.
    """
    verts = tm.read_tri_map(ACTIVE_MAP["tri_path"])
    xs, ys, zs = [], [], []
    for t in verts:
        for v in (t.v1, t.v2, t.v3):
            xs.append(v.x); ys.append(v.y); zs.append(v.z)
    mn = np.array([min(xs), min(ys), min(zs)], dtype=np.float64)
    mx = np.array([max(xs), max(ys), max(zs)], dtype=np.float64)
    return mn, mx, (mn + mx) / 2.0


def _look_at_pose(cam_world, target_world):
    """
    Build a render pose that places the camera at `cam_world` looking at
    `target_world`, in the renderer's convention.

    The modelview is R·T(c) with R = Ry(yaw)·Rx(pitch)·Rz(0), so a world point
    maps to eye space as R·(X + c); the camera's world position is therefore
    -c. The world-space forward direction is f = (sin yaw, -sin pitch cos yaw,
    -cos pitch cos yaw), which inverts to:
        yaw   = asin(f.x)
        pitch = atan2(-f.y, -f.z)
    (verified exact for any look-at direction around the terrain).
    Returns (c_x, c_y, c_z, r_x, r_y, r_z).
    """
    cam_world = np.asarray(cam_world, dtype=np.float64)
    f = np.asarray(target_world, dtype=np.float64) - cam_world
    n = np.linalg.norm(f)
    if n < 1e-9:
        f = np.array([0.0, 0.0, -1.0])
    else:
        f = f / n
    fx, fy, fz = float(f[0]), float(f[1]), float(f[2])
    yaw   = math.degrees(math.asin(max(-1.0, min(1.0, fx))))
    pitch = math.degrees(math.atan2(-fy, -fz))
    # Camera world position is the negation of the translate term.
    c_x, c_y, c_z = -cam_world[0], -cam_world[1], -cam_world[2]
    return (float(c_x), float(c_y), float(c_z),
            float(pitch), float(yaw), 0.0)


def _reference_viewpoints():
    """
    Generate N RANDOM reference camera poses, each positioned around/above the
    terrain and aimed so it looks generally at the world.

    Uses the real terrain bounds (from the mesh data) to size the sampling
    region, so the views adapt to whatever world is loaded. Configurable via
    CONFIG keys (all optional):
        feature_ref_count        : number of random viewpoints (default 12)
        feature_ref_seed         : RNG seed; use null/None for a fresh random
                                   set each run (default 0 for reproducibility)
        feature_ref_height_scale : extra camera height as a fraction of the
                                   terrain's XZ span (default 0.6)
        feature_ref_dist_scale   : camera distance from centre as a fraction of
                                   the terrain's XZ span, [min, max]
                                   (default [0.6, 1.2])
    Returns a list of (c_x, c_y, c_z, r_x, r_y, r_z).
    """
    mn, mx, center = _terrain_world_bounds()
    span_xz = float(max(mx[0] - mn[0], mx[2] - mn[2]))
    top_y   = float(mx[1])

    n          = int(CONFIG.get("feature_ref_count", 12))
    seed       = CONFIG.get("feature_ref_seed", 0)   # None -> non-deterministic
    h_scale    = float(CONFIG.get("feature_ref_height_scale", 0.6))
    dist_lo, dist_hi = CONFIG.get("feature_ref_dist_scale", [0.6, 1.2])

    rng = random.Random(seed)
    poses = []
    for _ in range(n):
        # Random horizontal direction and distance around the terrain centre.
        ang  = rng.uniform(0.0, 2.0 * math.pi)
        dist = rng.uniform(dist_lo, dist_hi) * span_xz
        # Random height above the terrain top, scaled to the world size.
        height = top_y + rng.uniform(0.3, 1.0) * h_scale * span_xz

        cam_world = np.array([
            center[0] + math.cos(ang) * dist,
            height,
            center[2] + math.sin(ang) * dist,
        ], dtype=np.float64)

        # Aim at a random point near the terrain centre so views vary slightly
        # rather than all converging on the exact midpoint.
        jitter = 0.15 * span_xz
        target = np.array([
            center[0] + rng.uniform(-jitter, jitter),
            center[1] + rng.uniform(-jitter, jitter) * 0.3,
            center[2] + rng.uniform(-jitter, jitter),
        ], dtype=np.float64)

        poses.append(_look_at_pose(cam_world, target))
    return poses


def _render_reference_view(view_w, view_h, pose):
    """
    Render the terrain into the left viewport from `pose`, returning
    (bgr_image, modelview, projection, viewport).

    Uses the SAME canonical-axis camera convention as render_scene's left view
    so unprojected keypoints land on the terrain consistently.
    """
    cx, cy, cz, rx, ry, rz = pose

    glViewport(0, 0, view_w, view_h)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(45, view_w / view_h, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_gradient_background()

    glLoadIdentity()
    glRotatef(ry, 0, 1, 0)
    glRotatef(rx, 1, 0, 0)
    glRotatef(rz, 0, 0, 1)
    glTranslatef(cx, cy, cz)
    draw_terrain_vbo(terrain_vbo, terrain_vertex_count)

    modelview  = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport   = glGetIntegerv(GL_VIEWPORT)

    glReadBuffer(GL_BACK)
    pixels = glReadPixels(0, 0, view_w, view_h, GL_RGB, GL_UNSIGNED_BYTE)
    arr = np.frombuffer(pixels, dtype=np.uint8).reshape((view_h, view_w, 3))
    arr = np.flipud(arr)                       # OpenGL origin is bottom-left
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr, modelview, projection, viewport


def _unproject_keypoint(kp_x, kp_y, view_h, modelview, projection, viewport):
    """
    Unproject an image-space keypoint (origin top-left, as in cv2/pixels) to a
    3D world point using the given view's depth buffer. Returns None if the
    keypoint falls on the background (no terrain).
    """
    real_y = view_h - kp_y                     # flip to GL bottom-left origin
    depth  = glReadPixels(int(round(kp_x)), int(round(real_y)),
                          1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
    depth_value = float(depth[0][0])
    if depth_value >= 1.0:
        return None                            # background pixel
    wx, wy, wz = gluUnProject(kp_x, real_y, depth_value,
                              modelview, projection, viewport)
    return (wx, wy, wz)


def build_feature_database(view_w, view_h):
    """
    Startup stage: render the terrain from several reference viewpoints, detect
    SIFT features in each, unproject them to 3D, and assemble a descriptor +
    3D-point database with a FLANN matcher ready for queries.

    Renders to the BACK buffer and does NOT flip, so it leaves nothing visible;
    the normal draw loop repaints afterwards.
    """
    sift = _make_sift()
    descriptors_all = []
    points3d_all    = []

    poses = _reference_viewpoints()
    print(f"[features] learning from {len(poses)} reference viewpoint(s)...")

    for i, pose in tqdm.tqdm(enumerate(poses), desc="[features] rendering & detecting",
                             total=len(poses), unit="view"):
        bgr, modelview, projection, viewport = _render_reference_view(
            view_w, view_h, pose)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        kps, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(kps) == 0:
            continue
        kept = 0
        for kp, d in zip(kps, desc):
            p3d = _unproject_keypoint(kp.pt[0], kp.pt[1], view_h,
                                      modelview, projection, viewport)
            if p3d is None:
                continue                       # keypoint on background sky
            descriptors_all.append(d)
            points3d_all.append(p3d)
            kept += 1
        #print(f"[features]   view {i+1}/{len(poses)} "
        #      f"(yaw={pose[4]:.0f}, pitch={pose[3]:.0f}): "
        #      f"{len(kps)} keypoints, {kept} mapped to terrain")

    if not descriptors_all:
        print("[features] WARNING: no features learned - database empty.")
        _FEATURE_DB["ready"] = False
        return

    _FEATURE_DB["descriptors"] = np.asarray(descriptors_all, dtype=np.float32)
    _FEATURE_DB["points3d"]    = np.asarray(points3d_all,    dtype=np.float32)
    _FEATURE_DB["sift"]        = sift
    _FEATURE_DB["flann"]       = _make_flann()
    _FEATURE_DB["flann"].add([_FEATURE_DB["descriptors"]])
    _FEATURE_DB["flann"].train()
    _FEATURE_DB["ready"]       = True
    print(f"[features] database ready: {_FEATURE_DB['descriptors'].shape[0]} "
          f"descriptors mapped to 3D.")


def estimate_pose_from_features(query_bgr, view_w, view_h,
                                ratio=0.7, min_matches=6, actual_cam=None):
    """
    Solver stage: SIFT-detect on the query image, FLANN-match against the
    learned database with Lowe's ratio test, build 2D-3D correspondences, and
    run the existing solve_pnp.

    Returns (cam_pos, euler, R, tvec) on success, or None on failure (caller
    falls back to the placeholder).
    """
    if not _FEATURE_DB["ready"]:
        print("[features] database not ready - run startup learning first.")
        return None

    sift  = _FEATURE_DB["sift"]
    flann = _FEATURE_DB["flann"]
    pts3d = _FEATURE_DB["points3d"]

    gray = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
    kps_q, desc_q = sift.detectAndCompute(gray, None)
    if desc_q is None or len(kps_q) < 2:
        print("[features] too few keypoints in query image.")
        return None

    # k-NN match query descriptors against the database, then Lowe ratio test.
    knn = flann.knnMatch(desc_q, k=2)
    correspondences = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            q_pt = kps_q[m.queryIdx].pt          # 2D in query image (top-left origin)
            w_pt = pts3d[m.trainIdx]             # 3D world point from database
            correspondences.append(((float(q_pt[0]), float(q_pt[1])),
                                    (float(w_pt[0]), float(w_pt[1]), float(w_pt[2]))))

    print(f"[features] {len(kps_q)} query keypoints, "
          f"{len(correspondences)} good matches after ratio test.")
    if len(correspondences) < min_matches:
        print(f"[features] not enough matches "
              f"({len(correspondences)} < {min_matches}) - estimation aborted.")
        return None

    # solve_pnp expects 2D in the same pixel frame as the query (top-left origin,
    # view_w x view_h), which is exactly what the screenshot keypoints use.
    ok, cam_pos, euler, R, tvec = solve_pnp(correspondences,view_w,view_h,actual_cam=actual_cam)
    if not ok:
        print("[features] solve_pnp failed on feature correspondences.")
        return None
    return cam_pos, euler, R, tvec



def get_reprojected_world_points(picked_correspondences, pnp_result, view_w, view_h):
    """
    For each correspondence cast a ray from the estimated camera through the
    picked 2D pixel and intersect it with the horizontal plane Y = (original Y).
    This gives a world-space point that can be compared directly to the picked
    3D point - the line between them is the world-space reprojection error.
    """
    ok, cam_pos, euler_deg, R, tvec = pnp_result
    if R is None or cam_pos is None:
        return []

    K    = build_camera_intrinsics(view_w, view_h)
    dist = np.zeros((4, 1))
    K_inv = np.linalg.inv(K)
    cam   = np.array(cam_pos, dtype=np.float64)

    pts3d = np.array([[w[0], w[1], w[2]] for (_, w) in picked_correspondences],
                     dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)

    # Project the original 3D points through the estimated camera
    projected, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)

    reproj_world = []
    for i, ((img_x, img_y), (wx, wy, wz)) in enumerate(picked_correspondences):
        px, py = projected[i][0]
        # Ray from estimated camera through projected pixel, back into world
        ray_cam   = K_inv @ np.array([px, py, 1.0])
        ray_world = R.T @ ray_cam
        ray_world /= np.linalg.norm(ray_world)

        # Intersect with plane Y = wy so error is shown at terrain height
        if abs(ray_world[1]) > 1e-6:
            t = (wy - cam[1]) / ray_world[1]
            if t > 0:
                hit = cam + t * ray_world
                reproj_world.append((float(hit[0]), float(hit[1]), float(hit[2])))
                continue
        # Fallback: project along ray a distance matching the picked point
        dist_to_pt = float(np.linalg.norm(np.array([wx, wy, wz]) - cam))
        hit = cam + ray_world * dist_to_pt
        reproj_world.append((float(hit[0]), float(hit[1]), float(hit[2])))

    return reproj_world


# ---------------------------------------------------------------------------
# World-space PnP error visualisation (drawn in the right 3D view)
# ---------------------------------------------------------------------------
def draw_pnp_world_overlay(pnp_result, picked_correspondences):
    """
    Drawn on top of the right 3D viewport (depth-test disabled = always visible).

    Per correspondence:
      • Cyan sphere - original picked world point
      • Yellow line - world-space error vector to the estimated camera's ray hit
    """
    if pnp_result is None:
        return
    ok, cam_pos, euler_deg, R, tvec = pnp_result
    if cam_pos is None:
        return

    width, height = pygame.display.get_surface().get_size()
    view_w = width // 2
    view_h = height

    reproj = get_reprojected_world_points(
        picked_correspondences, pnp_result, view_w, view_h)

    # Restore right-view camera matrices so 3D world coords map correctly
    glViewport(width // 2, 0, view_w, view_h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, view_w / view_h, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glRotatef(r_y2, 0, 1, 0)
    glRotatef(r_x2, 1, 0, 0)
    glRotatef(r_z2, 0, 0, 1)
    glTranslatef(c_x2, c_y2, c_z2)

    glDisable(GL_DEPTH_TEST)
    glLineWidth(3.0)

    q = get_sphere_quadric()

    for i, (_, (wx, wy, wz)) in enumerate(picked_correspondences):
        # Cyan: original picked world point
        glPushMatrix()
        glTranslatef(wx, wy, wz)
        glColor3f(0.0, 1.0, 1.0)
        gluSphere(q, 0.25, 12, 6)
        glPopMatrix()

        if i < len(reproj):
            rx, ry, rz = reproj[i]

            # Yellow error line in world space
            glBegin(GL_LINES)
            glColor3f(1.0, 1.0, 0.0)
            glVertex3f(wx, wy, wz)
            glVertex3f(rx, ry, rz)
            glEnd()

    glLineWidth(1.0)
    glEnable(GL_DEPTH_TEST)


# ---------------------------------------------------------------------------
# Sphere / pyramid helpers
# ---------------------------------------------------------------------------
_sphere_quadric = None


def get_sphere_quadric():
    global _sphere_quadric
    if _sphere_quadric is None:
        _sphere_quadric = gluNewQuadric()
    return _sphere_quadric


def draw_sphere(x, y, z, color=(1, 0, 0), radius=0.3):
    glPushMatrix()
    glTranslatef(x, y, z)
    glColor3f(*color)
    gluSphere(get_sphere_quadric(), radius, 32, 4)
    glPopMatrix()


def draw_2d_pick_marker(point, color, size=7):
    x, y = point
    glColor3f(*color)
    glBegin(GL_LINES)
    glVertex2f(x - size, y)
    glVertex2f(x + size, y)
    glVertex2f(x, y - size)
    glVertex2f(x, y + size)
    glEnd()


def draw_2d_pick_dot(point, color=(1.0, 0.0, 0.0), radius=5):
    x, y = point
    glColor3f(*color)
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    for i in range(17):
        angle = 2.0 * math.pi * i / 16
        glVertex2f(x + math.cos(angle) * radius,
                   y + math.sin(angle) * radius)
    glEnd()


def draw_left_world_pick_markers(completed_points, pending_point):
    if not completed_points and pending_point is None:
        return
    width, height = pygame.display.get_surface().get_size()
    setup_left_view_matrices(width, height)
    viewport = glGetIntegerv(GL_VIEWPORT)
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)

    projected_completed = []
    for point in completed_points:
        sx, sy, sz = gluProject(
            point[0], point[1], point[2], modelview, projection, viewport)
        if 0.0 <= sz <= 1.0:
            projected_completed.append((sx, height - sy))

    projected_pending = None
    if pending_point is not None:
        sx, sy, sz = gluProject(
            pending_point[0], pending_point[1], pending_point[2],
            modelview, projection, viewport)
        if 0.0 <= sz <= 1.0:
            projected_pending = (sx, height - sy)

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, width, height, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()

    glDisable(GL_DEPTH_TEST)
    glLineWidth(2.0)
    for point in projected_completed:
        draw_2d_pick_marker(point, (0.0, 1.0, 0.0))
    if projected_pending is not None:
        draw_2d_pick_marker(projected_pending, (1.0, 1.0, 0.0), size=9)
    glLineWidth(1.0)
    glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def draw_right_image_points_2d(correspondences):
    if not correspondences:
        return
    width, height = pygame.display.get_surface().get_size()
    half_w = width // 2

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, width, height, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()

    glDisable(GL_DEPTH_TEST)
    glLineWidth(2.0)
    for (img_x, img_y), _ in correspondences:
        draw_2d_pick_dot((half_w + img_x, img_y))
    glLineWidth(1.0)
    glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def draw_tracker_sphere(x, y, z, r=0.0, g=1.0, b=0.2):
    """Draw a sphere at (x,y,z) with the given RGB colour and TRACKER_RADIUS."""
    glPushMatrix()
    glTranslatef(x, y, z)
    glColor3f(r, g, b)
    gluSphere(get_sphere_quadric(), TRACKER_RADIUS, 16, 8)
    glPopMatrix()


def draw_camera_pyramid(x, y, z, rx, ry, rz):
    global recording_index, recording_mode, saved_positions
    global pyramid_vbo, pyramid_vertex_count
    is_current = (not recording_mode and
                  saved_positions[recording_index] == (x, y, z, rx, ry, rz))
    tint = (1.0, 0.0, 0.0) if is_current else (0.0, 0.0, 1.0)
    glPushMatrix()
    glTranslatef(-x, -y, -z)
    glRotatef(-ry, 0, 1, 0)
    glRotatef(-rx, 1, 0, 0)
    glRotatef(-rz, 0, 0, 1)
    glScalef(_PYRAMID_SCALE, _PYRAMID_SCALE, _PYRAMID_SCALE)
    draw_pyramid_vbo(pyramid_vbo, pyramid_vertex_count, tint)
    glPopMatrix()


def draw_tracker_cam_pairs(pairs):
    """
    Draw blue pyramid for actual cam pos and green pyramid for estimated cam pos.
    Each entry in pairs: ((ax,ay,az,arx,ary,arz), (ex,ey,ez,erx,ery,erz))
    """
    global pyramid_vbo, pyramid_vertex_count
    s = _PYRAMID_SCALE * 1.5   # slightly larger than recording pyramids
    for actual, estimated in pairs:
        ax, ay, az, arx, ary, arz = actual
        # Blue pyramid — actual camera position
        glPushMatrix()
        glTranslatef(-ax, -ay, -az)
        glRotatef(-ary, 0, 1, 0)
        glRotatef(-arx, 1, 0, 0)
        glRotatef(-arz, 0, 0, 1)
        glScalef(s, s, s)
        draw_pyramid_vbo(pyramid_vbo, pyramid_vertex_count, (0.0, 0.3, 1.0))
        glPopMatrix()

        ex, ey, ez, erx, ery, erz = estimated
        # Green pyramid — PnP-estimated camera position
        glPushMatrix()
        glTranslatef(-ex, -ey, -ez)
        glRotatef(-ery, 0, 1, 0)
        glRotatef(-erx, 1, 0, 0)
        glRotatef(-erz, 0, 0, 1)
        glScalef(s, s, s)
        draw_pyramid_vbo(pyramid_vbo, pyramid_vertex_count, (0.0, 1.0, 0.2))
        glPopMatrix()


# ---------------------------------------------------------------------------
# Scene rendering
# ---------------------------------------------------------------------------
def render_scene(apply_input=True, recording_mode=True, trackers_mode=False,
                 feature_mode=False):
    global CONFIG, ACTIVE_MAP
    global c_x, c_y, c_z, r_x, r_y, r_z
    global c_x2, c_y2, c_z2, r_x2, r_y2, r_z2
    global picking_mode, picked_points, pending_left_world_point
    global terrain_vbo, terrain_vertex_count
    global tracker_points, tracker_cam_pairs
    global tracker_overlay_active, tracker_current_pair_index
    global feature_cam_pairs, feature_overlay_active, feature_current_pair_index

    r_speed   = 0.1
    rot_speed = 0.5

    dx =  math.sin(r_y * math.pi / 180)
    dz = -math.cos(r_y * math.pi / 180)
    rx =  math.cos(r_y * math.pi / 180)
    rz =  math.sin(r_y * math.pi / 180)

    if apply_input and (not picking_mode or trackers_mode or feature_mode):
        keys_pressed = pygame.key.get_pressed()
        if recording_mode or trackers_mode or feature_mode:
            moved = False
            yaw_delta = 0.0
            if keys_pressed[K_LEFT]:   yaw_delta -= rot_speed;  moved = True
            if keys_pressed[K_RIGHT]:  yaw_delta += rot_speed;  moved = True
            tilt = 0.0
            if keys_pressed[K_DOWN]:   tilt += rot_speed;  moved = True
            if keys_pressed[K_UP]:     tilt -= rot_speed;  moved = True

            if yaw_delta:
                # Preserve the current vertical tilt magnitude/direction while
                # yawing: r_x,r_z encode tilt T as (T*cos(yaw), T*sin(yaw)).
                # Recover T (signed by the old yaw) and re-project at new yaw.
                old_yaw = r_y * math.pi / 180
                tilt_mag = (r_x * math.cos(old_yaw) + r_z * math.sin(old_yaw))
                r_y += yaw_delta
                new_yaw = r_y * math.pi / 180
                r_x = tilt_mag * math.cos(new_yaw)
                r_z = tilt_mag * math.sin(new_yaw)

            if tilt:
                yaw_rad = r_y * math.pi / 180
                r_x += tilt * math.cos(yaw_rad)
                r_z += tilt * math.sin(yaw_rad)
            if keys_pressed[K_a]:      c_x += rx * r_speed;  c_z += rz * r_speed;  moved = True
            if keys_pressed[K_d]:      c_x -= rx * r_speed;  c_z -= rz * r_speed;  moved = True
            if keys_pressed[K_w]:      c_x -= dx * r_speed;  c_z -= dz * r_speed;  moved = True
            if keys_pressed[K_s]:      c_x += dx * r_speed;  c_z += dz * r_speed;  moved = True
            if keys_pressed[K_SPACE]:  c_y -= r_speed;  moved = True
            if keys_pressed[K_LSHIFT]: c_y += r_speed;  moved = True
            if moved and trackers_mode:
                tracker_overlay_active = False
            if moved and feature_mode:
                feature_overlay_active = False
            if keys_pressed[K_BACKSPACE]:
                img = cv2.imread(ACTIVE_MAP["height_path"])
                ih, iw, _ = img.shape
                m = ACTIVE_MAP["margin"]
                c_x, c_y, c_z = -iw/m/2, CONFIG.get("start_h"), -(ih/m)-100
                r_x, r_y, r_z = CONFIG.get("start_a"), 0.0, 0.0
                tracker_overlay_active = False
                feature_overlay_active = False
        else:
            global saved_positions, recording_index
            c_x, c_y, c_z, r_x, r_y, r_z = saved_positions[recording_index]

    elif not apply_input and not picking_mode:
        c_x, c_y, c_z = c_x2, c_y2, c_z2
        r_x, r_y, r_z = r_x2, r_y2, r_z2
        if trackers_mode and tracker_cam_pairs:
            # In trackers mode the right view shows actual(blue) & estimated(green) pyramids
            for pos in tracker_cam_pairs:
                glPushMatrix()
                glLoadIdentity()
                glRotatef(r_y2, 0, 1, 0); glRotatef(r_x2, 1, 0, 0)
                glRotatef(r_z2, 0, 0, 1); glTranslatef(c_x2, c_y2, c_z2)
                draw_tracker_cam_pairs([pos])
                glPopMatrix()
        elif feature_mode and feature_cam_pairs:
            # In feature mode the right view shows actual(blue) & estimated(green) pyramids
            for pos in feature_cam_pairs:
                glPushMatrix()
                glLoadIdentity()
                glRotatef(r_y2, 0, 1, 0); glRotatef(r_x2, 1, 0, 0)
                glRotatef(r_z2, 0, 0, 1); glTranslatef(c_x2, c_y2, c_z2)
                draw_tracker_cam_pairs([pos])
                glPopMatrix()
        elif not trackers_mode and not feature_mode:
            for pos in saved_positions:
                px, py, pz, prx, pry, prz = pos
                glPushMatrix()
                glLoadIdentity()
                glRotatef(r_y2, 0, 1, 0); glRotatef(r_x2, 1, 0, 0)
                glRotatef(r_z2, 0, 0, 1); glTranslatef(c_x2, c_y2, c_z2)
                draw_camera_pyramid(px, py, pz, prx, pry, prz)
                glPopMatrix()

    elif not apply_input and picking_mode:
        c_x, c_y, c_z = c_x2, c_y2, c_z2
        r_x, r_y, r_z = r_x2, r_y2, r_z2

    glLoadIdentity()
    glRotatef(r_y, 0, 1, 0)
    glRotatef(r_x, 1, 0, 0)
    glRotatef(r_z, 0, 0, 1)
    glTranslatef(c_x, c_y, c_z)
    draw_terrain_vbo(terrain_vbo, terrain_vertex_count)

    # Draw trackers after terrain, reusing the same world-space matrix so
    # their positions stay fixed relative to the terrain regardless of camera.
    if apply_input and trackers_mode:
        for t in tracker_points:
            px, py, pz = t[0], t[1], t[2]
            tr, tg, tb  = t[3] / 255.0, t[4] / 255.0, t[5] / 255.0
            draw_tracker_sphere(px, py, pz, tr, tg, tb)


# ---------------------------------------------------------------------------
# Main draw (split-screen)
# ---------------------------------------------------------------------------
def draw(recording_mode, trackers_mode=False, feature_mode=False):
    global picking_mode, pnp_result, picked_correspondences
    global c_x, c_y, c_z, r_x, r_y, r_z
    global c_x2, c_y2, c_z2, r_x2, r_y2, r_z2
    global tracker_cam_pairs, tracker_overlay_active, tracker_current_pair_index
    global tracker_est_view_mats
    global feature_cam_pairs, feature_overlay_active, feature_current_pair_index
    global feature_est_view_mats
    try:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    except Exception as e:
        global running
        running = False
        return
    width, height = pygame.display.get_surface().get_size()

    # LEFT VIEW
    glViewport(0, 0, width // 2, height)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(45, (width/2) / height, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW)
    draw_gradient_background()
    render_scene(apply_input=True, recording_mode=recording_mode,
                 trackers_mode=trackers_mode, feature_mode=feature_mode)
    if picking_mode:
        draw_left_world_pick_markers(picked_points, pending_left_world_point)

    # Tracker overlay on left view: atop the real-position camera view (drawn
    # above by render_scene), overlay the camera view from the estimated
    # position at 36% opacity.
    if (trackers_mode and tracker_overlay_active and tracker_cam_pairs
            and tracker_current_pair_index in tracker_est_view_mats):
        # Re-render the terrain from the PnP-estimated camera, blended over the
        # existing left view. We use the exact OpenGL view matrix built from the
        # PnP extrinsics [R|tvec] at save time, which avoids the Euler-angle
        # flip ambiguity that made an angles-based transform point the wrong way
        # on "good" estimates (PnP can report a 180°-flipped Euler branch).
        view_mat = tracker_est_view_mats[tracker_current_pair_index]

        glViewport(0, 0, width // 2, height)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(45, (width / 2) / height, NEAR, FAR)
        glMatrixMode(GL_MODELVIEW)

        # The base (real-position) terrain is already in the colour + depth
        # buffers; with depth testing on, the overlay fragments sit at the same
        # depths and get rejected. Disable depth testing and blend on top.
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        # Per-vertex map colours carry no alpha, so fade the whole overlay pass
        # with a constant blend factor (GL 1.4+, supported by the render context).
        glBlendColor(0.0, 0.0, 0.0, 0.36)
        glBlendFunc(GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA)

        # Load the precomputed estimated-camera view matrix (column-major).
        glLoadMatrixd(view_mat)

        # Draw the terrain in its natural map colours at 36% opacity.
        draw_terrain_vbo(terrain_vbo, terrain_vertex_count)

        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)

        # Show the saved pair's position/rotation error in the top-left corner.
        real_cam, est_cam = tracker_cam_pairs[tracker_current_pair_index]
        t_pos_err = math.sqrt((est_cam[0] - real_cam[0]) ** 2 +
                              (est_cam[1] - real_cam[1]) ** 2 +
                              (est_cam[2] - real_cam[2]) ** 2)
        t_rot_err = rotation_geodesic_error(
            gl_rotation_from_view_matrix(view_mat),
            real_cam[3], real_cam[4], real_cam[5])
        draw_text_2d(f"pos err: {t_pos_err:.2f} units", 12, 10,
                     color=(255, 230, 120))
        draw_text_2d(f"rot err: {t_rot_err:.2f} deg", 12, 34,
                     color=(255, 230, 120))

    # Feature-matching overlay on left view: identical treatment to the tracker
    # overlay, but driven by the feature-mode pairs/matrices.
    if (feature_mode and feature_overlay_active and feature_cam_pairs
            and feature_current_pair_index in feature_est_view_mats):
        view_mat = feature_est_view_mats[feature_current_pair_index]

        glViewport(0, 0, width // 2, height)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(45, (width / 2) / height, NEAR, FAR)
        glMatrixMode(GL_MODELVIEW)

        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        glBlendColor(0.0, 0.0, 0.0, 0.36)
        glBlendFunc(GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA)

        glLoadMatrixd(view_mat)
        draw_terrain_vbo(terrain_vbo, terrain_vertex_count)

        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)

        # Show the saved pair's position/rotation error in the top-left corner.
        real_cam, est_cam = feature_cam_pairs[feature_current_pair_index]
        f_pos_err = math.sqrt((est_cam[0] - real_cam[0]) ** 2 +
                              (est_cam[1] - real_cam[1]) ** 2 +
                              (est_cam[2] - real_cam[2]) ** 2)
        f_rot_err = rotation_geodesic_error(
            gl_rotation_from_view_matrix(view_mat),
            real_cam[3], real_cam[4], real_cam[5])
        draw_text_2d(f"pos err: {f_pos_err:.2f} units", 12, 34,
                     color=(255, 230, 120))
        draw_text_2d(f"rot err: {f_rot_err:.2f} deg", 12, 58,
                     color=(255, 230, 120))

    glViewport(width // 2, 0, width // 2, height)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(45, (width/2) / height, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW)
    draw_gradient_background()

    old_cam = (c_x, c_y, c_z, r_x, r_y, r_z)
    c_x, c_y, c_z = c_x2, c_y2, c_z2
    r_x, r_y, r_z = r_x2, r_y2, r_z2
    render_scene(apply_input=False, recording_mode=recording_mode,
                 trackers_mode=trackers_mode, feature_mode=feature_mode)
    c_x, c_y, c_z, r_x, r_y, r_z = old_cam

    # Picking-mode overlay on the RIGHT view: atop the right-position camera
    # view, overlay the terrain as seen from the PnP-estimated camera at 36%
    # opacity (same idea as the trackers overlay on the left view).
    if picking_mode and pnp_result is not None and pnp_result[0]:
        # pnp_result = (ok, cam_pos, euler, R, tvec). Build the OpenGL view
        # matrix straight from the PnP extrinsics [R|tvec] to avoid Euler-angle
        # flip ambiguity:
        #   x_cam = R·X + tvec        (OpenCV: +Z forward, y down)
        #   M_gl  = diag(1,-1,-1)·[R|tvec]   (OpenGL: -Z forward, y up)
        _, _, _, est_R, est_tvec = pnp_result
        ext = np.eye(4, dtype=np.float64)
        ext[:3, :3] = est_R
        ext[:3, 3]  = np.asarray(est_tvec, dtype=np.float64).flatten()
        m_gl = np.diag([1.0, -1.0, -1.0, 1.0]) @ ext
        view_mat = m_gl.T.flatten().tolist()   # column-major for OpenGL

        glViewport(width // 2, 0, width // 2, height)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(45, (width / 2) / height, NEAR, FAR)
        glMatrixMode(GL_MODELVIEW)

        # The base (right-position) terrain is already in the colour + depth
        # buffers; with depth testing on, the overlay fragments sit at the same
        # depths and get rejected. Disable depth testing and blend on top.
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glEnable(GL_BLEND)
        # Per-vertex map colours carry no alpha, so fade the whole overlay pass
        # with a constant blend factor (GL 1.4+, supported by the render context).
        glBlendColor(0.0, 0.0, 0.0, 0.36)
        glBlendFunc(GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA)

        glLoadMatrixd(view_mat)
        draw_terrain_vbo(terrain_vbo, terrain_vertex_count)

        glDisable(GL_BLEND)
        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)

        # Show the PnP position/rotation error vs the actual (right) camera in
        # the top-left corner of the RIGHT viewport. Mirrors solve_pnp exactly:
        # the estimate is stored negated, so error uses (est + cam2).
        _, est_pos, est_euler, _, _ = pnp_result
        p_pos_err = math.sqrt((est_pos[0] + c_x2) ** 2 +
                              (est_pos[1] + c_y2) ** 2 +
                              (est_pos[2] + c_z2) ** 2)
        p_rot_err = rotation_geodesic_error(m_gl[:3, :3], r_x2, r_y2, r_z2)
        draw_text_2d(f"pos err: {p_pos_err:.2f} units", width // 2 + 12, 10,
                     color=(255, 230, 120))
        draw_text_2d(f"rot err: {p_rot_err:.2f} deg", width // 2 + 12, 34,
                     color=(255, 230, 120))
    if picking_mode and pnp_result is not None:
        draw_pnp_world_overlay(pnp_result, picked_correspondences)

    if picking_mode:
        draw_right_image_points_2d(picked_correspondences)

    if picking_mode:
        draw_text_2d("PICKING MODE", 12, height - 28, color=(255, 100, 100))
    elif trackers_mode:
        draw_text_2d("TRACKERS MODE", 12, height - 28, color=(200, 200, 200))
    elif recording_mode:
        
        draw_text_2d("RECORDING MODE", 12, height - 28, color=(100, 100, 255))
    elif feature_mode:
        draw_text_2d("2D FEATURE MATCHING MODE", 12, height - 28, color=(255, 255, 100))
    else:
        draw_text_2d("VIEWING MODE", 12, height - 28, color=(100, 255, 100))
    draw_seperator_line()
    pygame.display.flip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    global CONFIG, ACTIVE_MAP
    global c_x, c_y, c_z, r_x, r_y, r_z
    global c_x2, c_y2, c_z2, r_x2, r_y2, r_z2
    global saved_positions, recording_index, recording_mode
    global picking_mode, picked_points, picked_correspondences
    global pending_left_world_point
    global terrain_vbo, terrain_vertex_count
    global pyramid_vbo, pyramid_vertex_count
    global pnp_result
    global trackers_mode, tracker_points, tracker_cam_pairs
    global tracker_est_view_mats
    global tracker_overlay_active, tracker_current_pair_index
    global feature_mode, feature_cam_pairs, feature_est_view_mats
    global feature_overlay_active, feature_current_pair_index
    global running
    CONFIG = read_config()
    ACTIVE_MAP = select_map_profile()

    color_summary = ACTIVE_MAP["color_path"] or "height-map colors"
    print(
        "Selected map: "
        f"{ACTIVE_MAP['name']} | "
        f"height={ACTIVE_MAP['height_path']} | "
        f"color={color_summary} | "
        f"margin={ACTIVE_MAP['margin']} | "
        f"map_scale={ACTIVE_MAP['map_scale']} | "
        f"blur_sigma={ACTIVE_MAP['blur_sigma']} | "
        f"tri={ACTIVE_MAP['tri_path']}"
    )
    
    pygame.init()

    picking_mode           = False
    saved_positions        = []
    picked_points          = []
    picked_correspondences = []
    pending_left_world_point = None
    pnp_result             = None
    recording_mode         = True
    recording_index        = 0
    trackers_mode          = False
    tracker_points         = []
    tracker_cam_pairs           = []    # [(actual_6tuple, estimated_6tuple), ...]
    tracker_est_view_mats       = {}    # pair_index -> 16-float column-major GL view matrix
    tracker_overlay_active      = False # True when N/M loaded a pair onto left view
    tracker_current_pair_index  = 0     # which pair is shown
    feature_mode                = False
    feature_cam_pairs           = []    # [(actual_6tuple, estimated_6tuple), ...]
    feature_est_view_mats       = {}    # pair_index -> 16-float column-major GL view matrix
    feature_overlay_active      = False # True when a pair is loaded onto left view
    feature_current_pair_index  = 0     # which pair is shown
    trackers_path = CONFIG.get("trackers_path", "trackers.txt")
    try:
        tracker_points = get_trackers_from_file(trackers_path)
        print(f"Loaded {len(tracker_points)} tracker(s) from '{trackers_path}'")
    except FileNotFoundError:
        print(f"Trackers file '{trackers_path}' not found – starting with no trackers")

    image = cv2.imread(ACTIVE_MAP["height_path"])
    if image is None:
        print(f"Error: Failed to load map image from '{ACTIVE_MAP['height_path']}'")
        sys.exit(1)
    h, w, _ = image.shape
    margin = ACTIVE_MAP["margin"]
    starting_pos = (-w/margin/2, float(CONFIG.get("start_h")), -(h/margin)-100)
    saved_positions.append((*starting_pos, 10, 0.0, 0.0))
    c_x,  c_y,  c_z  = map(float, starting_pos)
    c_x2, c_y2, c_z2 = map(float, starting_pos)
    r_x = r_y = r_z = 0.0
    r_x2 = r_y2 = r_z2 = 0.0
    r_x = r_x2 = CONFIG.get("start_a")

    display = (640*2, 480)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.event.pump()
    focus_pygame_window()
    pygame.display.set_caption("NOW LOADING...")
    icon = pygame.image.load("icon.png")
    pygame.display.set_icon(icon)
    resize(*display)
    init()

    color_map_path = ACTIVE_MAP["color_path"]
    col = cv2.imread(color_map_path) if color_map_path else None
    if col is not None and col.shape[:2] != image.shape[:2]:
        print(
            f"Warning: resizing color map '{color_map_path}' from "
            f"{col.shape[1]}x{col.shape[0]} to {w}x{h}."
        )
        col = cv2.resize(col, (w, h), interpolation=cv2.INTER_LINEAR)
    terrain_vbo, terrain_vertex_count = build_terrain_vbo(
        ACTIVE_MAP["tri_path"],
        ACTIVE_MAP["height_path"],
        image,
        margin,
        ACTIVE_MAP["map_scale"],
        ACTIVE_MAP["blur_sigma"],
        col,
    )
    pyramid_vbo, pyramid_vertex_count = build_pyramid_vbo()
    print(f"Terrain VBO built: {terrain_vertex_count} vertices")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                if event.key == K_F11:
                    pygame.quit()
                    running = False
                    main()
                if event.key == K_b:
                    if trackers_mode:
                        # --- Screenshot the left half ---
                        sw, sh = pygame.display.get_surface().get_size()
                        half_w = sw // 2
                        glReadBuffer(GL_FRONT)
                        pixels = glReadPixels(0, 0, half_w, sh,
                                              GL_RGB, GL_UNSIGNED_BYTE)
                        img_array = np.frombuffer(pixels, dtype=np.uint8)
                        img_array = img_array.reshape((sh, half_w, 3))
                        img_array = np.flipud(img_array)   # OpenGL is bottom-left
                        img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        screenshot_path = "screenshot.png"
                        cv2.imwrite(screenshot_path, img_bgr)
                        print("Screenshot saved to screenshot.png")

                        # --- Find 2-D blob centres for each tracker colour ---
                        colors_rgb = [
                            (int(t[3]), int(t[4]), int(t[5]))
                            for t in tracker_points
                        ]
                        blob_map = find_blob_centers(
                            screenshot_path,
                            colors=colors_rgb,
                        )
                        # os.remove(screenshot_path) clean up the temporary screenshot
                        # --- Build fresh 2D-3D pairs for this press only ---
                        tracker_2d_3d_pairs = []
                        for t in tracker_points:
                            color_key = (int(t[3]), int(t[4]), int(t[5]))
                            world_pos = (t[0], t[1], t[2])
                            blobs     = blob_map.get(color_key, [])
                            if not blobs:
                                continue
                            # Use the single centroid; if multiple blobs share
                            # the same colour take the one closest to image centre
                            if len(blobs) == 1:
                                row, col = blobs[0]
                            else:
                                img_cy, img_cx = sh / 2.0, half_w / 2.0
                                row, col = min(
                                    blobs,
                                    key=lambda rc: (rc[0]-img_cy)**2 + (rc[1]-img_cx)**2
                                )
                            screen_pos = (col, row)   # (x, y) pixel convention
                            tracker_2d_3d_pairs.append((screen_pos, world_pos))

                        print(f"Tracker 2D-3D pairs ({len(tracker_2d_3d_pairs)}):")
                        for pair in tracker_2d_3d_pairs:
                            print(f"  2D {pair[0]}  ->  3D {pair[1]}")

                        # PnP estimation from tracker correspondences (print only)
                        pnp_tracker_result = solve_pnp_trackers(tracker_2d_3d_pairs, half_w, sh)
                        if pnp_tracker_result is not None:
                            est_cam_pos, est_euler, est_R, est_tvec = pnp_tracker_result
                            actual_tuple    = (c_x, c_y, c_z, r_x, r_y, r_z)
                            # Pyramid uses render-convention angles; convert the
                            # PnP rotation so the estimate isn't drawn flipped.
                            pyr_euler = (render_euler_from_pnp_R(est_R)
                                         if est_R is not None else est_euler)
                            estimated_tuple = (float(est_cam_pos[0]), float(est_cam_pos[1]),
                                               float(est_cam_pos[2]), float(pyr_euler[0]),
                                               float(pyr_euler[1]), float(pyr_euler[2]))
                            tracker_cam_pairs.append((actual_tuple, estimated_tuple))
                            tracker_current_pair_index = len(tracker_cam_pairs) - 1
                            # Store the unambiguous OpenGL view matrix for this
                            # estimate, built directly from the PnP extrinsics
                            # [R|tvec] (avoids all Euler-angle flip ambiguity).
                            #   x_cam = R·X + tvec     (OpenCV: +Z forward, y down)
                            #   M_gl  = diag(1,-1,-1)·[R|tvec]   (OpenGL: -Z forward, y up)
                            ext = np.eye(4, dtype=np.float64)
                            ext[:3, :3] = est_R
                            ext[:3, 3]  = np.asarray(est_tvec, dtype=np.float64).flatten()
                            flip = np.diag([1.0, -1.0, -1.0, 1.0])
                            m_gl = flip @ ext
                            # OpenGL wants column-major order; transpose to flat list.
                            tracker_est_view_mats[tracker_current_pair_index] = \
                                m_gl.T.flatten().astype(np.float64).tolist()
                            print(f"Tracker cam pair saved (total: {len(tracker_cam_pairs)})")
                            tracker_overlay_active = True
                    elif feature_mode:
                        # --- Screenshot the left half (same capture as trackers) ---
                        sw, sh = pygame.display.get_surface().get_size()
                        half_w = sw // 2
                        glReadBuffer(GL_FRONT)
                        pixels = glReadPixels(0, 0, half_w, sh,
                                              GL_RGB, GL_UNSIGNED_BYTE)
                        img_array = np.frombuffer(pixels, dtype=np.uint8)
                        img_array = img_array.reshape((sh, half_w, 3))
                        img_array = np.flipud(img_array)   # OpenGL is bottom-left
                        img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        #cv2.imwrite("screenshot_feature.png", img_bgr)
                        #print("Feature screenshot saved to screenshot_feature.png")

                        # Feature-matching pose estimation: match the query
                        # screenshot against the SIFT/FLANN database learned at
                        # startup, recover 2D-3D correspondences, run solve_pnp.
                        # Returns None on failure -> placeholder fallback below.
                        feat_result = estimate_pose_from_features(
                            img_bgr, half_w, sh, actual_cam=(c_x, c_y, c_z, r_x, r_y, r_z))
                        if feat_result is not None:
                            est_cam_pos, est_euler, est_R, est_tvec = feat_result
                            # solve_pnp returns cam_pos as the TRUE world
                            # position, but actual_tuple/the error display use
                            # the render convention (negated world position).
                            # Negate so the stored estimate matches actual_tuple's
                            # convention and the on-screen pos error agrees with
                            # the console (which uses cam_pos + c).
                            est_cam_pos = (-est_cam_pos[0], -est_cam_pos[1],
                                           -est_cam_pos[2])
                            # Pyramid uses render-convention angles; convert the
                            # PnP rotation so the estimate isn't drawn flipped.
                            if est_R is not None:
                                est_euler = render_euler_from_pnp_R(est_R)
                        else:
                            # Fallback: identity estimate = actual left camera.
                            # Convention (matches solve_pnp_trackers storage):
                            #   position stored same-sign as actual  -> pos err 0
                            #   euler stored negated                 -> rot err 0
                            est_cam_pos = (c_x, c_y, c_z)
                            est_euler   = (-r_x, -r_y, -r_z)
                            est_R, est_tvec = None, None

                        actual_tuple    = (c_x, c_y, c_z, r_x, r_y, r_z)
                        estimated_tuple = (float(est_cam_pos[0]), float(est_cam_pos[1]),
                                           float(est_cam_pos[2]), float(est_euler[0]),
                                           float(est_euler[1]), float(est_euler[2]))
                        feature_cam_pairs.append((actual_tuple, estimated_tuple))
                        feature_current_pair_index = len(feature_cam_pairs) - 1

                        if est_R is not None and est_tvec is not None:
                            ext = np.eye(4, dtype=np.float64)
                            ext[:3, :3] = est_R
                            ext[:3, 3]  = np.asarray(est_tvec, dtype=np.float64).flatten()
                            flip = np.diag([1.0, -1.0, -1.0, 1.0])
                            m_gl = flip @ ext
                            view_mat = m_gl.T.flatten().astype(np.float64).tolist()
                        else:
                            # Placeholder: reproduce the actual left-view matrix
                            # so the overlay coincides exactly with the real view.
                            # Built in numpy with the SAME (canonical-axis)
                            # composition render_scene now uses, stored
                            # column-major for glLoadMatrixd.
                            def _rot(angle, ax, ay, az):
                                a = math.radians(angle)
                                c, s = math.cos(a), math.sin(a)
                                n = math.sqrt(ax*ax + ay*ay + az*az)
                                if n == 0:
                                    return np.eye(4)
                                ax, ay, az = ax/n, ay/n, az/n
                                return np.array([
                                    [ax*ax*(1-c)+c,    ax*ay*(1-c)-az*s, ax*az*(1-c)+ay*s, 0],
                                    [ay*ax*(1-c)+az*s, ay*ay*(1-c)+c,    ay*az*(1-c)-ax*s, 0],
                                    [az*ax*(1-c)-ay*s, az*ay*(1-c)+ax*s, az*az*(1-c)+c,    0],
                                    [0, 0, 0, 1]], dtype=np.float64)
                            def _trans(x, y, z):
                                m = np.eye(4, dtype=np.float64)
                                m[0, 3], m[1, 3], m[2, 3] = x, y, z
                                return m
                            M = (_rot(r_y, 0, 1, 0) @
                                 _rot(r_x, 1, 0, 0) @
                                 _rot(r_z, 0, 0, 1) @
                                 _trans(c_x, c_y, c_z))
                            view_mat = M.T.flatten().astype(np.float64).tolist()
                        feature_est_view_mats[feature_current_pair_index] = view_mat
                        print(f"Feature cam pair saved (total: {len(feature_cam_pairs)})")
                        feature_overlay_active = True
                    else:
                        print(f"Saved ({c_x},{c_y},{c_z}) rot ({r_x},{r_y},{r_z})")
                        saved_positions.append((c_x, c_y, c_z, r_x, r_y, r_z))
                if event.key == K_t:
                    trackers_mode = not trackers_mode
                    if trackers_mode:
                        if picking_mode:
                            clear_picking_state()
                        picking_mode = False  # disable picking mode when entering trackers mode
                        pygame.display.set_caption("World Split v3.5 - TRACKERS MODE")
                        # Disable other exclusive modes when entering trackers mode
                        recording_mode = False
                        feature_mode = False
                        feature_overlay_active = False
                        pnp_result   = None
                        tracker_overlay_active = False
                        # Snap right view to the first saved position (overview)
                        if saved_positions:
                            c_x2, c_y2, c_z2, r_x2, r_y2, r_z2 = saved_positions[0]
                            print("Trackers mode: right view snapped to starting position")
                    else:
                        if not picking_mode:
                            recording_mode = True
                    print(f"Trackers mode: {'on' if trackers_mode else 'off'}")
                if event.key == K_f:
                    feature_mode = not feature_mode
                    if feature_mode:
                        # Exclusive with the other modes.
                        if picking_mode:
                            clear_picking_state()
                        picking_mode = False
                        trackers_mode = False
                        recording_mode = False
                        tracker_overlay_active = False
                        pnp_result   = None
                        feature_overlay_active = False
                        pygame.display.set_caption("World Split v3.5 - FEATURE MATCHING MODE")
                        # Snap right view to the first saved position (overview)
                        if saved_positions:
                            c_x2, c_y2, c_z2, r_x2, r_y2, r_z2 = saved_positions[0]
                            print("Feature matching mode: right view snapped to starting position")
                    else:
                        if not picking_mode:
                            recording_mode = True
                    print(f"Feature matching mode: {'on' if feature_mode else 'off'}")
                if event.key == K_n and trackers_mode and tracker_cam_pairs:
                    tracker_current_pair_index = (tracker_current_pair_index - 1) % len(tracker_cam_pairs)
                    actual, _ = tracker_cam_pairs[tracker_current_pair_index]
                    c_x, c_y, c_z, r_x, r_y, r_z = actual
                    tracker_overlay_active = True
                    print(f"Tracker pair {tracker_current_pair_index + 1}/{len(tracker_cam_pairs)} : actual: {actual[:3]} estimated: {tracker_cam_pairs[tracker_current_pair_index][1][:3]}")
                if event.key == K_m and trackers_mode and tracker_cam_pairs:
                    tracker_current_pair_index = (tracker_current_pair_index + 1) % len(tracker_cam_pairs)
                    actual, _ = tracker_cam_pairs[tracker_current_pair_index]
                    c_x, c_y, c_z, r_x, r_y, r_z = actual
                    tracker_overlay_active = True
                    print(f"Tracker pair {tracker_current_pair_index + 1}/{len(tracker_cam_pairs)} : actual: {actual[:3]} estimated: {tracker_cam_pairs[tracker_current_pair_index][1][:3]}")
                if event.key == K_n and feature_mode and feature_cam_pairs:
                    feature_current_pair_index = (feature_current_pair_index - 1) % len(feature_cam_pairs)
                    actual, _ = feature_cam_pairs[feature_current_pair_index]
                    c_x, c_y, c_z, r_x, r_y, r_z = actual
                    feature_overlay_active = True
                    print(f"Feature pair {feature_current_pair_index + 1}/{len(feature_cam_pairs)} : actual: {actual[:3]} estimated: {feature_cam_pairs[feature_current_pair_index][1][:3]}")
                if event.key == K_m and feature_mode and feature_cam_pairs:
                    feature_current_pair_index = (feature_current_pair_index + 1) % len(feature_cam_pairs)
                    actual, _ = feature_cam_pairs[feature_current_pair_index]
                    c_x, c_y, c_z, r_x, r_y, r_z = actual
                    feature_overlay_active = True
                    print(f"Feature pair {feature_current_pair_index + 1}/{len(feature_cam_pairs)} : actual: {actual[:3]} estimated: {feature_cam_pairs[feature_current_pair_index][1][:3]}")
                if event.key == K_r:
                    recording_index = 0
                    if saved_positions:
                        recording_mode = not recording_mode
                    print(f"Recording mode: {recording_mode}")
                    if picking_mode:
                        clear_picking_state()
                    picking_mode = False  # disable picking mode when toggling recording mode
                    trackers_mode = False  # disable trackers mode when toggling recording mode
                    feature_mode = False   # disable feature mode when toggling recording mode
                    feature_overlay_active = False
                    if recording_mode:
                        
                        pygame.display.set_caption("World Split v3.5 - RECORDING MODE")
                        #print("Recording mode: right view shows saved positions")
                    else:
                        pygame.display.set_caption("World Split v3.5 - NAVIGATION MODE")
                        #print("Navigation mode: right view mirrors left view")
                if event.key == K_p:
                    picking_mode = not picking_mode
                    if picking_mode:
                        recording_mode = False  # disable recording mode when entering picking mode
                        trackers_mode = False  # disable trackers mode when entering picking mode
                        feature_mode = False   # disable feature mode when entering picking mode
                        feature_overlay_active = False
                        pygame.display.set_caption("World Split v3.5 - PICKING MODE")
                        # Snap right view to last saved position (or starting pos if none)
                        if saved_positions:
                            c_x2, c_y2, c_z2, r_x2, r_y2, r_z2 = saved_positions[-1]
                            print(f"Picking mode: right view set to last saved pos ({c_x2:.1f},{c_y2:.1f},{c_z2:.1f})")
                        else:
                            image = cv2.imread(ACTIVE_MAP["height_path"])
                            ih, iw, _ = image.shape
                            m = ACTIVE_MAP["margin"]
                            c_x2, c_y2, c_z2 = -iw/m/2, CONFIG.get("start_h"), -(ih/m)-100
                            r_x2, r_y2, r_z2 = CONFIG.get("start_a"), 0.0, 0.0
                            print("Picking mode: no saved positions, right view at starting pos")
                    else:
                        if trackers_mode == False and feature_mode == False:
                            recording_mode = True
                        clear_picking_state()
                        print("Picking mode: cleared picked points and PnP result")
                    print("picking mode", "on" if picking_mode else "off")
                if event.key == K_c and picking_mode:
                    sw, sh = pygame.display.get_surface().get_size()
                    candidate_pnp_result = solve_pnp(picked_correspondences, sw // 2, sh)
                    if candidate_pnp_result and candidate_pnp_result[0]:
                        pnp_result = candidate_pnp_result
                        print("PnP solved (ok) - overlay active")
                    else:
                        pnp_result = None
                if event.key == K_LEFT and not recording_mode:
                    recording_index = (recording_index - 1) % len(saved_positions)
                if event.key == K_RIGHT and not recording_mode:
                    recording_index = (recording_index + 1) % len(saved_positions)

            if event.type == VIDEORESIZE:
                resize(event.w, event.h)

            if event.type == MOUSEBUTTONDOWN:
                if picking_mode and event.button == 1:
                    sw, sh = pygame.display.get_surface().get_size()
                    half_w = sw // 2
                    if event.pos[0] < half_w:
                        world_point = get_world_coords(event.pos[0], event.pos[1],
                                                       view="left")
                        if world_point is not None:
                            pending_left_world_point = world_point
                            print(f"Pending 3D point selected on left: {world_point}")
                        else:
                            print("Left picking missed terrain")
                    else:
                        if pending_left_world_point is None:
                            print("Select a 3D point on the left before clicking the matching point on the right.")
                            continue
                        image_point = (event.pos[0] - half_w, event.pos[1])
                        world_point = pending_left_world_point
                        picked_points.append(world_point)
                        picked_correspondences.append((image_point, world_point))
                        pending_left_world_point = None
                        pnp_result = None
                        print(f"Picked 3D {world_point} -> 2D {image_point}")
        if trackers_mode:
            pair_info = f" | Pair {tracker_current_pair_index+1}/{len(tracker_cam_pairs)}" if tracker_cam_pairs else ""
            pygame.display.set_caption(f"Trackers mode | B - estimate | N/M - prev/next pair{pair_info} | P - picking | R - recording")
        elif feature_mode:
            pair_info = f" | Pair {feature_current_pair_index+1}/{len(feature_cam_pairs)}" if feature_cam_pairs else ""
            pygame.display.set_caption(f"2D Feature matching mode | B - estimate | N/M - prev/next pair{pair_info} | P - picking | R - recording | T - trackers")
        elif picking_mode:
            pygame.display.set_caption("Picking mode | Click to pick points | C - solve PnP | R - toggle recording mode | T - toggle trackers mode")
        elif recording_mode:
            pygame.display.set_caption("Recording mode | Arrow keys to change index | R - toggle recording mode | T - toggle trackers mode | P - enter picking mode | B - record position")
        else:
            pygame.display.set_caption("Navigation mode | R - toggle recording mode | T - toggle trackers mode | P - enter picking mode")
        draw(recording_mode, trackers_mode, feature_mode)

    glDeleteBuffers(1, [terrain_vbo])
    glDeleteBuffers(1, [pyramid_vbo])
    if _sphere_quadric is not None:
        gluDeleteQuadric(_sphere_quadric)
    pygame.quit()


if __name__ == "__main__":
    main()
