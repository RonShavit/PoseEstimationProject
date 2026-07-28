import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import trimap_beta as tm
import math
import cv2
import numpy as np
import ctypes
import argparse
import time
from read_config import read_config
from trackers import get_trackers_from_file
from color_picking import find_blob_centers
from feature_database import (
    FeatureDatabase,
    active_database_path,
    load_active_database,
)
from feature_matching import detect_sift_features, match_query_to_database
import sys
import os


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
        "map_5":{
        "height_path": os.path.join("maps", "map_5.png"),
        "color_path": os.path.join("colors", "col_5.png"),
        "margin": 1,
        "map_scale": 20.0,
        "blur_sigma": 0.0,
    }
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
# Feature Pre / Feature Run helpers
# ---------------------------------------------------------------------------

FEATURE_PRE_LIGHTING = "pre"
FEATURE_RUN_LIGHTING = "run"
FEATURE_PRE_MAX_CANDIDATES = 200
FEATURE_RUN_MIN_RANSAC_INLIERS = 6
FEATURE_RUN_MIN_RANSAC_INLIER_RATIO = 0.60
FEATURE_RUN_MAX_INLIER_REPROJECTION_ERROR = 4.0
FEATURE_PRE_RESET_DT_SECONDS = 1.5


def apply_feature_lighting_bgr(bgr, profile):
    img = bgr.astype(np.float32)
    if profile == FEATURE_RUN_LIGHTING:
        img[:, :, 0] *= 1.15
        img[:, :, 1] *= 0.86
        img[:, :, 2] *= 0.62
        img *= 0.72
    else:
        img = img * 1.05 + 6.0
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_feature_lighting_overlay(profile, viewport_x, viewport_y, view_w, view_h):
    if profile != FEATURE_RUN_LIGHTING:
        return
    glViewport(viewport_x, viewport_y, view_w, view_h)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, view_w, view_h, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.05, 0.11, 0.28, 0.35)
    glBegin(GL_QUADS)
    glVertex2f(0, 0); glVertex2f(view_w, 0)
    glVertex2f(view_w, view_h); glVertex2f(0, view_h)
    glEnd()
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def read_viewport_bgr(x, y, view_w, view_h, buffer=GL_FRONT, lighting_profile=None):
    glReadBuffer(buffer)
    pixels = glReadPixels(x, y, view_w, view_h, GL_RGB, GL_UNSIGNED_BYTE)
    arr = np.frombuffer(pixels, dtype=np.uint8).reshape((view_h, view_w, 3))
    arr = np.flipud(arr)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if lighting_profile:
        bgr = apply_feature_lighting_bgr(bgr, lighting_profile)
    return bgr


def render_pose_to_bgr(view_w, view_h, pose, lighting_profile=FEATURE_PRE_LIGHTING):
    cx, cy, cz, rx, ry, rz = pose
    glViewport(0, 0, view_w, view_h)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(45, view_w / view_h, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_gradient_background()
    glRotatef(ry, 0, 1, 0)
    glRotatef(rx, 1, 0, 0)
    glRotatef(rz, 0, 0, 1)
    glTranslatef(cx, cy, cz)
    draw_terrain_vbo(terrain_vbo, terrain_vertex_count)
    return read_viewport_bgr(0, 0, view_w, view_h, buffer=GL_BACK,
                             lighting_profile=lighting_profile)


def render_view_matrix_to_bgr(view_w, view_h, view_mat,
                              lighting_profile=FEATURE_RUN_LIGHTING):
    glViewport(0, 0, view_w, view_h)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(45, view_w / view_h, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_gradient_background()
    glLoadMatrixd(view_mat)
    draw_terrain_vbo(terrain_vbo, terrain_vertex_count)
    return read_viewport_bgr(0, 0, view_w, view_h, buffer=GL_BACK,
                             lighting_profile=lighting_profile)


def draw_bgr_image_in_view(bgr, viewport_x, viewport_y, view_w, view_h):
    if bgr is None:
        return
    img = cv2.resize(bgr, (view_w, view_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = np.flipud(rgb).copy()
    glViewport(viewport_x, viewport_y, view_w, view_h)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, view_w, 0, view_h, -1, 1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glRasterPos2i(0, 0)
    glDrawPixels(view_w, view_h, GL_RGB, GL_UNSIGNED_BYTE, rgb)
    glEnable(GL_DEPTH_TEST)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def feature_view_matrix_from_pnp(R, tvec):
    ext = np.eye(4, dtype=np.float64)
    ext[:3, :3] = R
    ext[:3, 3] = np.asarray(tvec, dtype=np.float64).flatten()
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return (flip @ ext).T.flatten().astype(np.float64).tolist()


def validate_feature_pnp_correspondences(correspondences, min_count=4,
                                         rounding_digits=6):
    valid = []
    distinct_2d = set()
    distinct_3d = set()
    for item in correspondences:
        try:
            image_point, world_point = item
            p2 = tuple(float(v) for v in image_point)
            p3 = tuple(float(v) for v in world_point)
        except (TypeError, ValueError):
            continue
        if len(p2) != 2 or len(p3) != 3:
            continue
        if not all(math.isfinite(v) for v in p2 + p3):
            continue
        valid.append((p2, p3))
        distinct_2d.add(tuple(round(v, rounding_digits) for v in p2))
        distinct_3d.add(tuple(round(v, rounding_digits) for v in p3))

    stats = {
        "valid": len(valid),
        "distinct_2d": len(distinct_2d),
        "distinct_3d": len(distinct_3d),
    }
    if (stats["valid"] < min_count or
            stats["distinct_2d"] < min_count or
            stats["distinct_3d"] < min_count):
        reason = (
            "Feature PnP requires at least 4 distinct valid 2D-3D "
            "correspondences. "
            f"Valid: {stats['valid']}, distinct 2D: {stats['distinct_2d']}, "
            f"distinct 3D: {stats['distinct_3d']}."
        )
        return False, valid, stats, reason
    return True, valid, stats, None


def solve_pnp_feature(correspondences, view_w, view_h, actual_cam):
    result = {
        "success": False,
        "failure_reason": None,
        "estimated_pose": None,
        "R": None,
        "tvec": None,
        "view_matrix": None,
        "pnp_inlier_count": None,
        "reprojection_error": None,
        "position_error": None,
        "rotation_error": None,
        "low_confidence": False,
        "quality_warnings": [],
    }
    is_valid, valid_correspondences, validation_stats, failure_reason = (
        validate_feature_pnp_correspondences(correspondences)
    )
    result["validation_stats"] = validation_stats
    if not is_valid:
        # The one true hard floor: PnP is mathematically undefined below 4
        # distinct correspondences, so there is nothing a solver can do.
        # Every other check below is a *policy* gate, not a hard floor --
        # those get downgraded to warnings so a pose is still returned
        # whenever there are enough pairs to attempt one at all.
        result["failure_reason"] = failure_reason
        return result

    K = build_camera_intrinsics(view_w, view_h)
    dist = np.zeros((4, 1))
    pts3d = np.array([[w[0], w[1], w[2]] for (_, w) in valid_correspondences],
                     dtype=np.float64)
    pts2d = np.array([[p[0], p[1]] for (p, _) in valid_correspondences],
                     dtype=np.float64)

    used_fallback = False
    inlier_indices = None
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, K, dist, flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        ok = False
        inliers = None
    if ok and inliers is not None and len(inliers) > 0:
        inlier_indices = np.asarray(inliers, dtype=np.int32).reshape(-1)
    else:
        # RANSAC couldn't find a consensus set. We still have >= 4
        # correspondences, so fall back to a direct (non-robust) solve over
        # every pair -- this always yields *a* pose given enough points, just
        # without outlier rejection, and is flagged as low-confidence below.
        used_fallback = True
        solvers = [(cv2.SOLVEPNP_EPNP, "EPNP"), (cv2.SOLVEPNP_IPPE, "IPPE")]
        rvec, tvec = _try_solvers(pts3d, pts2d, K, dist, solvers)
        if rvec is None:
            result["failure_reason"] = (
                "Feature Run: no solver could produce any pose from these "
                f"{len(valid_correspondences)} correspondence(s) -- "
                "the points are likely degenerate (e.g. collinear)."
            )
            return result
        inlier_indices = np.arange(len(valid_correspondences), dtype=np.int32)

    valid_count = int(len(valid_correspondences))
    inlier_count = int(len(inlier_indices))
    inlier_ratio = inlier_count / float(valid_count)

    try:
        cv2.solvePnPRefineLM(pts3d[inlier_indices], pts2d[inlier_indices],
                             K, dist, rvec, tvec)
    except cv2.error:
        pass

    R, _ = cv2.Rodrigues(rvec)
    proj, _ = cv2.projectPoints(pts3d[inlier_indices], rvec, tvec, K, dist)
    reproj_err = float(np.mean(np.linalg.norm(
        proj.reshape(-1, 2) - pts2d[inlier_indices], axis=1)))

    quality_warnings = []
    if used_fallback:
        quality_warnings.append(
            "RANSAC did not converge on a consensus set; used a direct "
            "solve over all pairs with no outlier rejection."
        )
    if inlier_count < FEATURE_RUN_MIN_RANSAC_INLIERS:
        quality_warnings.append(
            f"only {inlier_count} inlier(s), below the usual "
            f"{FEATURE_RUN_MIN_RANSAC_INLIERS}-inlier guideline."
        )
    if inlier_ratio < FEATURE_RUN_MIN_RANSAC_INLIER_RATIO:
        quality_warnings.append(
            f"inlier ratio {inlier_ratio:.2f} is below the usual "
            f"{FEATURE_RUN_MIN_RANSAC_INLIER_RATIO:.2f} guideline."
        )
    if reproj_err > FEATURE_RUN_MAX_INLIER_REPROJECTION_ERROR:
        quality_warnings.append(
            f"mean reprojection error {reproj_err:.2f}px is above the "
            f"usual {FEATURE_RUN_MAX_INLIER_REPROJECTION_ERROR:.2f}px guideline."
        )

    cam_pos = (-R.T @ tvec).flatten()
    estimated_pose = (
        float(-cam_pos[0]), float(-cam_pos[1]), float(-cam_pos[2]),
        *tuple(float(v) for v in render_euler_from_pnp_R(R)),
    )
    ax, ay, az, arx, ary, arz = actual_cam
    pos_err = math.sqrt((cam_pos[0] + ax) ** 2 +
                        (cam_pos[1] + ay) ** 2 +
                        (cam_pos[2] + az) ** 2)
    view_mat = feature_view_matrix_from_pnp(R, tvec)
    rot_err = rotation_geodesic_error(
        gl_rotation_from_view_matrix(view_mat), arx, ary, arz)
    result.update({
        "success": True,
        "estimated_pose": estimated_pose,
        "R": R,
        "tvec": tvec,
        "view_matrix": view_mat,
        "pnp_inlier_count": inlier_count,
        "pnp_inlier_ratio": inlier_ratio,
        "reprojection_error": reproj_err,
        "position_error": float(pos_err),
        "rotation_error": float(rot_err),
        "low_confidence": bool(quality_warnings),
        "quality_warnings": quality_warnings,
    })
    return result



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


def draw_feature_pre_world_points():
    if feature_pre_db is None or not feature_pre_db.views:
        return
    active_view = feature_pre_db.views[feature_pre_active_index]
    seen_point_ids = set()
    points = []
    for mapping in feature_pre_db.mappings_for_view(active_view.view_id):
        point_id = mapping.get("point_id")
        if point_id is not None and point_id in seen_point_ids:
            continue  # an ASIFT variant of a point already counted for this view
        if point_id is not None:
            seen_point_ids.add(point_id)
        points.append(mapping["world_point"])
    if not points:
        return
    width, height = pygame.display.get_surface().get_size()
    setup_left_view_matrices(width, height)
    viewport = glGetIntegerv(GL_VIEWPORT)
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    projected = []
    for point in points:
        sx, sy, sz = gluProject(point[0], point[1], point[2],
                                modelview, projection, viewport)
        if 0.0 <= sz <= 1.0:
            projected.append((sx, height - sy))
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, width, height, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    for point in projected:
        draw_2d_pick_dot(point, color=(1.0, 0.0, 0.0), radius=4)
    glEnable(GL_DEPTH_TEST)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def get_feature_pre_candidate_indices(view):
    cached = feature_pre_candidate_cache.get(view.view_id)
    if cached is not None:
        return cached

    count = len(view.keypoints)
    if count <= FEATURE_PRE_MAX_CANDIDATES:
        indices = np.arange(count, dtype=np.int32)
        feature_pre_candidate_cache[view.view_id] = indices
        return indices

    points = np.asarray(view.keypoints, dtype=np.float32)
    view_w, view_h = feature_pre_db.metadata.get("view_resolution", [640, 480])
    view_w = max(float(view_w), 1.0)
    view_h = max(float(view_h), 1.0)
    cols = int(math.ceil(math.sqrt(FEATURE_PRE_MAX_CANDIDATES)))
    rows = int(math.ceil(FEATURE_PRE_MAX_CANDIDATES / cols))
    cell_best = {}

    for keypoint_id, (x, y) in enumerate(points):
        col = min(cols - 1, max(0, int((float(x) / view_w) * cols)))
        row = min(rows - 1, max(0, int((float(y) / view_h) * rows)))
        center_x = (col + 0.5) * view_w / cols
        center_y = (row + 0.5) * view_h / rows
        dist2 = (float(x) - center_x) ** 2 + (float(y) - center_y) ** 2
        cell = (row, col)
        current = cell_best.get(cell)
        if current is None or dist2 < current[0] or (
                dist2 == current[0] and keypoint_id < current[1]):
            cell_best[cell] = (dist2, keypoint_id)

    selected = []
    selected_set = set()
    for row in range(rows):
        for col in range(cols):
            current = cell_best.get((row, col))
            if current is not None:
                selected.append(current[1])
                selected_set.add(current[1])
                if len(selected) >= FEATURE_PRE_MAX_CANDIDATES:
                    break
        if len(selected) >= FEATURE_PRE_MAX_CANDIDATES:
            break

    if len(selected) < FEATURE_PRE_MAX_CANDIDATES:
        for keypoint_id in range(count):
            if keypoint_id not in selected_set:
                selected.append(keypoint_id)
                if len(selected) >= FEATURE_PRE_MAX_CANDIDATES:
                    break

    indices = np.asarray(selected, dtype=np.int32)
    feature_pre_candidate_cache[view.view_id] = indices
    return indices


def draw_feature_pre_keypoints_2d():
    if feature_pre_db is None or not feature_pre_db.views:
        return
    view = feature_pre_db.views[feature_pre_active_index]
    width, height = pygame.display.get_surface().get_size()
    half_w = width // 2
    mapped_ids = feature_pre_db.mapped_keypoint_ids(view.view_id)
    pending_id = None
    if feature_pre_pending and feature_pre_pending["view_id"] == view.view_id:
        pending_id = feature_pre_pending["keypoint_id"]

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0, width, height, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glLineWidth(2.0)
    if feature_pre_show_keypoints:
        glColor3f(0.0, 0.55, 0.75)
        glPointSize(4.0)
        glBegin(GL_POINTS)
        for keypoint_id in get_feature_pre_candidate_indices(view):
            x, y = view.keypoints[int(keypoint_id)]
            glVertex2f(half_w + float(x), float(y))
        glEnd()
        glPointSize(1.0)
    for keypoint_id in mapped_ids:
        if keypoint_id < len(view.keypoints):
            draw_2d_pick_marker((half_w + float(view.keypoints[keypoint_id][0]),
                                 float(view.keypoints[keypoint_id][1])),
                                (0.1, 1.0, 0.2), size=6)
    if pending_id is not None and pending_id < len(view.keypoints):
        draw_2d_pick_marker((half_w + float(view.keypoints[pending_id][0]),
                             float(view.keypoints[pending_id][1])),
                            (1.0, 1.0, 0.0), size=9)
    glLineWidth(1.0)
    glEnable(GL_DEPTH_TEST)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def _pose_to_render_position(pose):
    return (-pose[0], -pose[1], -pose[2])


def draw_world_polyline(points, color, width=3.0):
    if len(points) < 2:
        return
    glDisable(GL_LIGHTING)
    glLineWidth(width)
    glColor3f(*color)
    glBegin(GL_LINE_STRIP)
    for point in points:
        glVertex3f(float(point[0]), float(point[1]), float(point[2]))
    glEnd()
    glLineWidth(1.0)


def draw_feature_run_paths(attempts):
    true_points = [_pose_to_render_position(attempt["true_pose"])
                   for attempt in attempts]
    draw_world_polyline(true_points, (0.0, 0.28, 1.0), width=3.0)

    current_segment = []
    for attempt in attempts:
        if attempt["success"] and attempt.get("estimated_pose") is not None:
            current_segment.append(_pose_to_render_position(attempt["estimated_pose"]))
        else:
            draw_world_polyline(current_segment, (0.0, 1.0, 0.18), width=3.0)
            current_segment = []
    draw_world_polyline(current_segment, (0.0, 1.0, 0.18), width=3.0)


def draw_feature_run_attempts(attempts):
    global feature_current_pair_index
    draw_feature_run_paths(attempts)
    for index, attempt in enumerate(attempts):
        true_pose = attempt["true_pose"]
        draw_camera_pyramid(*true_pose)
        if attempt["success"] and attempt.get("estimated_pose") is not None:
            draw_tracker_cam_pairs([(true_pose, attempt["estimated_pose"])])
        elif not attempt["success"]:
            draw_sphere(-true_pose[0], -true_pose[1], -true_pose[2],
                        color=(1.0, 0.0, 0.0), radius=1.6)
        if index == feature_current_pair_index:
            draw_sphere(-true_pose[0], -true_pose[1], -true_pose[2],
                        color=(1.0, 1.0, 0.0), radius=2.2)


def draw_feature_run_overview_annotations(width, height):
    if not feature_run_attempts:
        return
    glViewport(width // 2, 0, width // 2, height)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(45, (width / 2) / height, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    glRotatef(r_y2, 0, 1, 0)
    glRotatef(r_x2, 1, 0, 0)
    glRotatef(r_z2, 0, 0, 1)
    glTranslatef(c_x2, c_y2, c_z2)

    depth_was_enabled = bool(glIsEnabled(GL_DEPTH_TEST))
    old_depth_mask = bool(np.asarray(glGetBooleanv(GL_DEPTH_WRITEMASK)).item())
    glDisable(GL_DEPTH_TEST)
    glDepthMask(GL_FALSE)
    draw_feature_run_attempts(feature_run_attempts)
    glDepthMask(GL_TRUE if old_depth_mask else GL_FALSE)
    if depth_was_enabled:
        glEnable(GL_DEPTH_TEST)
    else:
        glDisable(GL_DEPTH_TEST)


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
                 feature_mode=False, feature_pre_mode=False, motion_scale=1.0):
    global CONFIG, ACTIVE_MAP
    global c_x, c_y, c_z, r_x, r_y, r_z
    global c_x2, c_y2, c_z2, r_x2, r_y2, r_z2
    global picking_mode, picked_points, pending_left_world_point
    global terrain_vbo, terrain_vertex_count
    global tracker_points, tracker_cam_pairs
    global tracker_overlay_active, tracker_current_pair_index
    global feature_cam_pairs, feature_overlay_active, feature_current_pair_index
    global feature_run_attempts

    r_speed   = 1 * motion_scale
    rot_speed = 1 * motion_scale

    dx =  math.sin(r_y * math.pi / 180)
    dz = -math.cos(r_y * math.pi / 180)
    rx =  math.cos(r_y * math.pi / 180)
    rz =  math.sin(r_y * math.pi / 180)

    if apply_input and (not picking_mode or trackers_mode or feature_mode or feature_pre_mode):
        keys_pressed = pygame.key.get_pressed()
        mods = pygame.key.get_mods()
        allow_movement = not (mods & (KMOD_CTRL | KMOD_ALT))
        if recording_mode or trackers_mode or feature_mode or feature_pre_mode:
            moved = False
            yaw_delta = 0.0
            if allow_movement and keys_pressed[K_LEFT]:   yaw_delta -= rot_speed;  moved = True
            if allow_movement and keys_pressed[K_RIGHT]:  yaw_delta += rot_speed;  moved = True
            tilt = 0.0
            if allow_movement and keys_pressed[K_DOWN]:   tilt += rot_speed;  moved = True
            if allow_movement and keys_pressed[K_UP]:     tilt -= rot_speed;  moved = True

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
            if allow_movement and keys_pressed[K_a]:      c_x += rx * r_speed;  c_z += rz * r_speed;  moved = True
            if allow_movement and keys_pressed[K_d]:      c_x -= rx * r_speed;  c_z -= rz * r_speed;  moved = True
            if allow_movement and keys_pressed[K_w]:      c_x -= dx * r_speed;  c_z -= dz * r_speed;  moved = True
            if allow_movement and keys_pressed[K_s]:      c_x += dx * r_speed;  c_z += dz * r_speed;  moved = True
            if allow_movement and keys_pressed[K_SPACE]:  c_y -= r_speed;  moved = True
            if allow_movement and keys_pressed[K_LSHIFT]: c_y += r_speed;  moved = True
            if moved and trackers_mode:
                tracker_overlay_active = False
            if moved and feature_mode:
                feature_overlay_active = False
            if keys_pressed[K_BACKSPACE]:
                img = cv2.imread(ACTIVE_MAP["height_path"])
                ih, iw, _ = img.shape
                m = ACTIVE_MAP["margin"]
                c_x, c_y, c_z = -iw/m/2, float(CONFIG.get("start_h")), -(ih/m)-100
                r_x, r_y, r_z = float(CONFIG.get("start_a")), 0.0, 0.0
                tracker_overlay_active = False
                feature_overlay_active = False
        else:
            global saved_positions, recording_index
            c_x, c_y, c_z, r_x, r_y, r_z = saved_positions[recording_index]

    elif not apply_input and not picking_mode:
        if not feature_pre_mode:
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
def draw(recording_mode, trackers_mode=False, feature_mode=False,
         feature_pre_mode=False, motion_scale=1.0):
    global picking_mode, pnp_result, picked_correspondences
    global c_x, c_y, c_z, r_x, r_y, r_z
    global c_x2, c_y2, c_z2, r_x2, r_y2, r_z2
    global tracker_cam_pairs, tracker_overlay_active, tracker_current_pair_index
    global tracker_est_view_mats
    global feature_cam_pairs, feature_overlay_active, feature_current_pair_index
    global feature_est_view_mats
    global feature_run_attempts, feature_diff_mode
    global feature_pre_db, feature_pre_active_index
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
                 trackers_mode=trackers_mode, feature_mode=feature_mode,
                 feature_pre_mode=feature_pre_mode,
                 motion_scale=motion_scale)
    if feature_mode:
        draw_feature_lighting_overlay(FEATURE_PRE_LIGHTING, 0, 0, width // 2, height)
    if picking_mode:
        draw_left_world_pick_markers(picked_points, pending_left_world_point)
    if feature_pre_mode:
        draw_feature_pre_world_points()

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
    if (feature_mode and feature_overlay_active and feature_run_attempts
            and feature_current_pair_index in feature_est_view_mats):
        view_mat = feature_est_view_mats[feature_current_pair_index]
        attempt = feature_run_attempts[feature_current_pair_index]

        if feature_diff_mode and attempt.get("diff_image") is not None:
            draw_bgr_image_in_view(attempt["diff_image"], 0, 0, width // 2, height)
            draw_text_2d("absolute difference", 12, 10, color=(255, 230, 120))
        else:

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
        draw_text_2d(f"pos err: {attempt['position_error']:.2f} units", 12, 34,
                     color=(255, 230, 120))
        draw_text_2d(f"rot err: {attempt['rotation_error']:.2f} deg", 12, 58,
                     color=(255, 230, 120))

    glViewport(width // 2, 0, width // 2, height)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(45, (width/2) / height, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW)
    draw_gradient_background()

    old_cam = (c_x, c_y, c_z, r_x, r_y, r_z)
    c_x, c_y, c_z = c_x2, c_y2, c_z2
    r_x, r_y, r_z = r_x2, r_y2, r_z2
    if feature_pre_mode and feature_pre_db and feature_pre_db.views:
        c_x, c_y, c_z, r_x, r_y, r_z = feature_pre_db.views[feature_pre_active_index].pose
    render_scene(apply_input=False, recording_mode=recording_mode,
                 trackers_mode=trackers_mode, feature_mode=feature_mode,
                 feature_pre_mode=feature_pre_mode)
    if feature_mode:
        draw_feature_lighting_overlay(FEATURE_RUN_LIGHTING,
                                      width // 2, 0, width // 2, height)
        draw_feature_run_overview_annotations(width, height)
    c_x, c_y, c_z, r_x, r_y, r_z = old_cam
    if feature_pre_mode:
        draw_feature_pre_keypoints_2d()

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
    elif feature_pre_mode:
        mode_label = "FEATURE PRE DEMO" if feature_pre_demo else "FEATURE PRE EDIT"
        if feature_pre_db and feature_pre_db.views:
            view = feature_pre_db.views[feature_pre_active_index]
            mode_label += (
                f" | view {feature_pre_active_index + 1}/{feature_pre_db.view_count()}"
                f" | mapped {view.mapped_count(feature_pre_db.mappings)}"
            )
        draw_text_2d(mode_label, 12, height - 28, color=(255, 255, 100))
    elif recording_mode:
        
        draw_text_2d("RECORDING MODE", 12, height - 28, color=(100, 100, 255))
    elif feature_mode:
        label = "FEATURE RUN"
        if feature_run_attempts:
            attempt = feature_run_attempts[feature_current_pair_index]
            if attempt["success"] and attempt.get("low_confidence"):
                status = "LOW CONF"
            elif attempt["success"]:
                status = "OK"
            else:
                status = "FAILED"
            label += f" {feature_current_pair_index + 1}/{len(feature_run_attempts)} {status}"
        draw_text_2d(label, 12, height - 28, color=(255, 255, 100))
        if feature_run_attempts:
            attempt = feature_run_attempts[feature_current_pair_index]
            if attempt["success"] and attempt.get("low_confidence"):
                draw_text_2d(
                    f"matches {attempt['good_descriptor_match_count']} | "
                    f"inliers {attempt['pnp_inlier_count']} | LOW CONFIDENCE",
                    12, height - 54, color=(255, 200, 80))
                warnings = attempt.get("quality_warnings") or []
                if warnings:
                    draw_text_2d(warnings[0], 12, height - 80, color=(255, 170, 130))
            elif attempt["success"]:
                draw_text_2d(
                    f"matches {attempt['good_descriptor_match_count']} | inliers {attempt['pnp_inlier_count']}",
                    12, height - 54, color=(255, 230, 120))
            else:
                draw_text_2d("FEATURE MATCHING FAILED", 12, height - 54,
                             color=(255, 80, 80))
                draw_text_2d(str(attempt["failure_reason"]), 12, height - 80,
                             color=(255, 170, 170))
    else:
        draw_text_2d("VIEWING MODE", 12, height - 28, color=(100, 255, 100))
    draw_seperator_line()
    pygame.display.flip()


def update_window_caption(mode):
    if mode == "pre":
        if feature_pre_demo:
            caption = (
                "Feature Pre [DEMO - READ ONLY] | WASD/Arrows - move 3D view | "
                "V - new temporary view | [/] - browse temporary views | H - keypoints | "
                "Right click - feature | Left click - 3D match | U - undo | Esc - exit | "
                "Saving disabled"
            )
        else:
            caption = (
                "Feature Pre [EDIT] | WASD/Arrows - move 3D view | V - new view | "
                "[/] - browse views | H - keypoints | Right click - feature | "
                "Left click - 3D match | U - undo | Ctrl+A - ASIFT augment | "
                "Ctrl+S - save | Ctrl+P - snapshot | Esc - exit"
            )
    elif feature_mode:
        attempt_info = (
            f" | Attempt {feature_current_pair_index + 1}/{len(feature_run_attempts)}"
            if feature_run_attempts else ""
        )
        caption = (
            "Feature Run | WASD/Arrows - move | B - estimate pose | "
            "N/M - browse attempts | O - overlay/diff | F - exit feature run"
            f"{attempt_info}"
        )
    elif trackers_mode:
        pair_info = (
            f" | Pair {tracker_current_pair_index + 1}/{len(tracker_cam_pairs)}"
            if tracker_cam_pairs else ""
        )
        caption = (
            "Trackers Mode | WASD/Arrows - move | B - estimate | "
            f"N/M - browse pairs{pair_info} | P - picking | R - recording"
        )
    elif picking_mode:
        caption = (
            "Picking Mode | Left click - 3D point | Right view click - 2D match | "
            "C - solve PnP | R - recording | T - trackers"
        )
    elif recording_mode:
        caption = (
            "Recording Mode | WASD/Arrows - move | B - save pose | "
            "R - navigation | P - picking | T - trackers | F - feature run"
        )
    else:
        caption = (
            "Navigation Mode | R - recording | P - picking | "
            "T - trackers | F - feature run"
        )
    pygame.display.set_caption(caption)


def select_feature_pre_action(map_name):
    active_path = active_database_path(map_name)
    has_active = os.path.exists(active_path)
    print(f"\nFeature Pre Mode for: {map_name}\n")
    if not has_active:
        print("No active feature database exists for this map.")
    print("1. Continue / edit existing active database")
    print("2. Create a new empty database")
    print("3. Demo mode - temporary, read-only, never saves")
    print("4. Cancel")
    while True:
        choice = input("\nEnter selection: ").strip()
        if choice == "1":
            if has_active:
                return "continue"
            print("No active database to continue. Choose 2 for a new database or 3 for demo.")
        elif choice == "2":
            return "new"
        elif choice == "3":
            return "demo"
        elif choice == "4":
            return "cancel"
        else:
            print("Invalid selection. Enter 1, 2, 3, or 4.")


def create_feature_reference_view(view_w, view_h):
    global feature_pre_active_index, feature_pre_pending
    pose = (c_x, c_y, c_z, r_x, r_y, r_z)
    bgr = render_pose_to_bgr(view_w, view_h, pose, FEATURE_PRE_LIGHTING)
    keypoints, descriptors = detect_sift_features(bgr)
    view = feature_pre_db.add_reference_view(pose, keypoints, descriptors)
    feature_pre_active_index = feature_pre_db.view_count() - 1
    feature_pre_pending = None
    print(
        f"Created reference view {view.view_id}: "
        f"{len(keypoints)} SIFT keypoints detected."
    )


def select_nearest_feature(local_x, local_y, max_dist=12.0):
    global feature_pre_pending
    if feature_pre_db is None or not feature_pre_db.views:
        print("Create a reference view with V before selecting features.")
        return
    view = feature_pre_db.views[feature_pre_active_index]
    if len(view.keypoints) == 0:
        print("Active reference view has no detected SIFT features.")
        return
    candidate_indices = get_feature_pre_candidate_indices(view)
    if len(candidate_indices) == 0:
        print("Active reference view has no visible SIFT candidates.")
        return
    click = np.array([local_x, local_y], dtype=np.float32)
    candidate_points = view.keypoints[candidate_indices]
    dists = np.linalg.norm(candidate_points - click, axis=1)
    candidate_pos = int(np.argmin(dists))
    keypoint_id = int(candidate_indices[candidate_pos])
    if float(dists[candidate_pos]) > max_dist:
        print("No detected feature near the clicked location.")
        return
    if feature_pre_db.mapping_exists(view.view_id, keypoint_id):
        print("That detected SIFT keypoint is already mapped in this reference view.")
        return
    feature_pre_pending = {
        "view_id": view.view_id,
        "keypoint_id": keypoint_id,
    }
    x, y = view.keypoints[keypoint_id]
    print(f"Pending feature selected at exact keypoint ({x:.1f}, {y:.1f}).")


def complete_pending_feature_mapping(world_point):
    global feature_pre_pending
    if feature_pre_pending is None:
        print("Right click a detected SIFT feature before choosing the 3D match.")
        return
    mapping = feature_pre_db.add_mapping(
        feature_pre_pending["view_id"],
        feature_pre_pending["keypoint_id"],
        world_point,
    )
    if mapping is None:
        print("That detected SIFT keypoint is already mapped in this reference view.")
    else:
        feature_pre_session_added.append(mapping)
        print(f"Mapped feature to 3D point {world_point}.")
    feature_pre_pending = None


def undo_feature_pre_mapping():
    if not feature_pre_session_added:
        print("Nothing to undo from this Pre session.")
        return
    mapping = feature_pre_session_added.pop()
    if feature_pre_db.remove_mapping(mapping):
        print("Undid the most recent mapping from this Pre session.")


# ---------------------------------------------------------------------------
# ASIFT-style multi-view descriptor augmentation
#
# Plain SIFT only tolerates modest viewpoint change. To let a sparse set of
# manually-mapped points still match from viewpoints/rotations that weren't
# directly captured, we simulate several affine "tilt" views of each existing
# reference view (the classic Affine-SIFT / ASIFT trick), re-detect SIFT on
# each simulated view, and snap any synthetic keypoint that lands on an
# already-mapped point back onto that point's point_id -- adding a new
# descriptor *variant* for the same 3D landmark rather than a new landmark.
# No new manual clicking is required: the 3D world_point is already known
# from the original mapping, only the descriptor differs per simulated view.
# ---------------------------------------------------------------------------
def _rotation_bound_matrix(angle_deg, w, h):
    """3x3 homogeneous matrix rotating a w x h image about its center and
    translating so the rotated content stays fully inside a new bounding
    box. Returns (M, new_w, new_h)."""
    phi = math.radians(angle_deg)
    c, s = math.cos(phi), math.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64)
    rotated_corners = (R @ corners.T).T
    min_x, min_y = rotated_corners[:, 0].min(), rotated_corners[:, 1].min()
    max_x, max_y = rotated_corners[:, 0].max(), rotated_corners[:, 1].max()
    T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    M = T @ R
    new_w = max(1, int(math.ceil(max_x - min_x)))
    new_h = max(1, int(math.ceil(max_y - min_y)))
    return M, new_w, new_h


def simulate_asift_view(bgr, tilt, phi_deg):
    """
    Simulate viewing `bgr` from a different out-of-plane angle. Returns
    (warped_bgr, inverse_matrix) where inverse_matrix maps a pixel coordinate
    in warped_bgr back to a pixel coordinate in the original bgr.
    """
    h, w = bgr.shape[:2]
    if abs(phi_deg) < 1e-6:
        rotated, rw, rh = bgr, w, h
        m_rot = np.eye(3, dtype=np.float64)
    else:
        m_rot, rw, rh = _rotation_bound_matrix(phi_deg, w, h)
        rotated = cv2.warpAffine(bgr, m_rot[:2, :], (rw, rh),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    if abs(tilt - 1.0) < 1e-6:
        tilted, m_tilt = rotated, np.eye(3, dtype=np.float64)
    else:
        sigma = 0.8 * math.sqrt(max(tilt * tilt - 1.0, 0.0))
        blurred = cv2.GaussianBlur(rotated, (0, 0), sigmaX=sigma, sigmaY=0.01)
        tw = max(1, int(round(rw / tilt)))
        tilted = cv2.resize(blurred, (tw, rh), interpolation=cv2.INTER_AREA)
        m_tilt = np.array([[1.0 / tilt, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    m_total = m_tilt @ m_rot  # forward: original pixel -> warped pixel
    m_inv = np.linalg.inv(m_total)
    return tilted, m_inv


def _apply_affine_2d(matrix3x3, x, y):
    v = matrix3x3 @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


ASIFT_TILTS = (1.4, 2.0, 2.8)
ASIFT_ROTATIONS_DEG = (0, 45, 90, 135)


def augment_reference_view_with_asift(view, view_w, view_h,
                                      tilts=ASIFT_TILTS,
                                      rotations_deg=ASIFT_ROTATIONS_DEG,
                                      match_px_tol=3.0):
    existing = feature_pre_db.mappings_for_view(view.view_id)
    if not existing:
        return 0
    bgr = render_pose_to_bgr(view_w, view_h, view.pose, FEATURE_PRE_LIGHTING)
    orig_px = np.asarray([m["keypoint"] for m in existing], dtype=np.float32)
    added = 0
    for tilt in tilts:
        for phi in rotations_deg:
            if tilt == 1.0 and phi == 0:
                continue  # identical to the already-stored original descriptors
            warped, m_inv = simulate_asift_view(bgr, tilt, phi)
            if warped.shape[0] < 16 or warped.shape[1] < 16:
                continue
            kps, descs = detect_sift_features(warped)
            if len(kps) == 0:
                continue
            back = np.asarray(
                [_apply_affine_2d(m_inv, x, y) for x, y in kps], dtype=np.float32
            )
            diffs = back[:, None, :] - orig_px[None, :, :]
            dist2 = np.einsum('ijk,ijk->ij', diffs, diffs)
            nearest_idx = np.argmin(dist2, axis=1)
            nearest_dist = np.sqrt(dist2[np.arange(len(back)), nearest_idx])
            for k in range(len(back)):
                if nearest_dist[k] > match_px_tol:
                    continue
                base = existing[int(nearest_idx[k])]
                feature_pre_db.add_mapping_variant(
                    point_id=base["point_id"],
                    view_id=view.view_id,
                    keypoint=base["keypoint"],
                    descriptor=descs[k],
                    world_point=base["world_point"],
                )
                added += 1
    return added


def augment_feature_pre_db_with_asift(view_w, view_h):
    if feature_pre_demo:
        print("Demo mode: augmenting is fine to try, but won't be saved.")
    if feature_pre_db is None or not feature_pre_db.views:
        print("No reference views to augment yet -- create one with V first.")
        return
    before_rows = feature_pre_db.mapped_feature_count()
    before_points = feature_pre_db.distinct_point_count()
    total_added = 0
    print(f"Running ASIFT augmentation across {feature_pre_db.view_count()} view(s)...")
    for view in feature_pre_db.views:
        n = augment_reference_view_with_asift(view, view_w, view_h)
        print(f"  {view.view_id}: +{n} descriptor variant(s)")
        total_added += n
    print(
        f"ASIFT augmentation done: {total_added} variants added "
        f"({before_rows} -> {feature_pre_db.mapped_feature_count()} mapping rows, "
        f"still {feature_pre_db.distinct_point_count()} distinct 3D points "
        f"[{before_points} before]). Press Ctrl+S to save if this looks good."
    )


def save_feature_pre_active():
    if feature_pre_demo:
        print("Demo mode: changes are temporary and will not be saved.")
        return
    path = feature_pre_db.save_active(ACTIVE_MAP["name"])
    print(
        f"Saved active feature database to {path}: "
        f"{feature_pre_db.view_count()} views, "
        f"{feature_pre_db.mapped_feature_count()} mapped features."
    )


def save_feature_pre_snapshot():
    if feature_pre_demo:
        print("Demo mode: changes are temporary and will not be saved.")
        return
    path = feature_pre_db.save_snapshot(ACTIVE_MAP["name"])
    print(f"Saved feature database snapshot to {path}.")


def record_feature_run_attempt(view_w, view_h):
    global feature_current_pair_index, feature_overlay_active
    actual_pose = (c_x, c_y, c_z, r_x, r_y, r_z)
    if active_feature_db is None:
        reason = "no valid feature database loaded"
        print(f"Feature Run: {reason}. Use: python world_split_v5.17.py --pre")
        attempt = {
            "true_pose": actual_pose,
            "estimated_pose": None,
            "success": False,
            "failure_reason": reason,
            "query_keypoint_count": 0,
            "good_descriptor_match_count": 0,
            "pnp_inlier_count": None,
            "reprojection_error": None,
            "position_error": None,
            "rotation_error": None,
            "true_image": None,
            "diff_image": None,
        }
        feature_run_attempts.append(attempt)
        feature_current_pair_index = len(feature_run_attempts) - 1
        feature_overlay_active = False
        return

    query_bgr = render_pose_to_bgr(
        view_w, view_h, actual_pose, FEATURE_RUN_LIGHTING)
    correspondences, _, metrics, match_failure = match_query_to_database(
        active_feature_db, query_bgr)
    print(
        f"[features] {metrics['query_keypoint_count']} query keypoints, "
        f"{metrics['good_descriptor_match_count']} good descriptor matches."
    )
    pnp = solve_pnp_feature(
        correspondences, view_w, view_h,
        actual_cam=actual_pose,
    )
    if match_failure and not pnp["success"]:
        pnp["failure_reason"] = match_failure
    estimated_image = None
    diff_image = None
    if pnp["success"]:
        estimated_image = render_view_matrix_to_bgr(view_w, view_h, pnp["view_matrix"])
        diff_image = cv2.absdiff(query_bgr, estimated_image)
        if pnp["low_confidence"]:
            print(
                "Feature Run result (LOW CONFIDENCE): "
                f"inliers={pnp['pnp_inlier_count']} "
                f"reproj={pnp['reprojection_error']:.2f}px "
                f"pos_err={pnp['position_error']:.3f} "
                f"rot_err={pnp['rotation_error']:.3f}"
            )
            for warning in pnp["quality_warnings"]:
                print(f"  - {warning}")
        else:
            print(
                "Feature Run success: "
                f"inliers={pnp['pnp_inlier_count']} "
                f"reproj={pnp['reprojection_error']:.2f}px "
                f"pos_err={pnp['position_error']:.3f} "
                f"rot_err={pnp['rotation_error']:.3f}"
            )
    else:
        print(f"Feature Run failed: {pnp['failure_reason']}")

    attempt = {
        "true_pose": actual_pose,
        "estimated_pose": pnp["estimated_pose"],
        "success": pnp["success"],
        "low_confidence": pnp["low_confidence"],
        "quality_warnings": pnp["quality_warnings"],
        "failure_reason": pnp["failure_reason"],
        "query_keypoint_count": metrics["query_keypoint_count"],
        "good_descriptor_match_count": metrics["good_descriptor_match_count"],
        "pnp_inlier_count": pnp["pnp_inlier_count"],
        "reprojection_error": pnp["reprojection_error"],
        "position_error": pnp["position_error"],
        "rotation_error": pnp["rotation_error"],
        "true_image": query_bgr,
        "estimated_image": estimated_image,
        "diff_image": diff_image,
    }
    feature_run_attempts.append(attempt)
    feature_current_pair_index = len(feature_run_attempts) - 1
    if pnp["success"]:
        feature_est_view_mats[feature_current_pair_index] = pnp["view_matrix"]
        feature_overlay_active = True
    else:
        feature_overlay_active = False


def browse_feature_attempt(delta):
    global feature_current_pair_index, feature_overlay_active
    if not feature_run_attempts:
        return
    feature_current_pair_index = (
        feature_current_pair_index + delta
    ) % len(feature_run_attempts)
    attempt = feature_run_attempts[feature_current_pair_index]
    c_x, c_y, c_z, r_x, r_y, r_z = attempt["true_pose"]
    globals()["c_x"], globals()["c_y"], globals()["c_z"] = c_x, c_y, c_z
    globals()["r_x"], globals()["r_y"], globals()["r_z"] = r_x, r_y, r_z
    feature_overlay_active = bool(attempt["success"])
    status = "success" if attempt["success"] else f"failed: {attempt['failure_reason']}"
    print(
        f"Feature attempt {feature_current_pair_index + 1}/"
        f"{len(feature_run_attempts)}: {status}"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(argv=None):
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
    global feature_run_attempts, feature_diff_mode, active_feature_db
    global feature_pre_db, feature_pre_active_index, feature_pre_pending
    global feature_pre_show_keypoints, feature_pre_session_added
    global feature_pre_demo, feature_pre_save_confirm
    global feature_pre_candidate_cache
    global running
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", action="store_true", help="launch Feature Pre Mode")
    args = parser.parse_args(argv)

    CONFIG = read_config()
    ACTIVE_MAP = select_map_profile()

    pre_action = None
    if args.pre:
        pre_action = select_feature_pre_action(ACTIVE_MAP["name"])
        if pre_action == "cancel":
            print("Feature Pre Mode cancelled.")
            return

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
    feature_run_attempts        = []
    feature_diff_mode           = False
    active_feature_db           = None
    feature_pre_db              = None
    feature_pre_active_index    = 0
    feature_pre_pending         = None
    feature_pre_show_keypoints  = True
    feature_pre_session_added   = []
    feature_pre_demo            = (pre_action == "demo")
    feature_pre_save_confirm    = False
    feature_pre_candidate_cache = {}
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

    view_size = (display[0] // 2, display[1])
    if args.pre:
        loaded_db, db_path, db_errors = load_active_database(
            ACTIVE_MAP, image.shape, view_size)
        if pre_action == "continue":
            if loaded_db is None or db_errors:
                print("Feature database does not match selected map/settings.")
                for error in db_errors:
                    print(f"  {error}")
                print("Starting with a new empty in-memory database.")
                feature_pre_db = FeatureDatabase.empty(ACTIVE_MAP, image.shape, view_size)
            else:
                feature_pre_db = loaded_db
                print(
                    f"Continuing feature database: {feature_pre_db.view_count()} views, "
                    f"{feature_pre_db.mapped_feature_count()} mapped features."
                )
        elif pre_action == "demo":
            if loaded_db is not None and not db_errors:
                feature_pre_db = loaded_db.clone()
                print(
                    f"Demo mode loaded active database copy: "
                    f"{feature_pre_db.view_count()} views, "
                    f"{feature_pre_db.mapped_feature_count()} mapped features."
                )
            else:
                feature_pre_db = FeatureDatabase.empty(ACTIVE_MAP, image.shape, view_size)
                print("Demo mode using a temporary empty database.")
        else:
            feature_pre_db = FeatureDatabase.empty(ACTIVE_MAP, image.shape, view_size)
            print("Created a new empty in-memory feature database.")
        if feature_pre_db.views:
            feature_pre_active_index = 0
            c_x2, c_y2, c_z2, r_x2, r_y2, r_z2 = feature_pre_db.views[0].pose
    else:
        loaded_db, db_path, db_errors = load_active_database(
            ACTIVE_MAP, image.shape, view_size)
        if loaded_db is not None and not db_errors:
            active_feature_db = loaded_db
            print(
                f"Loaded feature database for {ACTIVE_MAP['name']}: "
                f"{active_feature_db.view_count()} views, "
                f"{active_feature_db.mapped_feature_count()} mapped features."
            )
        elif loaded_db is None:
            print(f"No feature database found for {ACTIVE_MAP['name']}.")
            print("Use: python world_split_v5.17.py --pre")
        else:
            print("Feature database does not match selected map/settings.")
            for error in db_errors:
                print(f"  {error}")
            print("Run: python world_split_v5.17.py --pre")

    running = True
    if args.pre:
        feature_pre_last_time = time.perf_counter()
        while running:
            now = time.perf_counter()
            dt = now - feature_pre_last_time
            feature_pre_last_time = now
            if dt > FEATURE_PRE_RESET_DT_SECONDS:
                dt = 0.0
            motion_scale = dt * 60.0
            update_window_caption("pre")
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                if (hasattr(pygame, "WINDOWFOCUSGAINED") and
                        event.type == pygame.WINDOWFOCUSGAINED):
                    feature_pre_last_time = time.perf_counter()
                if event.type == VIDEORESIZE:
                    resize(event.w, event.h)
                if event.type == KEYDOWN:
                    mods = pygame.key.get_mods()
                    if feature_pre_save_confirm:
                        if event.key == K_y:
                            save_feature_pre_active()
                            feature_pre_save_confirm = False
                        elif event.key == K_n or event.key == K_ESCAPE:
                            print("Save cancelled.")
                            feature_pre_save_confirm = False
                        continue
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_v:
                        sw, sh = pygame.display.get_surface().get_size()
                        create_feature_reference_view(sw // 2, sh)
                    elif event.key == K_LEFTBRACKET and feature_pre_db.views:
                        feature_pre_active_index = (
                            feature_pre_active_index - 1
                        ) % feature_pre_db.view_count()
                        feature_pre_pending = None
                    elif event.key == K_RIGHTBRACKET and feature_pre_db.views:
                        feature_pre_active_index = (
                            feature_pre_active_index + 1
                        ) % feature_pre_db.view_count()
                        feature_pre_pending = None
                    elif event.key == K_h:
                        feature_pre_show_keypoints = not feature_pre_show_keypoints
                        print(
                            "Detected keypoints visible: "
                            f"{feature_pre_show_keypoints}"
                        )
                    elif event.key == K_u:
                        undo_feature_pre_mapping()
                    elif event.key == K_s and (mods & KMOD_CTRL):
                        if feature_pre_demo:
                            print("Demo mode: changes are temporary and will not be saved.")
                        else:
                            feature_pre_save_confirm = True
                            print("Save changes to active database? Y = save, N = cancel")
                    elif event.key == K_p and (mods & KMOD_CTRL):
                        save_feature_pre_snapshot()
                    elif event.key == K_a and (mods & KMOD_CTRL):
                        sw, sh = pygame.display.get_surface().get_size()
                        augment_feature_pre_db_with_asift(sw // 2, sh)
                if event.type == MOUSEBUTTONDOWN:
                    sw, sh = pygame.display.get_surface().get_size()
                    half_w = sw // 2
                    if event.button == 3 and event.pos[0] >= half_w:
                        select_nearest_feature(event.pos[0] - half_w, event.pos[1])
                    elif event.button == 1 and event.pos[0] < half_w:
                        world_point = get_world_coords(event.pos[0], event.pos[1],
                                                       view="left")
                        if world_point is None:
                            print("Left click missed terrain.")
                        else:
                            complete_pending_feature_mapping(world_point)
            draw(recording_mode, feature_pre_mode=True,
                 motion_scale=motion_scale)
        glDeleteBuffers(1, [terrain_vbo])
        glDeleteBuffers(1, [pyramid_vbo])
        if _sphere_quadric is not None:
            gluDeleteQuadric(_sphere_quadric)
        pygame.quit()
        if feature_pre_demo:
            print("Demo mode: discarded temporary feature database changes.")
        return

    normal_mode_last_time = time.perf_counter()
    while running:
        now = time.perf_counter()
        dt = now - normal_mode_last_time
        normal_mode_last_time = now
        if dt > FEATURE_PRE_RESET_DT_SECONDS:
            dt = 0.0
        motion_scale = dt * 60.0

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            if (hasattr(pygame, "WINDOWFOCUSGAINED") and
                    event.type == pygame.WINDOWFOCUSGAINED):
                normal_mode_last_time = time.perf_counter()

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
                        sw, sh = pygame.display.get_surface().get_size()
                        record_feature_run_attempt(sw // 2, sh)
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
                        # Snap right view to the first saved position (overview)
                        if saved_positions:
                            c_x2, c_y2, c_z2, r_x2, r_y2, r_z2 = saved_positions[0]
                            print("Feature Run: right view snapped to starting position")
                        if active_feature_db is None:
                            print("Feature Run: no valid database loaded; B will record a failed attempt.")
                    else:
                        if not picking_mode:
                            recording_mode = True
                    print(f"Feature Run mode: {'on' if feature_mode else 'off'}")
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
                if event.key == K_n and feature_mode and feature_run_attempts:
                    browse_feature_attempt(-1)
                if event.key == K_m and feature_mode and feature_run_attempts:
                    browse_feature_attempt(1)
                if event.key == K_o and feature_mode:
                    feature_diff_mode = not feature_diff_mode
                    print("Feature comparison mode:", "diff" if feature_diff_mode else "overlay")
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
        update_window_caption("normal")
        draw(recording_mode, trackers_mode, feature_mode,
             motion_scale=motion_scale)

    glDeleteBuffers(1, [terrain_vbo])
    glDeleteBuffers(1, [pyramid_vbo])
    if _sphere_quadric is not None:
        gluDeleteQuadric(_sphere_quadric)
    pygame.quit()


if __name__ == "__main__":
    main()
