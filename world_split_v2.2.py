import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import trimap_beta as tm
import math
import cv2
import numpy as np
import ctypes
from read_config import read_config

# ---------------------------------------------------------------------------
# Clipping planes
# ---------------------------------------------------------------------------
NEAR =   0.1
FAR  = 5000.0

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
_PYRAMID_SCALE = 0.3


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
def build_terrain_vbo(tri_path, image, margin):
    import image_to_tris
    image_to_tris.main()
    tris = list(tm.read_tri_map(tri_path))
    vertex_count = len(tris) * 3
    data = np.empty(vertex_count * 6, dtype=np.float32)
    idx = 0
    for tri in tris:
        for v in (tri.v1, tri.v2, tri.v3):
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
    glColor3f(0.2, 0.2, 0.6); glVertex2f(-1,  1); glVertex2f( 1,  1)
    glColor3f(0.6, 0.6, 0.8); glVertex2f( 1, -1); glVertex2f(-1, -1)
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
# Camera / picking helpers
# ---------------------------------------------------------------------------
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


def get_world_coords(mouse_x, mouse_y):
    width, height = pygame.display.get_surface().get_size()
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


def solve_pnp(picked_correspondences, view_w, view_h):
    """
    Estimate camera pose from N >= 1 2D-3D correspondences.
    Returns (success, cam_pos_world, (rx_deg, ry_deg, rz_deg), R, tvec)
    or      (False, None, None, None, None) on failure.

    Terrain points are nearly always coplanar (flat mesh), so SQPNP is avoided
    and EPNP / IPPE are preferred - they handle planar configurations correctly.

    1 point : position-only hint, no rotation.
    2 points: pad to 3 and use planar solvers.
    3 points: IPPE (planar) then EPNP.
    4+      : RANSAC(EPNP) then direct EPNP fallback, then LM refinement.
    """
    n = len(picked_correspondences)
    if n == 0:
        print("No points picked yet.")
        return False, None, None, None, None

    K    = build_camera_intrinsics(view_w, view_h)
    dist = np.zeros((4, 1))

    pts3d = np.array([[w[0], w[1], w[2]] for (_, w) in picked_correspondences],
                     dtype=np.float64)
    pts2d = np.array([[p[0], p[1]] for (p, _) in picked_correspondences],
                     dtype=np.float64)

    if n == 1:
        print("Only 1 point: position-only hint, orientation unknown.")
        cam_pos = pts3d[0].copy()
        return False, cam_pos, (0.0, 0.0, 0.0), np.eye(3), np.zeros((3, 1))

    if n == 2:
        # Pad to 3 with a duplicated point + tiny 2D jitter so solvers don't degenerate
        pts3d = np.vstack([pts3d, pts3d[[0]]])
        pts2d = np.vstack([pts2d, pts2d[[0]] + np.array([[0.5, 0.5]])])
        print("Warning: 2 points padded to 3 - result is very unreliable.")

    coplanar = _points_are_coplanar(pts3d)
    if coplanar:
        print("Points appear coplanar (expected for terrain) - using planar solvers.")

    if n <= 3:  # includes the padded-2 case
        # IPPE is the best planar solver; EPNP also handles planar; P3P for 3 non-planar
        solvers = ([(cv2.SOLVEPNP_IPPE,  "IPPE"),
                    (cv2.SOLVEPNP_EPNP,  "EPNP")]
                   if coplanar else
                   [(cv2.SOLVEPNP_P3P,   "P3P"),
                    (cv2.SOLVEPNP_EPNP,  "EPNP")])
        rvec, tvec = _try_solvers(pts3d, pts2d, K, dist, solvers)
        if rvec is None:
            print("All solvers failed.")
            return False, None, None, None, None

    else:
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
    print(f"actual cam pos={c_x2, c_y2, c_z2}  euler(deg)={r_x2, r_y2, r_z2}")
    pos_error = 0
    pos_error += math.pow(cam_pos[0]+c_x2, 2)
    pos_error += math.pow(cam_pos[1]+c_y2, 2)
    pos_error += math.pow(cam_pos[2]+c_z2, 2)
    pos_error = math.sqrt(pos_error)
    rot_error = 0
    rot_error += min(abs(euler[0]+r_x2), 360-abs(euler[0]+r_x2))
    rot_error += min(abs(euler[1]+r_y2), 360-abs(euler[1]+r_y2))
    rot_error += min(abs(euler[2]+r_z2), 360-abs(euler[2]+r_z2))
    print(f"PnP position error vs actual camera: {pos_error} units")
    print(f"PnP rotation error vs actual camera: {rot_error%360} degrees")
    return True, cam_pos, euler, R, tvec


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
      • Cyan sphere    - original picked world point
      • Magenta sphere - where the estimated camera's ray actually hits the terrain
      • Yellow line    - world-space error vector between them

    Also draws a larger magenta pyramid at the estimated camera position.
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

            # Magenta: where the ray from estimated camera hits terrain
            glPushMatrix()
            glTranslatef(rx, ry, rz)
            glColor3f(1.0, 0.0, 1.0)
            gluSphere(q, 0.25, 12, 6)
            glPopMatrix()

            # Yellow error line in world space
            glBegin(GL_LINES)
            glColor3f(1.0, 1.0, 0.0)
            glVertex3f(wx, wy, wz)
            glVertex3f(rx, ry, rz)
            glEnd()

    # Magenta ghost pyramid at estimated camera position
    if ok and euler_deg is not None:
        rx_deg, ry_deg, rz_deg = euler_deg
        px, py, pz = cam_pos
        glPushMatrix()
        glTranslatef(-px, -py, -pz)
        glRotatef(-ry_deg, 0, 1, 0)
        glRotatef(-rx_deg, 1, 0, 0)
        glRotatef(-rz_deg, 0, 0, 1)
        s = _PYRAMID_SCALE * 2
        glScalef(s, s, s)
        draw_pyramid_vbo(pyramid_vbo, pyramid_vertex_count, (1.0, 0.0, 1.0))
        glPopMatrix()

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


def draw_sphere(x, y, z):
    glPushMatrix()
    glTranslatef(x, y, z)
    glColor3f(1, 0, 0)
    gluSphere(get_sphere_quadric(), 0.3, 16, 8)
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


# ---------------------------------------------------------------------------
# Scene rendering
# ---------------------------------------------------------------------------
def render_scene(apply_input=True, recording_mode=True):
    global CONFIG
    global c_x, c_y, c_z, r_x, r_y, r_z
    global c_x2, c_y2, c_z2, r_x2, r_y2, r_z2
    global picking_mode
    global terrain_vbo, terrain_vertex_count

    r_speed   = 0.1
    rot_speed = 0.5

    dx =  math.sin(r_y * math.pi / 180)
    dz = -math.cos(r_y * math.pi / 180)
    rx =  math.cos(r_y * math.pi / 180)
    rz =  math.sin(r_y * math.pi / 180)

    if apply_input and not picking_mode:
        keys_pressed = pygame.key.get_pressed()
        if recording_mode:
            if keys_pressed[K_LEFT]:   r_y -= rot_speed
            if keys_pressed[K_RIGHT]:  r_y += rot_speed
            if keys_pressed[K_DOWN]:   r_x += rot_speed
            if keys_pressed[K_UP]:     r_x -= rot_speed
            if keys_pressed[K_a]:      c_x += rx * r_speed;  c_z += rz * r_speed
            if keys_pressed[K_d]:      c_x -= rx * r_speed;  c_z -= rz * r_speed
            if keys_pressed[K_w]:      c_x -= dx * r_speed;  c_z -= dz * r_speed
            if keys_pressed[K_s]:      c_x += dx * r_speed;  c_z += dz * r_speed
            if keys_pressed[K_SPACE]:  c_y -= r_speed
            if keys_pressed[K_LSHIFT]: c_y += r_speed
            if keys_pressed[K_BACKSPACE]:
                img = cv2.imread(CONFIG.get("map_path"))
                ih, iw, _ = img.shape
                m = CONFIG.get("margin")
                c_x, c_y, c_z = -iw/m/2, CONFIG.get("start_h"), -(ih/m)-100
                r_x, r_y, r_z = 30, 0.0, 0.0
        else:
            global saved_positions, recording_index, picked_points
            c_x, c_y, c_z, r_x, r_y, r_z = saved_positions[recording_index]

    elif not apply_input and not picking_mode:
        c_x, c_y, c_z = c_x2, c_y2, c_z2
        r_x, r_y, r_z = r_x2, r_y2, r_z2
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
        for pos in picked_points:
            px, py, pz = pos
            glPushMatrix()
            glLoadIdentity()
            glRotatef(r_y2, 0, 1, 0); glRotatef(r_x2, 1, 0, 0)
            glRotatef(r_z2, 0, 0, 1); glTranslatef(c_x2, c_y2, c_z2)
            draw_sphere(px, py, pz)
            glPopMatrix()

    glLoadIdentity()
    glRotatef(r_y,  0,  1,  0)
    glRotatef(r_x, rx,  0, rz)
    glRotatef(r_z, rz,  0, rx)
    glTranslatef(c_x, c_y, c_z)
    draw_terrain_vbo(terrain_vbo, terrain_vertex_count)


# ---------------------------------------------------------------------------
# Main draw (split-screen)
# ---------------------------------------------------------------------------
def draw(recording_mode):
    global picking_mode, pnp_result, picked_correspondences
    global c_x, c_y, c_z, r_x, r_y, r_z
    global c_x2, c_y2, c_z2, r_x2, r_y2, r_z2

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    width, height = pygame.display.get_surface().get_size()

    # LEFT VIEW
    glViewport(0, 0, width // 2, height)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(45, (width/2) / height, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW)
    draw_gradient_background()
    render_scene(apply_input=True, recording_mode=recording_mode)

    # RIGHT VIEW
    glViewport(width // 2, 0, width // 2, height)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(45, (width/2) / height, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW)
    draw_gradient_background()

    old_cam = (c_x, c_y, c_z, r_x, r_y, r_z)
    c_x, c_y, c_z = c_x2, c_y2, c_z2
    r_x, r_y, r_z = r_x2, r_y2, r_z2
    render_scene(apply_input=False, recording_mode=recording_mode)
    c_x, c_y, c_z, r_x, r_y, r_z = old_cam

    # World-space PnP overlay on top of the right view
    if picking_mode and pnp_result is not None:
        draw_pnp_world_overlay(pnp_result, picked_correspondences)

    draw_seperator_line()
    pygame.display.flip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    global CONFIG
    global c_x, c_y, c_z, r_x, r_y, r_z
    global c_x2, c_y2, c_z2, r_x2, r_y2, r_z2
    global saved_positions, recording_index, recording_mode
    global picking_mode, picked_points, picked_correspondences
    global terrain_vbo, terrain_vertex_count
    global pyramid_vbo, pyramid_vertex_count
    global pnp_result

    CONFIG = read_config()
    pygame.init()

    picking_mode           = False
    saved_positions        = []
    picked_points          = []
    picked_correspondences = []
    pnp_result             = None
    recording_mode         = True
    recording_index        = 0

    image = cv2.imread(CONFIG.get("map_path"))
    h, w, _ = image.shape
    margin   = CONFIG.get("margin")
    starting_pos = (-w/margin/2, CONFIG.get("start_h"), -(h/margin)-100)
    c_x,  c_y,  c_z  = map(float, starting_pos)
    c_x2, c_y2, c_z2 = map(float, starting_pos)
    r_x = r_y = r_z = 0.0
    r_x2 = r_y2 = r_z2 = 0.0
    r_x = r_x2 = 30

    display = (640*2, 480)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    resize(*display)
    init()

    image = cv2.imread(CONFIG.get("map_path"))
    terrain_vbo, terrain_vertex_count = build_terrain_vbo("test2.tri", image, margin)
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
                if event.key == K_f:
                    pygame.display.toggle_fullscreen()
                if event.key == K_b:
                    print(f"Saved ({c_x},{c_y},{c_z}) rot ({r_x},{r_y},{r_z})")
                    saved_positions.append((c_x, c_y, c_z, r_x, r_y, r_z))
                if event.key == K_r:
                    recording_index = 0
                    if saved_positions:
                        recording_mode = not recording_mode
                    print(f"Recording mode: {recording_mode}")
                if event.key == K_p:
                    picking_mode = not picking_mode
                    if not picking_mode:
                        pnp_result = None      # clear overlay when leaving picking mode
                    print("picking mode", "on" if picking_mode else "off")
                if event.key == K_c and picking_mode:
                    sw, sh = pygame.display.get_surface().get_size()
                    pnp_result = solve_pnp(picked_correspondences, sw // 2, sh)
                    status = "ok" if (pnp_result and pnp_result[0]) else "low-confidence"
                    print(f"PnP solved ({status}) - overlay active")
                if event.key == K_LEFT and not recording_mode:
                    recording_index = (recording_index - 1) % len(saved_positions)
                if event.key == K_RIGHT and not recording_mode:
                    recording_index = (recording_index + 1) % len(saved_positions)

            if event.type == VIDEORESIZE:
                resize(event.w, event.h)

            if event.type == MOUSEBUTTONDOWN:
                if picking_mode and event.button == 1:
                    sw, sh = pygame.display.get_surface().get_size()
                    if event.pos[0] >= sw // 2:
                        world_point = get_world_coords(event.pos[0], event.pos[1])
                        if world_point is not None:
                            image_point = (event.pos[0] - sw // 2, event.pos[1])
                            picked_points.append(world_point)
                            picked_correspondences.append((image_point, world_point))
                            print(f"Picked 2D {image_point} -> 3D {world_point}")
                        else:
                            print("Picking missed terrain")

        draw(recording_mode)

    glDeleteBuffers(1, [terrain_vbo])
    glDeleteBuffers(1, [pyramid_vbo])
    if _sphere_quadric is not None:
        gluDeleteQuadric(_sphere_quadric)
    pygame.quit()


if __name__ == "__main__":
    main()
