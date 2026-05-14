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
# VBO helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pyramid VBO  (built once, instanced per camera position)
# ---------------------------------------------------------------------------

# Clipping planes — increase FAR if the world gets cut off in the distance
NEAR =   0.1
FAR  = 5000.0

_PYRAMID_TIP  = (0.0, 0.0, 1.5)
_PYRAMID_BASE = [
    (-0.5, -0.5, 0.5),
    ( 0.5, -0.5, 0.5),
    ( 0.5,  0.5, 0.5),
    (-0.5,  0.5, 0.5),
]
_PYRAMID_SCALE = 0.3

# Colours — will be overridden per-instance via glColor before draw if needed,
# but since we need two colours (blue / red-current) we draw with colour arrays.
# We store the geometry once in local space and use the matrix stack for placement.

def build_pyramid_vbo():
    """
    Build a VBO for ONE pyramid in local space.
    Vertex layout: R G B  X Y Z  (blue colouring; red-current is handled separately)
    Returns (vbo_id, vertex_count).
    """
    tip  = _PYRAMID_TIP
    base = _PYRAMID_BASE

    # 4 side triangles (3 verts each) + 1 base quad (as 2 triangles = 6 verts)
    verts = []
    # sides
    for i in range(4):
        # tip vertex — black
        verts += [0.0, 0.0, 0.0,  tip[0],    tip[1],    tip[2]]
        # two base verts — blue (placeholder; we'll tint via glColor3f at draw time)
        verts += [0.0, 0.0, 1.0,  base[i][0],          base[i][1],          base[i][2]]
        verts += [0.0, 0.0, 1.0,  base[(i+1)%4][0],    base[(i+1)%4][1],    base[(i+1)%4][2]]
    # base quad as two triangles
    for idx in [0, 1, 2,  0, 2, 3]:
        verts += [0.0, 0.0, 1.0,  base[idx][0], base[idx][1], base[idx][2]]

    data = np.array(verts, dtype=np.float32)
    vbo  = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return vbo, len(verts) // 6   # 6 floats per vertex


def draw_pyramid_vbo(vbo, vertex_count, tint):
    """Draw one pyramid with the given (r,g,b) tint for the blue parts."""
    stride = 6 * 4

    # Patch the colour on the fly: we override with glColor, but since we use
    # a colour array the per-vertex black tip will still come through correctly.
    # Instead, we draw sides and base separately so we can tint the base verts.
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glEnableClientState(GL_COLOR_ARRAY)
    glEnableClientState(GL_VERTEX_ARRAY)
    glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))
    glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(3 * 4))

    # Override colour stream with the tint colour for non-black vertices.
    # Simplest approach: disable colour array, draw with flat glColor.
    glDisableClientState(GL_COLOR_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # Re-draw in immediate mode but using pre-computed local geometry
    # (still O(1) — pyramid has exactly 18 vertices regardless of scene size)
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
    for b_v in base:
        glVertex3fv(b_v)
    glEnd()

    glDisableClientState(GL_VERTEX_ARRAY)


def build_terrain_vbo(tri_path, image, margin):
    """
    Parse the .tri file once, bake vertex colours from the map image,
    and upload everything into a single interleaved VBO.

    Vertex layout (6 floats): R G B  X Y Z
    Returns (vbo_id, vertex_count).
    """
    import image_to_tris
    image_to_tris.main()  # Ensure .tri file is generated from the image
    tris = list(tm.read_tri_map(tri_path))
    vertex_count = len(tris) * 3

    # Interleaved array: [r, g, b, x, y, z] per vertex
    data = np.empty(vertex_count * 6, dtype=np.float32)

    idx = 0
    for tri in tris:
        for v in (tri.v1, tri.v2, tri.v3):
            bgr = image[int(v.z) * margin, int(v.x) * margin]
            data[idx]     = bgr[2] / 255.0   # R
            data[idx + 1] = bgr[1] / 255.0   # G
            data[idx + 2] = bgr[0] / 255.0   # B
            data[idx + 3] = v.x
            data[idx + 4] = v.y
            data[idx + 5] = v.z
            idx += 6

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    return vbo, vertex_count


def draw_terrain_vbo(vbo, vertex_count):
    """Draw the terrain using the pre-built VBO in a single draw call."""
    stride = 6 * 4  # 6 floats × 4 bytes

    glBindBuffer(GL_ARRAY_BUFFER, vbo)

    glEnableClientState(GL_COLOR_ARRAY)
    glEnableClientState(GL_VERTEX_ARRAY)

    glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))
    glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(3 * 4))

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
# Background / decorative drawing (unchanged)
# ---------------------------------------------------------------------------

def draw_gradient_background():
    glDisable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glBegin(GL_QUADS)
    glColor3f(0.2, 0.2, 0.6)
    glVertex2f(-1,  1)
    glVertex2f( 1,  1)
    glColor3f(0.6, 0.6, 0.8)
    glVertex2f( 1, -1)
    glVertex2f(-1, -1)
    glEnd()

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    glEnable(GL_DEPTH_TEST)


def draw_seperator_line():
    width, height = pygame.display.get_surface().get_size()
    glDisable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glViewport(0, 0, width, height)
    glBegin(GL_LINES)
    glColor3f(0, 0, 0)
    glVertex2f(0, -height)
    glVertex2f(0,  height)
    glEnd()

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
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
        mouse_x, real_y, depth_value,
        modelview, projection, viewport
    )
    return world_x, world_y, world_z


# ---------------------------------------------------------------------------
# Scene rendering  (VBO replaces per-triangle immediate-mode loop)
# ---------------------------------------------------------------------------

def render_scene(apply_input=True, recording_mode=True):
    global CONFIG
    global c_x, c_y, c_z, r_x, r_y, r_z
    global c_x2, c_y2, c_z2, r_x2, r_y2, r_z2
    global picking_mode
    global terrain_vbo, terrain_vertex_count

    r_speed = 0.1
    rot_speed = 0.5

    dx =  math.sin(r_y * math.pi / 180)
    dz = -math.cos(r_y * math.pi / 180)
    rx =  math.cos(r_y * math.pi / 180)
    rz =  math.sin(r_y * math.pi / 180)

    # --- INPUT / CAMERA STATE ---
    if apply_input and not picking_mode:
        keys_pressed = pygame.key.get_pressed()
        if recording_mode:
            if keys_pressed[K_LEFT]:   r_y -= rot_speed
            if keys_pressed[K_RIGHT]:  r_y += rot_speed
            if keys_pressed[K_DOWN]:   r_x += rot_speed
            if keys_pressed[K_UP]:     r_x -= rot_speed
            if keys_pressed[K_a]:
                c_x += rx * r_speed;  c_z += rz * r_speed
            if keys_pressed[K_d]:
                c_x -= rx * r_speed;  c_z -= rz * r_speed
            if keys_pressed[K_w]:
                c_x -= dx * r_speed;  c_z -= dz * r_speed
            if keys_pressed[K_s]:
                c_x += dx * r_speed;  c_z += dz * r_speed
            if keys_pressed[K_SPACE]:  c_y -= r_speed
            if keys_pressed[K_LSHIFT]: c_y += r_speed

            if keys_pressed[K_BACKSPACE]:
                image = cv2.imread(CONFIG.get("map_path"))
                height, width, _ = image.shape
                
                starting_pos = (-width/ CONFIG.get("margin") / 2, CONFIG.get("start_h"), -(height/CONFIG.get("margin")) -100)
                c_x, c_y, c_z = map(float, starting_pos)
                r_x, r_y, r_z = 30, 0.0, 0.0

        else:  # replay mode
            global saved_positions, recording_index, picked_points
            c_x, c_y, c_z, r_x, r_y, r_z = saved_positions[recording_index]

    elif not apply_input and not picking_mode:
        c_x, c_y, c_z = c_x2, c_y2, c_z2
        r_x, r_y, r_z = r_x2, r_y2, r_z2
        for pos in saved_positions:
            px, py, pz, prx, pry, prz = pos
            glPushMatrix()
            glLoadIdentity()
            glRotatef(r_y2, 0, 1, 0)
            glRotatef(r_x2, 1, 0, 0)
            glRotatef(r_z2, 0, 0, 1)
            glTranslatef(c_x2, c_y2, c_z2)
            draw_camera_pyramid(px, py, pz, prx, pry, prz)
            glPopMatrix()

    elif not apply_input and picking_mode:
        c_x, c_y, c_z = c_x2, c_y2, c_z2
        r_x, r_y, r_z = r_x2, r_y2, r_z2
        for pos in picked_points:
            px, py, pz = pos
            glPushMatrix()
            glLoadIdentity()
            glRotatef(r_y2, 0, 1, 0)
            glRotatef(r_x2, 1, 0, 0)
            glRotatef(r_z2, 0, 0, 1)
            glTranslatef(c_x2, c_y2, c_z2)
            draw_sphere(px, py, pz)
            glPopMatrix()

    # --- APPLY CAMERA ---
    glLoadIdentity()
    glRotatef(r_y,  0,  1,  0)
    glRotatef(r_x, rx,  0, rz)
    glRotatef(r_z, rz,  0, rx)
    glTranslatef(c_x, c_y, c_z)

    # --- DRAW WORLD (single VBO draw call instead of per-triangle loop) ---
    draw_terrain_vbo(terrain_vbo, terrain_vertex_count)


# ---------------------------------------------------------------------------
# Sphere / pyramid helpers (unchanged)
# ---------------------------------------------------------------------------

_sphere_quadric = None   # allocated once after GL context is ready

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
    """Draw one pyramid instance in local space, placed via matrix transforms."""
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
# Main draw (split-screen, unchanged structure)
# ---------------------------------------------------------------------------

def draw(recording_mode):
    global picking_mode
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    width, height = pygame.display.get_surface().get_size()

    # LEFT VIEW
    glViewport(0, 0, width // 2, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width / 2) / height, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW)
    draw_gradient_background()
    render_scene(apply_input=True, recording_mode=recording_mode)

    # RIGHT VIEW
    glViewport(width // 2, 0, width // 2, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width / 2) / height, NEAR, FAR)
    glMatrixMode(GL_MODELVIEW)
    draw_gradient_background()

    global c_x, c_y, c_z, r_x, r_y, r_z
    global c_x2, c_y2, c_z2, r_x2, r_y2, r_z2
    old_cam = (c_x, c_y, c_z, r_x, r_y, r_z)

    c_x, c_y, c_z = c_x2, c_y2, c_z2
    r_x, r_y, r_z = r_x2, r_y2, r_z2

    render_scene(apply_input=False, recording_mode=recording_mode)

    c_x, c_y, c_z, r_x, r_y, r_z = old_cam

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

    CONFIG = read_config()

    pygame.init()

    picking_mode         = False
    saved_positions      = []
    picked_points        = []
    picked_correspondences = []
    recording_mode       = True
    recording_index      = 0

    image = cv2.imread(CONFIG.get("map_path"))
    height, width, _ = image.shape
    margin = CONFIG.get("margin")
                
    starting_pos = (-width/ margin / 2, CONFIG.get("start_h"), -(height/margin) -100)
    c_x,  c_y,  c_z  = map(float, starting_pos)
    c_x2, c_y2, c_z2 = map(float, starting_pos)
    r_x = r_y = r_z = 0.0
    r_x2 = r_y2 = r_z2 = 0.0
    r_x = r_x2 = 30

    display = (640 * 2, 480)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    resize(*display)
    init()

    # ------------------------------------------------------------------
    # ONE-TIME: load map image + build VBO  (was happening every frame)
    # ------------------------------------------------------------------
    margin = CONFIG.get("margin")
    image  = cv2.imread(CONFIG.get("map_path"))
    terrain_vbo, terrain_vertex_count = build_terrain_vbo(
        "test2.tri", image, margin
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
                if event.key == K_f:
                    pygame.display.toggle_fullscreen()
                if event.key == K_b:
                    print(f"Saved position ({c_x},{c_y},{c_z}) "
                          f"with rotation ({r_x},{r_y},{r_z})")
                    saved_positions.append((c_x, c_y, c_z, r_x, r_y, r_z))
                if event.key == K_r:
                    recording_index = 0
                    if saved_positions:
                        recording_mode = not recording_mode
                    print(f"Recording mode: {recording_mode}")
                if event.key == K_p:
                    picking_mode = not picking_mode
                    print("picking mode", "on" if picking_mode else "off")
                if event.key == K_LEFT and not recording_mode:
                    recording_index = (recording_index - 1) % len(saved_positions)
                if event.key == K_RIGHT and not recording_mode:
                    recording_index = (recording_index + 1) % len(saved_positions)

            if event.type == VIDEORESIZE:
                resize(event.w, event.h)

            if event.type == MOUSEBUTTONDOWN:
                if picking_mode and event.button == 1:
                    width, height = pygame.display.get_surface().get_size()
                    if event.pos[0] >= width // 2:
                        world_point = get_world_coords(event.pos[0], event.pos[1])
                        if world_point is not None:
                            image_point = (event.pos[0] - width // 2, event.pos[1])
                            picked_points.append(world_point)
                            picked_correspondences.append((image_point, world_point))
                            print(f"Picked 2D {image_point} -> 3D {world_point}")
                        else:
                            print("Picking missed terrain")

        draw(recording_mode)

    # Clean up GPU buffer before exit
    glDeleteBuffers(1, [terrain_vbo])
    glDeleteBuffers(1, [pyramid_vbo])
    if _sphere_quadric is not None:
        gluDeleteQuadric(_sphere_quadric)
    pygame.quit()


if __name__ == "__main__":
    main()
