import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import trimap_beta as tm
import math
from tqdm import tqdm
import cv2
import numpy as np
from read_config import read_config

CAMERA_FOV_Y_DEGREES = 45.0

# --- Resize function (like ReSizeGLScene) ---
def resize(width, height):
    if height == 0:
        height = 1

    glViewport(0, 0, width, height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    gluPerspective(CAMERA_FOV_Y_DEGREES, width / height, 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


# --- InitGL ---
def init():
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)

    #glClearColor(0.0, 0.0, 0.0, 1.0)  # black background
    
    
def draw_gradient_background():
    glDisable(GL_DEPTH_TEST)  # draw in background

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glBegin(GL_QUADS)
    
    # Top color 
    glColor3f(0.2, 0.2, 0.2)
    glVertex2f(-1, 1)
    glVertex2f(1, 1)

    # Bottom color 
    glColor3f(0.6, 0.6, 0.6)
    glVertex2f(1, -1)
    glVertex2f(-1, -1)

    glEnd()
    


    # Restore state
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    


    glEnable(GL_DEPTH_TEST)

def draw_seperator_line():
    width, height = pygame.display.get_surface().get_size()
    glDisable(GL_DEPTH_TEST)  # draw in background

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
    glVertex2f(0, height)
    glEnd()

    # Restore state
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def draw_2d_pick_marker(point, color, size=7):
    x, y = point
    glColor3f(*color)
    glBegin(GL_LINES)
    glVertex2f(x - size, y)
    glVertex2f(x + size, y)
    glVertex2f(x, y - size)
    glVertex2f(x, y + size)
    glEnd()


def draw_left_view_pick_markers():
    width, height = pygame.display.get_surface().get_size()

    glViewport(0, 0, width // 2, height)
    glDisable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, width // 2, height, 0, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glLineWidth(2)
    for point in picked_2d_points:
        draw_2d_pick_marker(point, (0.0, 1.0, 0.0))

    if pending_2d_point is not None:
        draw_2d_pick_marker(pending_2d_point, (1.0, 1.0, 0.0), size=9)

    glLineWidth(1)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    glEnable(GL_DEPTH_TEST)


def setup_right_view_matrices(width, height):
    """Match the projection/modelview used by the right global view."""
    glViewport(width // 2, 0, width // 2, height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(CAMERA_FOV_Y_DEGREES, (width / 2) / height, 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glRotatef(r_y2, 0, 1, 0)
    glRotatef(r_x2, 1, 0, 0)
    glRotatef(r_z2, 0, 0, 1)
    glTranslatef(c_x2, c_y2, c_z2)


def get_world_coords(mouse_x, mouse_y):
    width, height = pygame.display.get_surface().get_size()
    setup_right_view_matrices(width, height)

    # Get viewport
    viewport = glGetIntegerv(GL_VIEWPORT)

    # Flip Y (OpenGL origin is bottom-left)
    real_y = height - mouse_y

    # Read depth at mouse position
    depth = glReadPixels(mouse_x, real_y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
    depth_value = depth[0][0]

    # Depth 1.0 means the click hit the far plane/background, not the terrain.
    if depth_value >= 1.0:
        return None

    # Get matrices
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)

    # Convert to world coords
    world_x, world_y, world_z = gluUnProject(
        mouse_x, real_y, depth_value,
        modelview, projection, viewport
    )

    return world_x, world_y, world_z


def load_terrain_resources():
    image = cv2.imread(CONFIG.get("map_path"))
    if image is None:
        raise RuntimeError(f"Could not load map image: {CONFIG.get('map_path')}")

    tris = list(tm.read_tri_map("test2.tri"))
    return image, tris


def clear_picked_correspondences():
    global pending_2d_point

    pending_2d_point = None
    picked_2d_points.clear()
    picked_3d_points.clear()
    picked_correspondences.clear()
    clear_computed_pose()


def clear_computed_pose():
    global computed_pose_exists
    global computed_rvec
    global computed_tvec
    global computed_rotation_matrix
    global computed_gl_rotation_matrix
    global computed_camera_center
    global computed_modelview_matrix
    global computed_recording_index
    global computed_reprojection_error
    global show_computed_view

    computed_pose_exists = False
    computed_rvec = None
    computed_tvec = None
    computed_rotation_matrix = None
    computed_gl_rotation_matrix = None
    computed_camera_center = None
    computed_modelview_matrix = None
    computed_recording_index = None
    computed_reprojection_error = None
    show_computed_view = False


def build_camera_matrix(view_width, view_height):
    fov_y_radians = math.radians(CAMERA_FOV_Y_DEGREES)
    focal = (view_height / 2.0) / math.tan(fov_y_radians / 2.0)
    return np.array([
        [focal, 0.0, view_width / 2.0],
        [0.0, focal, view_height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def validate_pnp_inputs(image_points, object_points, view_width, view_height):
    if len(image_points) < 4:
        print("Need at least 4 correspondences before solvePnP.")
        return False

    if not saved_positions:
        print("No recorded pose is selected.")
        return False

    if recording_index < 0 or recording_index >= len(saved_positions):
        print("Selected recorded pose index is invalid.")
        return False

    if not np.all(np.isfinite(image_points)) or not np.all(np.isfinite(object_points)):
        print("PnP input contains non-finite values.")
        return False

    for u, v in image_points:
        if u < 0 or u >= view_width or v < 0 or v >= view_height:
            print(f"2D point ({u}, {v}) is outside the left camera view.")
            return False

    return True


def compute_reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    errors = np.linalg.norm(projected_points - image_points, axis=1)
    return float(np.mean(errors))


def solve_pnp_from_picks():
    global computed_pose_exists
    global computed_rvec
    global computed_tvec
    global computed_rotation_matrix
    global computed_gl_rotation_matrix
    global computed_camera_center
    global computed_modelview_matrix
    global computed_recording_index
    global computed_reprojection_error

    width, height = pygame.display.get_surface().get_size()
    view_width = width // 2

    image_points = np.array(
        [image_point for image_point, _ in picked_correspondences],
        dtype=np.float64,
    )
    object_points = np.array(
        [world_point for _, world_point in picked_correspondences],
        dtype=np.float64,
    )

    if not validate_pnp_inputs(image_points, object_points, view_width, height):
        return False

    camera_matrix = build_camera_matrix(view_width, height)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error as error:
        clear_computed_pose()
        print(f"solvePnP failed: {error}")
        return False

    if not success:
        clear_computed_pose()
        print("solvePnP failed.")
        return False

    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # OpenCV camera coordinates are x-right, y-down, z-forward.
    # This OpenGL view uses x-right, y-up, z-backward, so flip y and z.
    cv_to_gl = np.diag([1.0, -1.0, -1.0])
    gl_rotation_matrix = cv_to_gl @ rotation_matrix
    gl_translation = cv_to_gl @ tvec.reshape(3)
    camera_center = -gl_rotation_matrix.T @ gl_translation

    modelview_matrix = np.eye(4, dtype=np.float64)
    modelview_matrix[:3, :3] = gl_rotation_matrix
    modelview_matrix[:3, 3] = gl_translation

    try:
        reprojection_error = compute_reprojection_error(
            object_points,
            image_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
    except cv2.error as error:
        reprojection_error = None
        print(f"Could not compute reprojection error: {error}")

    true_pose = saved_positions[recording_index]
    true_camera_center = np.array([-true_pose[0], -true_pose[1], -true_pose[2]], dtype=np.float64)
    position_error = float(np.linalg.norm(camera_center - true_camera_center))

    computed_pose_exists = True
    computed_rvec = rvec
    computed_tvec = tvec
    computed_rotation_matrix = rotation_matrix
    computed_gl_rotation_matrix = gl_rotation_matrix
    computed_camera_center = camera_center
    computed_modelview_matrix = modelview_matrix
    computed_recording_index = recording_index
    computed_reprojection_error = reprojection_error

    print("solvePnP succeeded")
    print(f"Correspondences: {len(picked_correspondences)}")
    print(f"Camera matrix K:\n{camera_matrix}")
    print(f"rvec:\n{rvec}")
    print(f"tvec:\n{tvec}")
    print(f"computed camera center: {tuple(camera_center)}")
    print(f"true saved pose tuple: {true_pose}")
    print(f"true camera center: {tuple(true_camera_center)}")
    print(f"position error: {position_error}")
    if reprojection_error is not None:
        print(f"average reprojection error: {reprojection_error} pixels")
    return True


# --- DrawGLScene ---
def render_scene(apply_input=True, recording_mode=True, camera_modelview_matrix=None):
    global CONFIG
    global c_x, c_y, c_z
    global r_x, r_y, r_z
    global c_x2, c_y2, c_z2
    global r_x2, r_y2, r_z2
    global picking_mode
    global picked_3d_points
    global saved_positions
    global recording_index
    global terrain_image
    global terrain_tris
    global computed_pose_exists
    global computed_camera_center
    global computed_gl_rotation_matrix
    global computed_recording_index

    r_speed = 1
    margin = CONFIG.get("margin")

    # --- INPUT ONLY FOR LEFT VIEW ---
    if camera_modelview_matrix is not None:
        pass
    elif apply_input:
        keys_pressed = pygame.key.get_pressed()
        if not recording_mode:
            c_x, c_y, c_z, r_x, r_y, r_z = saved_positions[recording_index]
        elif not picking_mode:
            dx = math.sin(r_y * math.pi / 180)
            dz = -math.cos(r_y * math.pi / 180)
            rx = math.cos(r_y * math.pi / 180)
            rz = math.sin(r_y * math.pi / 180)

            if keys_pressed[K_LEFT]:
                r_y -= 4
            if keys_pressed[K_RIGHT]:
                r_y += 4
            if keys_pressed[K_DOWN]:
                r_x += 4
            if keys_pressed[K_UP]:
                r_x -= 4
            if keys_pressed[K_a]:
                c_x += rx * r_speed
                c_z += rz * r_speed
            if keys_pressed[K_d]:
                c_x -= rx * r_speed
                c_z -= rz * r_speed
            if keys_pressed[K_w]:
                c_x -= dx * r_speed
                c_z -= dz * r_speed
            if keys_pressed[K_s]:
                c_x += dx * r_speed
                c_z += dz * r_speed
            if keys_pressed[K_SPACE]:
                c_y -= r_speed
            if keys_pressed[K_LSHIFT]:
                c_y += r_speed
            

            if keys_pressed[K_BACKSPACE]:
                starting_pos = CONFIG.get("start_pos").split(",")
                c_x, c_y, c_z = map(float, starting_pos)
                r_x, r_y, r_z = 0.0, 0.0, 0.0
    elif not apply_input and not picking_mode:        # Fixed camera for right view
        c_x, c_y, c_z = c_x2, c_y2, c_z2
        r_x, r_y, r_z = r_x2, r_y2, r_z2
        for pos in saved_positions:
            px, py, pz, prx, pry, prz = pos

            glPushMatrix()

    # --- UNDO current camera transform ---
            glLoadIdentity()

    # Apply the SAME camera as the right view (fixed camera)
            glRotatef(r_y2, 0, 1, 0)
            glRotatef(r_x2, 1, 0, 0)
            glRotatef(r_z2, 0, 0, 1)
            glTranslatef(c_x2, c_y2, c_z2)

    # --- NOW draw pyramid in world space ---
            draw_camera_pyramid(px, py, pz, prx, pry, prz)

            glPopMatrix()

    elif not apply_input and picking_mode:
        c_x, c_y, c_z = c_x2, c_y2, c_z2
        r_x, r_y, r_z = r_x2, r_y2, r_z2

        if saved_positions:
            true_pose = saved_positions[recording_index]
            glPushMatrix()
            glLoadIdentity()
            glRotatef(r_y2, 0, 1, 0)
            glRotatef(r_x2, 1, 0, 0)
            glRotatef(r_z2, 0, 0, 1)
            glTranslatef(c_x2, c_y2, c_z2)
            draw_camera_pyramid(*true_pose)
            glPopMatrix()

        for pos in picked_3d_points:
            px, py, pz = pos

            glPushMatrix()

    # --- UNDO current camera transform ---
            glLoadIdentity()

    # Apply the SAME camera as the right view (fixed camera)
            glRotatef(r_y2, 0, 1, 0)
            glRotatef(r_x2, 1, 0, 0)
            glRotatef(r_z2, 0, 0, 1)
            glTranslatef(c_x2, c_y2, c_z2)

    # --- NOW draw pyramid in world space ---
            draw_sphere(px, py, pz, color=(1, 0, 0), radius=0.3)

            glPopMatrix()

        if computed_pose_exists and computed_recording_index == recording_index:
            glPushMatrix()
            glLoadIdentity()
            glRotatef(r_y2, 0, 1, 0)
            glRotatef(r_x2, 1, 0, 0)
            glRotatef(r_z2, 0, 0, 1)
            glTranslatef(c_x2, c_y2, c_z2)
            draw_sphere(
                computed_camera_center[0],
                computed_camera_center[1],
                computed_camera_center[2],
                color=(1, 0, 1),
                radius=0.55,
            )
            draw_pose_axes(computed_camera_center, computed_gl_rotation_matrix)
            glPopMatrix()

    # --- APPLY CAMERA ---
    if camera_modelview_matrix is not None:
        glLoadMatrixd(np.ascontiguousarray(camera_modelview_matrix.T, dtype=np.float64))
    else:
        rx_axis = math.cos(r_y * math.pi / 180)
        rz_axis = math.sin(r_y * math.pi / 180)
        glLoadIdentity()
        glRotatef(r_y, 0, 1, 0)
        glRotatef(r_x, rx_axis, 0, rz_axis)
        glRotatef(r_z, rz_axis, 0, rx_axis)
        glTranslatef(c_x, c_y, c_z)

    # --- DRAW WORLD ---
    for tri in terrain_tris:
        glBegin(GL_TRIANGLES)

        for v in [tri.v1, tri.v2, tri.v3]:
            color = terrain_image[int(v.z) * margin, int(v.x) * margin]
            glColor3f(color[2]/255, color[1]/255, color[0]/255)
            glVertex3f(v.x, v.y, v.z)

        glEnd()
def draw_sphere(x, y, z, color=(1, 0, 0), radius=0.3):
    glPushMatrix()
    glTranslatef(x,y,z)
    glColor3f(*color)
    quad = gluNewQuadric()
    gluSphere(quad, radius, 32, 4)
    gluDeleteQuadric(quad)
    glPopMatrix()


def draw_pose_axes(camera_center, gl_rotation_matrix, scale=2.0):
    center = np.array(camera_center, dtype=np.float64)
    rotation_inv = gl_rotation_matrix.T
    axis_specs = [
        (rotation_inv @ np.array([1.0, 0.0, 0.0]), (1.0, 0.0, 0.0)),
        (rotation_inv @ np.array([0.0, 1.0, 0.0]), (0.0, 1.0, 0.0)),
        (rotation_inv @ np.array([0.0, 0.0, -1.0]), (1.0, 1.0, 0.0)),
    ]

    glLineWidth(3)
    glBegin(GL_LINES)
    for direction, color in axis_specs:
        end = center + direction * scale
        glColor3f(*color)
        glVertex3f(*center)
        glVertex3f(*end)
    glEnd()
    glLineWidth(1)
    
def draw_camera_pyramid(x, y, z, rx, ry, rz, scale=0.3):
    global recording_index
    global recording_mode
    global saved_positions
    glPushMatrix()

    # Move to camera position
    glTranslatef(-x,-y,-z)

    # Apply same rotations as camera
    glRotatef(-ry, 0, 1, 0)
    glRotatef(-rx, 1, 0, 0)
    glRotatef(-rz, 0, 0, 1)
    

    glScalef(scale, scale, scale)

    glBegin(GL_TRIANGLES)
    # Tip of pyramid (forward direction)
    tip = (0, 0, 1.5)

    # Base square
    base = [
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]


    # 4 side triangles
    for i in range(4):
        glColor3f(0,0,0)  # black edges
        glVertex3fv(tip)
        
        if not recording_mode and saved_positions[recording_index] == (x,y,z,rx,ry,rz): # blue cameras
            glColor3f(1, 0, 0)  # red
        else:
            glColor3f(0, 0, 1)  # blue
        glVertex3fv(base[i])
        glVertex3fv(base[(i + 1) % 4])

    glEnd()

    glBegin(GL_QUADS)
    if not recording_mode and saved_positions[recording_index] == (x,y,z,rx,ry,rz): # blue cameras
        glColor3f(1, 0, 0)  # red
    else:
        glColor3f(0, 0, 1)  # blue
    for i in range(4):
        glVertex3fv(base[i])
    glEnd()
    glPopMatrix()

def draw(recording_mode):
    global picking_mode
    global show_computed_view
    global computed_pose_exists
    global computed_modelview_matrix
    global computed_recording_index
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    width, height = pygame.display.get_surface().get_size()
    


    # --- LEFT VIEW (NORMAL CAMERA) ---
    glViewport(0, 0, width // 2, height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(CAMERA_FOV_Y_DEGREES, (width/2) / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    draw_gradient_background()
    if (
        picking_mode
        and show_computed_view
        and computed_pose_exists
        and computed_recording_index == recording_index
    ):
        render_scene(
            apply_input=False,
            recording_mode=recording_mode,
            camera_modelview_matrix=computed_modelview_matrix,
        )
    else:
        render_scene(apply_input=True,recording_mode=recording_mode)
    if picking_mode:
        draw_left_view_pick_markers()

    # --- RIGHT VIEW (FIXED CAMERA) ---
    glViewport(width // 2, 0, width // 2, height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(CAMERA_FOV_Y_DEGREES, (width/2) / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    draw_gradient_background()

    # Save current camera
    global c_x, c_y, c_z, r_x, r_y, r_z
    global c_x2, c_y2, c_z2, r_x2, r_y2, r_z2
    old_cam = (c_x, c_y, c_z, r_x, r_y, r_z)

    # --- FIXED CAMERA POSITION ---
    c_x, c_y, c_z = c_x2, c_y2, c_z2  # ← change this if you want
    r_x, r_y, r_z = r_x2, r_y2, r_z2      # ← nice overview angle

    render_scene(apply_input=False,recording_mode=recording_mode)

    # Restore original camera
    c_x, c_y, c_z, r_x, r_y, r_z = old_cam
    
    draw_seperator_line()
    


    pygame.display.flip()
# --- Main ---
def main():
    global CONFIG
    CONFIG = read_config()
    global terrain_image
    global terrain_tris
    terrain_image, terrain_tris = load_terrain_resources()
    pygame.init()
    global c_x, c_y, c_z
    global r_x, r_y, r_z
    global c_x2, c_y2, c_z2
    global r_x2, r_y2, r_z2
    global saved_positions
    global recording_index
    global recording_mode
    global picking_mode
    global pending_2d_point
    global picked_2d_points
    global picked_3d_points
    global picked_correspondences
    global computed_pose_exists
    global computed_rvec
    global computed_tvec
    global computed_rotation_matrix
    global computed_gl_rotation_matrix
    global computed_camera_center
    global computed_modelview_matrix
    global computed_recording_index
    global computed_reprojection_error
    global show_computed_view
    picking_mode = False
    saved_positions = []
    pending_2d_point = None
    picked_2d_points = []
    picked_3d_points = []
    picked_correspondences = []
    clear_computed_pose()
    recording_mode = True
    starting_pos = CONFIG.get("start_pos").split(",")
    c_x2, c_y2, c_z2 = map(float, starting_pos)
    r_x2, r_y2, r_z2 = 0, 0, 0 
    
    starting_pos = CONFIG.get("start_pos").split(",")
    c_x, c_y, c_z = float(starting_pos[0]), float(starting_pos[1]), float(starting_pos[2])
    r_x, r_y, r_z = 0.0, 0.0, 0.0
    display = (640*2, 480)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    resize(*display)
    init()

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
                    #c_x2, c_y2, c_z2, r_x2, r_y2, r_z2 = c_x, c_y,c_z,r_x,r_y,r_z
                    print(f"Saved position ({c_x},{c_y},{c_z}) with rotation ({r_x},{r_y},{r_z})")
                    saved_positions.append((c_x, c_y, c_z, r_x, r_y, r_z))
                if event.key == K_r:
                    if saved_positions:
                        recording_index = 0
                        recording_mode = not recording_mode
                        if recording_mode and picking_mode:
                            picking_mode = False
                            pending_2d_point = None
                            clear_computed_pose()
                            print("picking mode off")
                        print(f"Recording mode: {recording_mode}")
                    else:
                        print("Save at least one pose with B before switching to playback mode")
                
                if event.key == K_p:
                    if recording_mode:
                        print("Switch to playback mode first using R, then enter picking mode")
                    else:
                        picking_mode = not picking_mode
                        if not picking_mode:
                            pending_2d_point = None
                            clear_computed_pose()
                        print("picking mode", "on" if picking_mode else "off")

                if event.key == K_x and picking_mode:
                    clear_picked_correspondences()
                    print("Cleared all picked correspondences")

                if event.key == K_c and picking_mode:
                    solve_pnp_from_picks()

                if event.key == K_v and picking_mode:
                    if computed_pose_exists and computed_recording_index == recording_index:
                        show_computed_view = not show_computed_view
                        print(
                            "Left camera view:",
                            "computed PnP view" if show_computed_view else "true recorded view",
                        )
                    else:
                        print("Compute a pose with C before toggling the computed view")
                    
                if event.key == K_LEFT and not recording_mode:
                    previous_index = recording_index
                    recording_index = (recording_index-1)% len(saved_positions)
                    if picking_mode and recording_index != previous_index:
                        clear_picked_correspondences()
                        print("Switched recorded pose; cleared picked correspondences")
                if event.key == K_RIGHT and not recording_mode:
                    previous_index = recording_index
                    recording_index = (recording_index+1) % len(saved_positions)
                    if picking_mode and recording_index != previous_index:
                        clear_picked_correspondences()
                        print("Switched recorded pose; cleared picked correspondences")

            if event.type == VIDEORESIZE:
                resize(event.w, event.h)
                
            if event.type == MOUSEBUTTONDOWN:
                if picking_mode and event.button == 1:
                    width, height = pygame.display.get_surface().get_size()
                    if event.pos[0] < width // 2:
                        pending_2d_point = (event.pos[0], event.pos[1])
                        print(f"Selected 2D image point {pending_2d_point}. Now click the matching 3D point in the right view.")
                    else:
                        if pending_2d_point is None:
                            print("Choose a 2D point in the left camera view first")
                            continue

                        world_point = get_world_coords(event.pos[0], event.pos[1])
                        if world_point is not None:
                            image_point = pending_2d_point
                            picked_2d_points.append(image_point)
                            picked_3d_points.append(world_point)
                            picked_correspondences.append((image_point, world_point))
                            pending_2d_point = None
                            clear_computed_pose()

                            count = len(picked_correspondences)
                            print(f"Picked correspondence #{count}: 2D {image_point} -> 3D {world_point}")
                            if count < 4:
                                print(f"Pick {4 - count} more correspondence(s) before solvePnP")
                            else:
                                print("You have 4+ correspondences, enough for solvePnP")
                        else:
                            print("Picking missed terrain")

                if picking_mode and event.button == 3 and pending_2d_point is not None:
                    pending_2d_point = None
                    print("Canceled pending 2D point")
                        
                

                

        
        draw(recording_mode)

    pygame.quit()


if __name__ == "__main__":
    main()
