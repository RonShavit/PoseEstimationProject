import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import trimap_beta as tm
import math
from tqdm import tqdm
import cv2
from read_config import read_config

# --- Resize function (like ReSizeGLScene) ---
def resize(width, height):
    if height == 0:
        height = 1

    glViewport(0, 0, width, height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    gluPerspective(45, width / height, 0.1, 100.0)

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

# --- DrawGLScene ---
def render_scene(apply_input=True):
    global CONFIG
    global c_x, c_y, c_z
    global r_x, r_y, r_z
    global c_x2, c_y2, c_z2
    global r_x2, r_y2, r_z2

    r_speed = 1
    margin = CONFIG.get("margin")

    dx = math.sin(r_y * math.pi / 180)
    dz = -math.cos(r_y * math.pi / 180)
    rx = math.cos(r_y * math.pi / 180)
    rz = math.sin(r_y * math.pi / 180)

    # --- INPUT ONLY FOR LEFT VIEW ---
    if apply_input:
        keys_pressed = pygame.key.get_pressed()

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

    # --- APPLY CAMERA ---
    glLoadIdentity()
    glRotatef(r_y, 0, 1, 0)
    glRotatef(r_x, rx, 0, rz)
    glRotatef(r_z, rz, 0, rx)
    glTranslatef(c_x, c_y, c_z)

    # --- DRAW WORLD ---
    image = cv2.imread(CONFIG.get("map_path"))

    for tri in tm.read_tri_map("test2.tri"):
        glBegin(GL_TRIANGLES)

        for v in [tri.v1, tri.v2, tri.v3]:
            color = image[int(v.z) * margin, int(v.x) * margin]
            glColor3f(color[2]/255, color[1]/255, color[0]/255)
            glVertex3f(v.x, v.y, v.z)

        glEnd()


def draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    width, height = pygame.display.get_surface().get_size()

    # --- LEFT VIEW (NORMAL CAMERA) ---
    glViewport(0, 0, width // 2, height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width/2) / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    draw_gradient_background()
    render_scene(apply_input=True)

    # --- RIGHT VIEW (FIXED CAMERA) ---
    glViewport(width // 2, 0, width // 2, height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width/2) / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    draw_gradient_background()

    # Save current camera
    global c_x, c_y, c_z, r_x, r_y, r_z
    global c_x2, c_y2, c_z2, r_x2, r_y2, r_z2
    old_cam = (c_x, c_y, c_z, r_x, r_y, r_z)

    # --- FIXED CAMERA POSITION ---
    c_x, c_y, c_z = c_x2, c_y2, c_z2  # ← change this if you want
    r_x, r_y, r_z = r_x2, r_y2, r_z2      # ← nice overview angle

    render_scene(apply_input=False)

    # Restore original camera
    c_x, c_y, c_z, r_x, r_y, r_z = old_cam

    pygame.display.flip()
# --- Main ---
def main():
    global CONFIG
    CONFIG = read_config()
    pygame.init()
    global c_x, c_y, c_z
    global r_x, r_y, r_z
    global c_x2, c_y2, c_z2
    global r_x2, r_y2, r_z2
    c_x2, c_y2, c_z2  = -12, -3, -10 
    r_x2, r_y2, r_z2 = 30, 45, 0 
    
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
                    c_x2, c_y2, c_z2, r_x2, r_y2, r_z2 = c_x, c_y,c_z,r_x,r_y,r_z

            if event.type == VIDEORESIZE:
                resize(event.w, event.h)
                

                


        draw()

    pygame.quit()


if __name__ == "__main__":
    main()