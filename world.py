import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import trimap_beta as tm
import math
from tqdm import tqdm
import cv2

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

    glClearColor(0.0, 0.0, 0.0, 1.0)  # black background


# --- DrawGLScene ---
def draw():
    global c_x, c_y, c_z
    global r_x, r_y, r_z
    r_speed = 0.01

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    dx = math.sin(r_y*math.pi/180)
    dz = -math.cos(r_y*math.pi/180)
    rx = math.cos(r_y*math.pi/180)
    rz = math.sin(r_y*math.pi/180)
    
    keys_pressed = pygame.key.get_pressed()
    
    if keys_pressed[K_LEFT]:
        r_y -= 1
    if keys_pressed[K_RIGHT]:
        r_y += 1
    if keys_pressed[K_DOWN]:
        r_x += 1
    if keys_pressed[K_UP]:
        r_x -= 1
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
        c_x, c_y, c_z = 0.0, 0.0, -5.0
        r_x, r_y, r_z = 0.0, 0.0, 0.0
    


    # move camera back
    glTranslatef(0,0,0)
    glRotatef(r_y , 0, 1, 0)
    glRotatef(r_x, rx, 0, rz )
    glRotatef(r_z , rz, 0, rx)
    glTranslatef(c_x, c_y, c_z)
    image = cv2.imread("map_1.jpg")
    for tri in tqdm(tm.read_tri_map("test2.tri")):
        glBegin(GL_TRIANGLES)
        xcolor = image[int(tri.v1.y), int(tri.v1.x)]
        glColor3f(xcolor[0]/255, xcolor[1]/255, xcolor[2]/255)  # red color
        glVertex3f(tri.v1.x, tri.v1.y, tri.v1.z)
        xcolor = image[int(tri.v2.y), int(tri.v2.x)]
        glColor3f(xcolor[0]/255, xcolor[1]/255, xcolor[2]/255)  # red color
        glVertex3f(tri.v2.x, tri.v2.y, tri.v2.z)
        xcolor = image[int(tri.v3.y), int(tri.v3.x)]
        glColor3f(xcolor[0]/255, xcolor[1]/255, xcolor[2]/255)  # red color
        glVertex3f(tri.v3.x, tri.v3.y, tri.v3.z)
        glEnd()

    # (no geometry yet — same as original tutorial)

    pygame.display.flip()


# --- Main ---
def main():
    pygame.init()
    global c_x, c_y, c_z
    global r_x, r_y, r_z
    c_x, c_y, c_z = 0.0, 0.0, -5.0
    r_x, r_y, r_z = 0.0, 0.0, 0.0
    display = (640, 480)
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

            if event.type == VIDEORESIZE:
                resize(event.w, event.h)
                

                


        draw()

    pygame.quit()


if __name__ == "__main__":
    main()