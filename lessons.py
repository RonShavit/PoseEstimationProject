import math
import random

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import trimap_beta as tm

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

    glClearColor(0.0, 0.25, 0.5, 1.0)  # black background


# --- DrawGLScene ---
def draw(nx,ny,nz,rh, rv):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1), (0,0,0), (0.5,0.5,0.5), (1,0.5,0), (0.5,1,0), (0,0.5,1), (0.5,0,1), (1,0,0.5), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5), (1,1,0.5), (1,0.5,1), (0.5,1,1)]

    # move camera back
    glTranslatef(0,0,0)
    glRotatef(rh, 0, 1, 0)
    glRotatef(rv, 1, 0, 0)
    glTranslatef(nx, ny, nz)
    
    tris_list = tm.read_tri_map("test.tri")
    c = 0
    for tri in tris_list:
        glBegin(GL_TRIANGLES)
        glColor3f(colors[c][0], colors[c][1], colors[c][2])
        glVertex3f(tri.v1.x, tri.v1.y, tri.v1.z)
        glVertex3f(tri.v2.x, tri.v2.y, tri.v2.z)
        glVertex3f(tri.v3.x, tri.v3.y, tri.v3.z)
        glEnd()
        c = (c + 1) % len(colors)
    

    

    pygame.display.flip()


def dome(base:list[float], top:tuple[float]):
    for i in range(len(base)):
        glBegin(GL_TRIANGLES)
        glColor3f(1,1,1)
        glVertex3f(base[i][0], base[i][1], base[i][2])

        glVertex3f(base[(i + 1) % len(base)][0], base[(i + 1) % len(base)][1], base[(i + 1) % len(base)][2])

        glVertex3f(top[0], top[1], top[2])
        glEnd()
def ball(radius:float, segments:int, center:tuple[float]):
    for i in range(segments):
        lat0 = math.pi * (-0.5 + float(i) / segments)
        z0 = math.sin(lat0) * radius
        zr0 = math.cos(lat0) * radius

        lat1 = math.pi * (-0.5 + float(i + 1) / segments)
        z1 = math.sin(lat1) * radius
        zr1 = math.cos(lat1) * radius

        glBegin(GL_QUAD_STRIP)
        for j in range(segments + 1):
            lng = 2 * math.pi * float(j) / segments
            x = math.cos(lng)
            y = math.sin(lng)

            glColor3f((x + 1) / 2, (y + 1) / 2, (z0 + radius) / (2 * radius))
            glVertex3f(center[0] + x * zr0, center[1] + y * zr0, center[2] + z0)

            glColor3f((x + 1) / 2, (y + 1) / 2, (z1 + radius) / (2 * radius))
            glVertex3f(center[0] + x * zr1, center[1] + y * zr1, center[2] + z1)
        glEnd()
# --- Main ---
def main():
    nx, ny,nz = 0.0, 0.0, -5.0
    t_running = 0
    rh,rv = 0.0, 0.0
    j_vel = 0.0
    pygame.init()

    display = (640, 480)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    resize(*display)
    init()

    running = True

    while running:
        keys_held = pygame.key.get_pressed()
        mouse_held = pygame.mouse.get_pressed()
        r_speed = 0.01
        rha = rh * math.pi / 180
    # forward
        dx = math.sin(rha)
        dz = -math.cos(rha)

    # right (perpendicular)
        rx = math.cos(rha)
        rz = math.sin(rha)

                
        if keys_held[K_LSHIFT] or keys_held[K_RSHIFT]:
            r_speed = 0.02
        elif keys_held[K_LCTRL] or keys_held[K_RCTRL]:
            r_speed = 0.005
        else:
            r_speedspeed = 0.01
        if keys_held[K_s]:
            nx += dx * r_speed
            nz += dz * r_speed

        if keys_held[K_w]:
            nx -= dx * r_speed
            nz -= dz * r_speed

        if keys_held[K_d]:
            nx -= rx * r_speed
            nz -= rz * r_speed

        if keys_held[K_a]:
            nx += rx * r_speed
            nz += rz * r_speed
        if keys_held[K_SPACE]:
            j_vel = -2.0 if ny == 0 else j_vel
        if keys_held[K_LEFT]:
            rh -= 1
        if keys_held[K_RIGHT]:
            rh += 1
        if keys_held[K_UP]:
            rv -= 0.3
        if keys_held[K_DOWN]:
            rv += 0.3
        rv = max(-30, min(30, rv))


            
            
            
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
                
            
        t_running += 1
        print(f"t: {t_running}, j_vel: {j_vel}, ny: {ny}, rh: {rh}, rha: {rha}")
        head_height = math.sin(t_running * 0.001 * math.pi) * 0.001
        
        ny += j_vel * 0.01 
        ny = min(0, ny)
        j_vel += 1 * 0.01
        if ny == 0:
            j_vel = 0
        
        
        

        draw(nx, ny + head_height, nz,rh, rv)
        

    pygame.quit()


if __name__ == "__main__":
    main()