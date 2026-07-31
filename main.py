import argparse
import math
from collections import deque
import colorsys
import numpy as np
import pygame
from pygame.locals import (
    DOUBLEBUF, OPENGL, RESIZABLE, QUIT, KEYDOWN, K_ESCAPE, K_p, K_r,
    K_w, K_s, K_a, K_d, K_SPACE, K_LCTRL, K_LSHIFT, MOUSEMOTION, MOUSEWHEEL,
    VIDEORESIZE,
)
from OpenGL.GL import *
from OpenGL.GLU import *

class NBodySimulation:
    def __init__(self, masses, positions, velocities, G=1.0, softening=0.2):
        self.masses = np.asarray(masses, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        self.velocities = np.asarray(velocities, dtype=float)
        self.G = G
        self.softening = softening
        self.n = len(self.masses)

    def accelerations(self, positions):
        r = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        dist_sqr = np.sum(r ** 2, axis=2) + self.softening ** 2
        inv_dist3 = dist_sqr ** (-1.5)
        np.fill_diagonal(inv_dist3, 0.0)
        acc = self.G * np.sum(
            r * inv_dist3[:, :, np.newaxis] * self.masses[np.newaxis, :, np.newaxis],
            axis=1,
        )
        return acc

    def step(self, dt):
        acc1 = self.accelerations(self.positions)
        self.velocities += 0.5 * acc1 * dt
        self.positions += self.velocities * dt
        acc2 = self.accelerations(self.positions)
        self.velocities += 0.5 * acc2 * dt

def preset_figure_eight():
    masses = [1.0, 1.0, 1.0]
    positions = [
        [0.9700436, -0.24308753, 0.0],
        [-0.9700436, 0.24308753, 0.0],
        [0.0, 0.0, 0.0],
    ]
    v = [0.466203685, 0.43236573, 0.0]
    velocities = [v, v, [-2 * v[0], -2 * v[1], 0.0]]
    dt = 0.003
    return masses, positions, velocities, dt


def preset_random(n=20, seed=None):
    rng = np.random.default_rng(seed)
    masses = rng.uniform(0.5, 3.0, size=n)
    positions = rng.uniform(-6.0, 6.0, size=(n, 3))
    velocities = rng.normal(scale=0.3, size=(n, 3))
    velocities -= np.average(velocities, axis=0, weights=masses)
    dt = 0.01
    return masses, positions, velocities, dt


def preset_solar():
    masses = [1000.0, 1.0, 1.5, 2.0, 0.5]
    positions = [
        [0.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
        [0.0, 13.0, 0.5],
        [-18.0, 0.0, -0.3],
        [0.0, -25.0, 0.8],
    ]

    def circular_v(r):
        return np.sqrt(1000.0 / r)

    velocities = [
        [0.0, 0.0, 0.0],
        [0.0, circular_v(8.0), 0.0],
        [-circular_v(13.0), 0.0, 0.0],
        [0.0, -circular_v(18.0), 0.0],
        [circular_v(25.0), 0.0, 0.0],
    ]
    dt = 0.004
    return masses, positions, velocities, dt


PRESETS = {"figure8": preset_figure_eight, "random": preset_random, "solar": preset_solar}


class FreeCamera:
    def __init__(self, position):
        self.pos = np.array(position, dtype=float)
        self.yaw = -90.0
        self.pitch = -15.0
        self.base_speed = 12.0
        self.sensitivity = 0.12

    def front_vector(self):
        yaw_r = math.radians(self.yaw)
        pitch_r = math.radians(self.pitch)
        x = math.cos(yaw_r) * math.cos(pitch_r)
        y = math.sin(pitch_r)
        z = math.sin(yaw_r) * math.cos(pitch_r)
        v = np.array([x, y, z])
        return v / np.linalg.norm(v)

    def right_vector(self):
        f = self.front_vector()
        up = np.array([0.0, 1.0, 0.0])
        r = np.cross(f, up)
        return r / np.linalg.norm(r)

    def process_mouse(self, dx, dy):
        self.yaw += dx * self.sensitivity
        self.pitch -= dy * self.sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def process_keys(self, keys, dt):
        speed = self.base_speed * (3.0 if keys[K_LSHIFT] else 1.0)
        step = speed * dt
        front = self.front_vector()
        right = self.right_vector()
        if keys[K_w]:
            self.pos += front * step
        if keys[K_s]:
            self.pos -= front * step
        if keys[K_a]:
            self.pos -= right * step
        if keys[K_d]:
            self.pos += right * step
        if keys[K_SPACE]:
            self.pos[1] += step
        if keys[K_LCTRL]:
            self.pos[1] -= step

    def apply(self):
        f = self.front_vector()
        center = self.pos + f
        gluLookAt(self.pos[0], self.pos[1], self.pos[2],
                  center[0], center[1], center[2],
                  0.0, 1.0, 0.0)

def draw_grid(size=60, step=4):
    glColor3f(0.18, 0.18, 0.25)
    glBegin(GL_LINES)
    i = -size
    while i <= size:
        glVertex3f(i, 0.0, -size)
        glVertex3f(i, 0.0, size)
        glVertex3f(-size, 0.0, i)
        glVertex3f(size, 0.0, i)
        i += step
    glEnd()


def draw_hud(lines, width, height, font):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, width, 0, height)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)

    y = height - 20
    for line in lines:
        surf = font.render(line, True, (255, 255, 255))
        w, h = surf.get_size()
        data = pygame.image.tostring(surf, "RGBA", True)
        glRasterPos2i(10, y - h)
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)
        y -= h + 4

    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

def main():
    parser = argparse.ArgumentParser(description="Interactive 3D N-body viewer")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="random")
    parser.add_argument("--n", type=int, default=20, help="number of bodies (random preset)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (random preset)")
    parser.add_argument("--substeps", type=int, default=6, help="physics steps per rendered frame")
    parser.add_argument("--trail", type=int, default=300, help="trail length in points per body")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--test-frames", type=int, default=0,
                         help="internal: render N frames then exit (for automated testing)")
    args = parser.parse_args()

    if args.preset == "random":
        masses, positions, velocities, dt = preset_random(n=args.n, seed=args.seed)
    else:
        masses, positions, velocities, dt = PRESETS[args.preset]()

    sim = NBodySimulation(masses, positions, velocities, G=1.0, softening=0.2)
    trails = [deque(maxlen=args.trail) for _ in range(sim.n)]
    colors = [colorsys.hsv_to_rgb(i / max(1, sim.n), 0.75, 1.0) for i in range(sim.n)]

    pygame.init()
    pygame.font.init()
    display = (args.width, args.height)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)
    pygame.display.set_caption("Interactive N-Body Simulation")
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    font = pygame.font.SysFont("consolas", 16)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glClearColor(0.02, 0.02, 0.05, 1.0)

    def set_projection(w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, w / float(h), 0.1, 5000.0)
        glMatrixMode(GL_MODELVIEW)

    set_projection(*display)

    span = float(np.max(np.abs(sim.positions))) + 5.0
    camera = FreeCamera(position=(0.0, span * 0.4, span * 1.8))
    quadric = gluNewQuadric()

    clock = pygame.time.Clock()
    paused = False
    running = True
    frame_count = 0

    controls_text = [
        "WASD move | Space/Ctrl up-down | Shift boost | Mouse look",
        "Scroll: speed | P: pause | R: reset camera | Esc: quit",
    ]

    while running:
        dt_frame = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_p:
                    paused = not paused
                elif event.key == K_r:
                    camera = FreeCamera(position=(0.0, span * 0.4, span * 1.8))
            elif event.type == MOUSEMOTION:
                dx, dy = event.rel
                camera.process_mouse(dx, dy)
            elif event.type == MOUSEWHEEL:
                camera.base_speed = max(1.0, camera.base_speed + event.y * 2.0)
            elif event.type == VIDEORESIZE:
                display = (event.w, event.h)
                set_projection(*display)

        keys = pygame.key.get_pressed()
        camera.process_keys(keys, dt_frame)

        if not paused:
            for _ in range(args.substeps):
                sim.step(dt)
            for i in range(sim.n):
                trails[i].append(sim.positions[i].copy())

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        camera.apply()

        draw_grid(size=max(20, int(span)), step=max(2, int(span // 10)))

        for i in range(sim.n):
            trail_list = trails[i]
            n_trail = len(trail_list)
            if n_trail > 1:
                glBegin(GL_LINE_STRIP)
                for idx, p in enumerate(trail_list):
                    alpha = idx / (n_trail - 1)
                    glColor4f(colors[i][0], colors[i][1], colors[i][2], alpha * 0.8)
                    glVertex3f(p[0], p[1], p[2])
                glEnd()

            glColor3f(*colors[i])
            glPushMatrix()
            glTranslatef(sim.positions[i][0], sim.positions[i][1], sim.positions[i][2])
            radius = 0.35 * (sim.masses[i] ** (1.0 / 3.0))
            gluSphere(quadric, radius, 16, 16)
            glPopMatrix()

        status = "PAUSED" if paused else f"running (x{args.substeps} substeps/frame)"
        hud_lines = [
            f"N-Body Viewer  |  bodies: {sim.n}  |  {status}  |  fps: {clock.get_fps():.0f}",
        ] + controls_text
        draw_hud(hud_lines, display[0], display[1], font)

        pygame.display.flip()

        frame_count += 1
        if args.test_frames and frame_count >= args.test_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()