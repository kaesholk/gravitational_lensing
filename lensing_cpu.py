import sys
import math
import numpy as np
import pygame
from pygame.locals import *
from numba import njit, prange

G = 6.67430e-11            # gravitational constant
c = 299792458.0            # speed of light (m/s)
M_sun = 1.98847e30         # kg
parsec = 3.085677581491367e16  # meters

# config
FPS = 30
max_dim = 600

# https://en.wikipedia.org/wiki/Schwarzschild_radius
def schwarzschild_radius_m(mass_solar):
    M = mass_solar * M_sun
    return 2.0 * G * M / (c**2)

# https://en.wikipedia.org/wiki/Einstein_radius
def einstein_radius_m(mass_solar, D_l, D_s):
    if D_s <= D_l:
        return 0.0
    M = mass_solar * M_sun
    D_ls = D_s - D_l
    return math.sqrt( (4.0 * G * M * D_l * D_ls) / (c**2 * D_s) )


@njit(parallel=True, fastmath=True)
def lens_image_numba(src, lens_x, lens_y, thetaE_px,
                     rs_px, redshift_strength,
                     shadow):

    h, w, _ = src.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for y in prange(h):
        for x in range(w):

            dx = x - lens_x
            dy = y - lens_y
            r = math.sqrt(dx*dx + dy*dy) + 1e-8

            alpha_x = thetaE_px * (dx / r)
            alpha_y = thetaE_px * (dy / r)

            beta_x = dx - alpha_x
            beta_y = dy - alpha_y

            src_x = lens_x + beta_x
            src_y = lens_y + beta_y

            # bilinear sampling, https://medium.com/@chathuragunasekera/image-resampling-algorithms-for-pixel-manipulation-bee65dda1488
            x0 = int(math.floor(src_x))
            y0 = int(math.floor(src_y))
            x1 = x0 + 1
            y1 = y0 + 1

            # clamp
            if x0 < 0: x0 = 0
            if y0 < 0: y0 = 0
            if x1 >= w: x1 = w-1
            if y1 >= h: y1 = h-1

            wx = src_x - x0
            wy = src_y - y0

            for c in range(3):
                Ia = src[y0, x0, c]
                Ib = src[y0, x1, c]
                Ic = src[y1, x0, c]
                Id = src[y1, x1, c]

                val = (
                    Ia * (1-wx)*(1-wy) +
                    Ib * wx*(1-wy) +
                    Ic * (1-wx)*wy +
                    Id * wx*wy
                )

                # redshift and dimming, just an approximation
                ratio = rs_px / r
                if ratio > 0.9999:
                    ratio = 0.9999
                if ratio < 0:
                    ratio = 0

                g = math.sqrt(1.0 - ratio)

                val = val * (g * redshift_strength + (1.0 - redshift_strength))

                if c == 2:
                    val *= g
                if c == 1:
                    val *= (0.5 + 0.5*g)

                if val < 0:
                    val = 0
                if val > 255:
                    val = 255

                out[y, x, c] = int(val)
                
            # shadow and photon sphere radii given here:
            # https://en.wikipedia.org/wiki/Black_hole
            # photon sphere = 1.5 * schwarzschild radius
            # shadow radius = 2.6 * schwarzschild radius

            # ensure the black hole shadow is fully black (photon sphere not included yet)
            if shadow:
                bcrit = 2.6 * rs_px
                if r < bcrit:
                    out[y, x, 0] = 0
                    out[y, x, 1] = 0
                    out[y, x, 2] = 0

    return out

def main():
    pygame.init()

    src = pygame.image.load("bullet.jpg")
    src = pygame.surfarray.array3d(src).swapaxes(0,1)
    W, H = src.shape[1], src.shape[0]

    if W > max_dim or H > max_dim:
        scale = max_dim / max(W, H)
        W = int(W * scale)
        H = int(H * scale)
        src = pygame.transform.smoothscale(pygame.surfarray.make_surface(src.swapaxes(0,1)), (W, H))
        src = pygame.surfarray.array3d(src).swapaxes(0,1)

    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Gravitational lensing demo")

    lens_pos = np.array([W//2, H//2], dtype=float)
    dragging = False

    mass_solar = 10         # black hole mass
    D_l = 1.0e20               # lens distance
    D_s = 1.0e21               # source distance
    meters_per_pixel = 5.0e10
    redshift_strength = 0.9

    font = pygame.font.SysFont(None, 20)

    while True:
        for ev in pygame.event.get():
            if ev.type == QUIT:
                pygame.quit(); sys.exit()

            elif ev.type == KEYDOWN:
                if ev.key == K_ESCAPE:
                    pygame.quit(); sys.exit()
                # mass controls
                elif ev.key == K_UP:
                    mass_solar *= 2.0
                elif ev.key == K_DOWN:
                    mass_solar = max(1e-6, mass_solar * 0.5)
                # meters-per-pixel
                elif ev.key == K_LEFT:
                    meters_per_pixel = max(1.0, meters_per_pixel * 0.5)
                elif ev.key == K_RIGHT:
                    meters_per_pixel *= 2.0
                # redshift
                elif ev.key == K_COMMA:
                    redshift_strength = max(0.0, redshift_strength - 0.05)
                elif ev.key == K_PERIOD:
                    redshift_strength = min(1.0, redshift_strength + 0.05)

            elif ev.type == MOUSEBUTTONDOWN:
                if ev.button == 1:
                    dragging = True
            elif ev.type == MOUSEBUTTONUP:
                if ev.button == 1:
                    dragging = False
            elif ev.type == MOUSEMOTION and dragging:
                mx, my = ev.pos
                lens_pos[0], lens_pos[1] = mx, my

        # compute physical radii
        rs_m = schwarzschild_radius_m(mass_solar)
        rs_px = rs_m / meters_per_pixel

        RE_m = einstein_radius_m(mass_solar, D_l, D_s)
        thetaE_px = RE_m / meters_per_pixel

        out = lens_image_numba(
            src,
            lens_pos[0], lens_pos[1],
            thetaE_px,
            rs_px,
            redshift_strength,
            True          # shadow
        )

        surf = pygame.surfarray.make_surface(out.swapaxes(0,1))
        screen.blit(surf, (0,0))

        #pygame.draw.circle(screen, (255,50,50), (int(lens_pos[0]), int(lens_pos[1])), max(4,int(min(10, rs_px))), 2)
        info_lines = [
            f"Mass = {mass_solar:.3g} M_sun   (UP/DOWN to x2 / /2)",
            f"meters_per_pixel = {meters_per_pixel:.2e} m/px   (LEFT/RIGHT to x0.5/x2)",
            f"r_s = {rs_px:.2e} px   ({rs_m:.2e} m)",
            f"Einstein R = {thetaE_px:.2f} px",
            f"redshift_strength = {redshift_strength:.2f}   (</> to change)",
            "Drag to move lens"
        ]
        for i, line in enumerate(info_lines):
            txt = font.render(line, True, (220,220,220))
            screen.blit(txt, (8, 8 + i*18))

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()