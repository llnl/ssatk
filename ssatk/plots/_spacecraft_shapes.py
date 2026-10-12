"""Small procedural viewing meshes; no CAD, textures, network, or dynamics.

Coordinates are metres in visual body axes. Parts are deliberately simplified;
see spacecraft_presets for mission-specific references and approximation notes.
"""
from __future__ import annotations

import numpy as np

SILVER = (0.73, 0.78, 0.84)
WHITE = (0.86, 0.88, 0.90)
GOLD = (0.82, 0.60, 0.19)
BLUE = (0.11, 0.27, 0.48)
DARK = (0.035, 0.050, 0.075)


class Builder:
    """Convex faces with consistent outward winding and named components."""

    def __init__(self):
        self.vertices, self.faces, self.colors, self.parts = [], [], [], []

    def face(self, vertices, color=SILVER, part="structure"):
        self.faces.append(np.arange(len(self.vertices), len(self.vertices) + len(vertices)))
        self.vertices.extend(vertices)
        self.colors.append(color)
        self.parts.append(part)

    def box(self, size, center=(0, 0, 0), color=SILVER, part="bus"):
        from ssatk.satellites import box_facets
        for facet in box_facets(size):
            self.face(np.asarray(facet.vertices_body) + center, color, part)

    def cylinder(self, radius, length, center=(0, 0, 0), color=SILVER,
                 part="barrel", axis=(0, 0, 1), segments=24, top_radius=None, caps=True):
        axis = np.asarray(axis, float)
        axis /= np.linalg.norm(axis)
        seed = np.eye(3)[np.argmin(np.abs(axis))]
        u = np.cross(seed, axis)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        angles = np.arange(segments) * 2 * np.pi / segments
        circle = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
        lower = np.asarray(center) - axis * length / 2 + radius * circle
        upper = np.asarray(center) + axis * length / 2 + (radius if top_radius is None else top_radius) * circle
        for j in range(segments):
            k = (j + 1) % segments
            self.face([lower[j], lower[k], upper[k], upper[j]], color, part)
        if caps:
            self.face(lower[::-1], color, part)
            self.face(upper, color, part)

    def disk(self, radius, center, normal=(0, 0, 1), color=DARK, part="aperture", segments=24):
        axis = np.asarray(normal, float)
        axis /= np.linalg.norm(axis)
        seed = np.eye(3)[np.argmin(np.abs(axis))]
        u = np.cross(seed, axis)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        angle = np.arange(segments) * 2 * np.pi / segments
        self.face(np.asarray(center) + radius * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v), color, part)

    def rod(self, start, end, radius=0.03, color=SILVER, part="support"):
        start, end = np.asarray(start), np.asarray(end)
        self.cylinder(radius, np.linalg.norm(end - start), (start + end) / 2,
                      color, part, axis=end - start, segments=8)

    def panel(self, center, u, v, rows=4, cols=6, color=BLUE, part="solar_array"):
        """A rectangular backing with inset cells; u/v are full side vectors."""
        center, u, v = np.asarray(center, float), np.asarray(u, float), np.asarray(v, float)
        normal = np.cross(u, v)
        normal /= np.linalg.norm(normal)
        self.face([center-u/2-v/2, center+u/2-v/2, center+u/2+v/2, center-u/2+v/2], SILVER, part + "_frame")
        for row in range(rows):
            for col in range(cols):
                p = center + ((col + 0.5) / cols - 0.5) * u + ((row + 0.5) / rows - 0.5) * v
                # Tiny visual offset prevents coplanar z fighting, not a physical cell thickness.
                p += normal * min(np.linalg.norm(u), np.linalg.norm(v)) * 1e-4
                a, b = 0.47 * u / cols, 0.47 * v / rows
                tint = np.asarray(color) * (0.92 if (row + col) % 2 else 1.0)
                self.face([p-a-b, p+a-b, p+a+b, p-a+b], tint, part)

    def wings(self, span, width, inner, z=0, normal="z", part="solar_array"):
        length = span / 2 - inner
        for sign in (-1, 1):
            center = [sign * (inner + length / 2), 0, z]
            if normal == "z":
                u, v = [length, 0, 0], [0, width, 0]
            else:
                u, v = [length, 0, 0], [0, 0, width]
            self.panel(center, u, v, part=part)
            self.rod([sign*inner, 0, z], [sign*(inner+length), 0, z],
                     radius=min(0.025, width / 40), part="array_boom")

    def aperture(self, radius, z, x=0, y=0, part="aperture"):
        self.disk(radius, [x, y, z], color=DARK, part=part)
        # A smaller inset tinted pupil makes openings readable without texture assets.
        self.disk(radius * 0.82, [x, y, z + 0.001], color=(0.12, 0.17, 0.22), part=part + "_pupil")


def cubesat(b, size=(0.1, 0.1, 0.3405), instrument="none", wings=False, count=1):
    """Rail-framed chassis; generic 3U envelope follows NASA's CDS table."""
    x, y, z = size
    rail = min(x, y) * 0.07
    b.box((x-2*rail, y-2*rail, z-0.012), color=DARK)
    for sx in (-1, 1):
        for sy in (-1, 1):
            b.box((rail, rail, z), (sx*(x-rail)/2, sy*(y-rail)/2, 0), SILVER, "rail")
    for sign in (-1, 1):
        b.box((x, y, .002), (0, 0, sign*(z/2-.001)), SILVER, "end_deck")
        # Cell faces sit inside the rail envelope, above the dark chassis.
        b.panel((sign*(x/2-rail+.0001), 0, 0), (0, sign*(y-2.4*rail), 0), (0, 0, z-.03),
                rows=max(2, round(z/.05)), cols=2, part="body_cells_x")
        b.panel((0, sign*(y/2-rail+.0001), 0), (-sign*(x-2.4*rail), 0, 0), (0, 0, z-.03),
                rows=max(2, round(z/.05)), cols=2, part="body_cells_y")
    if wings:
        b.wings(3.5*x, z*.85, x/2+.01)
    if instrument != "none":
        for i in range(count):
            cx = (i - (count-1)/2) * min(x/(count+1), .09)
            r = min(x/(count+1)/2, y*.3)
            b.disk(r, (cx, 0, z/2+.001), color=GOLD if instrument == "detector" else DARK,
                   part="detector" if instrument == "detector" else "aperture")


def hubble(b):
    b.cylinder(2.1, 4.0, (0, 0, -4.6), SILVER, "aft_shroud")
    b.cylinder(1.6, 9.2, (0, 0, 2.0), WHITE, "optical_tube")
    b.aperture(1.2, 6.601)
    b.wings(19.2, 2.6, 2.5, z=-2, normal="y")  # Each wing is 7.1 x 2.6 m.
    b.rod([0, -2, -4], [0, -3.5, -3], .10)
    b.disk(.65, [0, -3.5, -3], [0, -1, 0], SILVER, "antenna")


def webb(b):
    outline = np.array([[-10.6, 0, 0], [-5, -7.1, 0], [5, -5.5, 0],
                        [10.6, 0, 0], [5, 5.5, 0], [-5, 7.1, 0]])
    for layer in range(5):
        points = outline * [1-.018*layer, 1-.018*layer, 1] + [0, 0, .18*layer]
        b.face(points, np.array(SILVER) * (0.80 + .04*layer), f"sunshield_{layer+1}")
    b.box((3.2, 2.4, 1.2), (0, 0, -.8), GOLD)
    b.rod([0, 0, .7], [0, 0, 3], .35, part="tower")
    # 18 hexagons on two axial-coordinate rings; 1.32 m across flats.
    radius = 1.32 / np.sqrt(3)
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) > 2 or (q == 0 and r == 0):
                continue
            cx, cz = 1.32*(q+r/2), 1.5*radius*r + 4.2
            theta = np.pi/6 + np.arange(6)*np.pi/3
            b.face(np.column_stack((cx+radius*np.cos(theta), np.zeros(6), cz+radius*np.sin(theta))),
                   GOLD, "primary_segment")  # Optical face normal is -Y.
    secondary = [0, -3.7, 4.2]
    for start in ([-2.6, 0, 2.7], [2.6, 0, 2.7], [0, 0, 7.0]):
        b.rod(start, secondary, .035, DARK)
    b.disk(.37, secondary, (0, -1, 0), SILVER, "secondary")
    b.panel((0, 5.4, -1.1), (2.0, 0, 0), (0, 3.0, 0), rows=4)


def roman(b):
    b.box((2.5, 2.5, 1.6), (0, 0, -2.6), GOLD)
    b.cylinder(1.5, 5.2, (0, 0, .8), WHITE, "outer_barrel")
    b.aperture(1.2, 3.401)
    for row in range(2):
        for col in range(3):
            b.panel(((col-1)*2.4, -1.8, row*3.0-1.5), (2.35, 0, 0), (0, 0, 2.95),
                    rows=6, cols=4, part=f"sunshield_panel_{row*3+col+1}")
    b.face([[-1.5, -1.7, 3.4], [1.5, -1.7, 3.4], [1.5, -2.8, 6], [-1.5, -2.8, 6]], DARK, "aperture_cover")


def chandra(b):
    b.box((2.3, 2.0, 2.0), (0, 0, -5.9), GOLD)
    b.cylinder(.65, 10.3, (0, 0, .25), SILVER, "optical_bench")
    b.cylinder(1.2, 1.5, (0, 0, 6.15), GOLD, "mirror_assembly")
    b.aperture(.60, 6.901)
    b.wings(19.5, 2.4, 1.6, z=-5.3, normal="y")


def barrel(b, radius=.7, length=3, aperture=.5, color=WHITE, panel="wings", count=1):
    b.cylinder(radius, length, color=color, part="telescope_shroud")
    for i in range(count):
        x = (i-(count-1)/2) * min(radius, 2*aperture/count)
        b.aperture(aperture/count, length/2+.001, x=x)
    b.box((radius*1.5, radius*1.5, .35*length), (0, 0, -length*.325), GOLD)
    if panel == "wings":
        b.wings(radius*8, length*.48, radius*1.2, z=-length*.25, normal="y")
    elif panel == "side":
        b.panel((0, radius*1.15, 0), (radius*2.5, 0, 0), (0, 0, length), part="solar_shield")


def tess(b):
    b.box((.75, .75, .7), color=GOLD)
    for i, angle in enumerate(np.deg2rad([-36, -12, 12, 36])):
        axis = np.array([0, np.sin(angle), np.cos(angle)])
        center = np.array([0, (i-1.5)*.23, .58])
        b.cylinder(.095, .30, center, WHITE, "camera_hood", axis=axis, segments=16)
        b.disk(.0525, center+axis*.151, axis, DARK, "camera_pupil")
    b.wings(3.2, .85, .5)
    b.disk(.35, [0, -.40, -.15], [0, -1, 0], SILVER, "antenna")


def mast(b, length=10.14, count=2, optics_radius=.22):
    b.box((1.1, 1.0, .55), (0, 0, -length/2-.275), GOLD)
    b.wings(4.4, 1.3, .65, z=-length/2-.15)
    for x, y in ((-.20, -.20), (.20, -.20), (.20, .20), (-.20, .20)):
        b.rod([x, y, -length/2], [x, y, length/2], .018, GOLD, "mast")
    for i in range(12):
        lo, hi = -length/2+length*i/12, -length/2+length*(i+1)/12
        for y in (-.20, .20):
            b.rod([-.20, y, lo], [.20, y, hi], .010, SILVER, "mast_brace")
    for i in range(count):
        angle = (i * 2*np.pi/count) if count > 2 else np.pi*i
        x, y = .34*np.cos(angle), .34*np.sin(angle)
        b.cylinder(optics_radius, .4, (x, y, length/2-.2), WHITE, "optics")
        b.aperture(optics_radius*.8, length/2+.001, x, y)
        # Visible reference discs separated by the published focal length.
        b.disk(optics_radius*.7, (x, y, length/2), part="optics_reference")
        b.disk(.06, (x, y, -length/2), color=DARK, part="detector_reference")


def lisa(b):
    b.cylinder(1.25, .55, color=GOLD, part="bus", segments=12)
    b.disk(1.25, (0, 0, .277), color=BLUE, part="solar_array")
    for angle in np.deg2rad((-30, 30)):
        axis = np.array([np.cos(angle), np.sin(angle), 0])
        b.cylinder(.15, .7, axis*1.0, WHITE, "laser_telescope", axis=axis)
        b.disk(.12, axis*1.351, axis, DARK, "aperture")


def spherex(b):
    b.box((.9, .9, .65), (0, 0, -1.0), GOLD)
    for i in range(3):
        b.cylinder(.65-.10*i, 1.7, (0, 0, .45), WHITE if i == 0 else SILVER,
                   f"photon_shield_{i+1}", top_radius=1.6-.15*i, caps=False)
    b.cylinder(.22, .65, (0, 0, .1), DARK)
    b.aperture(.10, .426)
    b.panel((0, 0, -1.34), (2.0, 0, 0), (0, 1.4, 0))


def detector_bus(b, size=(2, 2, 1), layout="fermi", count=4):
    x, y, z = size
    b.box(size, color=GOLD)
    b.wings(x*4.5, y*.8, x*.65, z=-z*.2)
    if layout == "fermi":
        for i in range(4):
            for j in range(4):
                b.box((x*.22, y*.22, z*.35), ((i-1.5)*x*.24, (j-1.5)*y*.24, z*.675), DARK, "LAT_module")
    elif layout == "compton":
        for px, radius in ((-x*.23, .75), (x*.23, .6)):
            b.cylinder(radius, 2, (px, 0, z/2+1), WHITE, "instrument")
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    center = np.array([sx*x/2, sy*y/2, sz*z/2])
                    normal = np.array([sx, sy, sz], float)/np.sqrt(3)
                    b.cylinder(.22, .22, center+normal*.11, WHITE, "BATSE", axis=normal)
    elif layout == "swift":
        b.box((x*.9, y*.8, .25), (0, 0, z/2+.12), DARK, "coded_mask")
        for px, radius in ((-x*.30, .15), (x*.30, .30)):
            b.cylinder(radius, z*1.5, (px, y*.65, z*.25), SILVER, "telescope")
            b.aperture(radius*.8, z*1.001, px, y*.65)
    else:
        for i in range(count):
            cx = (i-(count-1)/2)*x/(count+1)
            b.cylinder(min(x/(count+1)*.35, .25), z*.5, (cx, 0, z*.75), WHITE, "detector")


def dish(b, diameter=1.4, length=2.0, dual=False, shield=False):
    b.cylinder(diameter*.45, length*.35, (0, 0, -length*.30), GOLD, "bus", segments=12)
    n = 2 if dual else 1
    for i in range(n):
        x = (i-(n-1)/2)*diameter*.6
        b.cylinder(diameter*.5/n, length*.55, (x, 0, length*.15), SILVER,
                   "reflector_shroud", top_radius=diameter*.52/n, caps=False)
        b.disk(diameter*.45/n, (x, 0, length*.425+.001), color=(.38,.45,.55), part="reflector")
    if shield:
        b.cylinder(diameter*1.15, .08, (0, 0, -length*.5), SILVER, "sunshield", segments=24)
        b.disk(diameter*1.14, (0, 0, -length*.545), (0,0,-1), BLUE, "solar_array")
    else:
        b.wings(diameter*4, length*.5, diameter*.6, z=-length*.30)


def hosted(b, count=56, layout="nicer"):
    b.box((1.1, .85, .50), color=WHITE, part="payload")
    if layout == "nicer":
        for i in range(7):
            for j in range(8):
                b.cylinder(.047, .09, ((i-3)*.14, (j-3.5)*.10, .295), SILVER, "concentrator", segments=12)
                b.disk(.04, ((i-3)*.14, (j-3.5)*.10, .341), color=DARK, part="pupil")
    elif layout == "astro":
        for x, radius, length in ((-.65,.3,2.1),(0,.25,1.7),(.6,.20,1.4)):
            b.cylinder(radius,length,(x,0,length/2+.25),WHITE,"instrument")
            b.aperture(radius*.9,length+.251,x)
    else:
        for i in range(count):
            b.cylinder(.12,.20,((i-(count-1)/2)*.28,0,.35),SILVER,"detector")
    for x in (-.35,.35):
        b.rod([x,0,-.25],[x,0,-.6],.035,part="mount")


def early(b, style="oao", length=3, radius=1):
    if style == "rae":
        b.cylinder(.45,1.1,color=GOLD,segments=12)
        for sx in (-1,1):
            for sz in (-1,1):
                b.rod([0,0,0],[sx*115,0,sz*198],.12,WHITE,"radio_antenna")
        b.wings(4,1,.6)
    elif style == "explorer":
        b.cylinder(radius,length,color=WHITE,segments=8)
        b.cylinder(radius,.4,(0,0,length/2+.2),GOLD,top_radius=radius*.2,segments=8)
        b.rod([0,0,-length/2],[0,0,-length],.025)
    elif style == "oso":
        b.cylinder(radius,length*.25,(0,0,-length*.3),GOLD,segments=16,part="wheel")
        b.box((radius*.7,radius*.6,length*.5),(0,0,length*.1),WHITE,"sail_instruments")
        b.panel((0,0,length*.35),(radius*2.4,0,0),(0,0,length*.65),part="sail")
    else:
        b.cylinder(radius,length,color=SILVER,segments=8)
        b.aperture(radius*.4,length/2+.001)
        b.wings(radius*6,length*.7,radius*1.1)
