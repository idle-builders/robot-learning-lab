"""APE0 obstacle world: a grid of terrain tiles, each parallel world spawning on one, sized for a 50 cm ape.

As in Microduck's training world (pollen-robotics/microduck_rl, microduck_velocity_env_cfg.py, commit cb70b792), every
tile draws its type and difficulty at random (mjlab TerrainGeneratorCfg, curriculum off), so the parallel worlds see
different ground at once. Sizes follow the robot: legs about 0.20 m, feet 0.10 x 0.065 m (x1.3), a reference swing
lift of 12-40 mm. They are a starting proposal for the owner, not an approved target:

| Element | Range | Why |
|---|---|---|
| Rough ground | 0-12 mm, blocks of 10 cm over the central 2 m | floor unevenness a sole of 13 cm meets indoors and outdoors |
| Rolling ground | 0-15 mm waves, 1 m long | gentle lawn or a sagging floor |
| Slopes | 0-10 deg (rise/run 0-0.18), hill and hollow | ramps and kerbs; Microduck trains 2-6 deg at half our size |
| Stairs | 5-30 mm risers, 15 cm treads | a step is 0.15 x leg length here; human stairs are about 0.19 |
| Box grid | 0-15 mm, 15 cm cells | cobbles and mats; cells wider than the sole |

Every tile is built from boxes and planes. Heightfield tiles were tried first and dropped (2026-11-09): mujoco_warp gave
the mesh feet contacts with downward normals and one or two support points on them, so a 3 mm heightfield toppled the
controller more often than 3 cm box steps. Slopes and waves are inclined boxes along one axis (SlopeTerrainCfg).
| Mats and sills | 5-20 mm flat slabs, 0.3-1 m | rugs, door sills and thresholds |
| Room | couch, table, chairs, a step stool and a wall with a doorway | human furniture is taller than the robot: things to walk around, under and step onto |
| Clutter | shoes, books, toy blocks, cables, a soft cushion | a lived-in floor: small things underfoot |

Every world also has (add_world_objects):
- a floor whose grip is drawn per world, from smooth wood or tile (sole friction 0.5) to carpet (1.0);
- a 14 cm, 120 g ball 0.3-1 m from the spawn point, sometimes rolling toward the robot;
- a light (2 kg) interior door in a short wall, ajar, 0.8-2.5 m away at any heading: in some worlds' paths;
- a carried toy of 0-100 g in each fist.
The owner set the sizes (home floor, 2026-11-08) and the aim: more realism and variety in training, so the controller
is robust; furniture and doors are explored by trial, with no fixed list. The robot spawns on each tile's flat
spawn square.

    from parcour import PARCOUR_TERRAIN
    cfg.scene.terrain = TerrainEntityCfg(terrain_type='generator', terrain_generator=PARCOUR_TERRAIN, max_init_terrain_level=...)
"""
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains import (BoxFlatTerrainCfg, BoxPyramidStairsTerrainCfg, BoxRandomGridTerrainCfg,
    BoxRandomSpreadTerrainCfg)
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_plane

TILE = 3.0  # m; box counts grow with (tile / cell)^2 (Microduck's 0.12 m cells on 8 m tiles ran out of memory)
SPAWN = 0.8  # m, free square at each tile's centre

WOOD, FABRIC, WALL = (.55, .40, .28, 1.), (.35, .45, .60, 1.), (.85, .83, .78, 1.)


def _box(body, size, pos, rgba, yaw=0.):
    g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=np.maximum(1e-6, np.asarray(size) / 2), pos=pos)
    g.quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    return TerrainGeometry(geom=g, color=rgba)


def _rotate(p, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])


def couch(body, at, yaw):
    """0.85 x 1.8 m, seat 0.42 m, back 0.80 m."""
    parts = [((.85, 1.8, .42), (0, 0, .21)), ((.18, 1.8, .38), (-.335, 0, .61)), ((.85, .16, .20), (0, .82, .52)),
             ((.85, .16, .20), (0, -.82, .52))]
    return [_box(body, s, at + _rotate(np.array(p), yaw), FABRIC, yaw) for s, p in parts]


def table(body, at, yaw, height=.72, top=(.8, 1.2)):
    """Four 4 cm legs and a 3 cm top: the robot (about 0.5 m) walks under it."""
    out = [_box(body, (*top, .03), at + np.array([0, 0, height - .015]), WOOD, yaw)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = np.array([sx * (top[0] / 2 - .05), sy * (top[1] / 2 - .05), (height - .03) / 2])
            out.append(_box(body, (.04, .04, height - .03), at + _rotate(p, yaw), WOOD, yaw))
    return out


def chair(body, at, yaw, seat=.45):
    out = [_box(body, (.42, .42, .03), at + np.array([0, 0, seat - .015]), WOOD, yaw),
           _box(body, (.03, .42, .45), at + _rotate(np.array([-.195, 0, seat + .225]), yaw), WOOD, yaw)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append(_box(body, (.03, .03, seat - .03), at + _rotate(np.array([sx * .18, sy * .18, (seat - .03) / 2]), yaw), WOOD, yaw))
    return out


def stool(body, at, yaw, height):
    """A step stool or low box the ape can step or climb onto."""
    return [_box(body, (.35, .45, height), at + np.array([0, 0, height / 2]), WOOD, yaw)]


def doorway(body, x, width, length, thickness=.10, height=1.0, gap=.80):
    """A wall across the tile at x with a door-sized gap (an open doorway, no door leaf)."""
    rest = (length - gap) / 2
    return [_box(body, (thickness, rest, height), (x, rest / 2, height / 2), WALL),
            _box(body, (thickness, rest, height), (x, length - rest / 2, height / 2), WALL)]


@dataclass(kw_only=True)
class FurnitureRoomTerrainCfg(SubTerrainCfg):
    """A floor with human furniture placed at random around a free spawn square; difficulty adds clutter and a
    higher step stool (2-8 cm)."""
    spawn: float = SPAWN
    stool_height_range: tuple[float, float] = (.02, .08)

    def function(self, difficulty, spec, rng):
        body = spec.body('terrain')
        L, W = self.size
        geoms = [TerrainGeometry(geom=make_plane(body, self.size, 0., center_zero=False)[0], color=(.62, .60, .56, 1.))]
        centre = np.array([L / 2, W / 2, 0.])
        taken = [(centre[:2], self.spawn / 2 + .15)]

        def place(radius):
            for _ in range(50):
                p = np.array([rng.uniform(radius, L - radius), rng.uniform(radius, W - radius), 0.])
                if all(np.linalg.norm(p[:2] - c) > radius + r for c, r in taken):
                    taken.append((p[:2], radius))
                    return p
            return None

        makers = [(1.0, lambda p, y: couch(body, p, y)), (.75, lambda p, y: table(body, p, y)),
                  (.32, lambda p, y: chair(body, p, y)), (.32, lambda p, y: chair(body, p, y)),
                  (.30, lambda p, y: stool(body, p, y, rng.uniform(*self.stool_height_range) * (.25 + .75 * difficulty)))]
        count = 2 + int(round(difficulty * (len(makers) - 2)))
        for radius, make in [makers[i] for i in rng.permutation(len(makers))[:count]]:
            p = place(radius)
            if p is not None:
                geoms += make(p, rng.uniform(0, 2 * np.pi))
        if rng.uniform() < .5:  # a wall with an open doorway on one side of the spawn square
            x = L / 2 + rng.choice([-1, 1]) * (self.spawn / 2 + .35)
            geoms += doorway(body, x, W, W)
        return TerrainOutput(origin=centre, geometries=geoms)


@dataclass(kw_only=True)
class SlopeTerrainCfg(SubTerrainCfg):
    """Ground that varies along x, built from inclined boxes, with a flat spawn square in the middle.
    hill: the spawn square is a plateau and the ground slopes down to the tile edges at the drawn rise/run;
    hollow: the spawn square is sunk and the ground slopes up to the edges;
    rolling: a triangular wave of the drawn height (peak to peak) on both sides of the spawn square."""
    shape: str = 'hill'
    slope_range: tuple[float, float] = (0., .18)
    amplitude_range: tuple[float, float] = (0., .015)
    wavelength: float = 1.0
    spawn: float = SPAWN
    thickness: float = .3

    def profile(self, difficulty):
        L, c, half = self.size[0], self.size[0] / 2, self.spawn / 2
        if self.shape == 'rolling':
            a = self.amplitude_range[0] + difficulty * (self.amplitude_range[1] - self.amplitude_range[0])
            right = [c + half + k * self.wavelength / 2 for k in range(int((c - half) / (self.wavelength / 2)) + 1)]
            right = [x for x in right if x < L - 1e-6] + [L]
            zs = [a * (1 - abs((x - right[0]) / (self.wavelength / 2) % 2 - 1)) for x in right]  # triangle wave, cut at the edge
            xs = [L - x for x in reversed(right)] + right
            return np.array(xs), np.array(list(reversed(zs)) + zs)
        slope = self.slope_range[0] + difficulty * (self.slope_range[1] - self.slope_range[0])
        rise = slope * (c - half) * (1 if self.shape == 'hill' else -1)
        return np.array([0., c - half, c + half, L]), np.array([0., rise, rise, 0.])

    def function(self, difficulty, spec, rng):
        body = spec.body('terrain')
        xs, zs = self.profile(difficulty)
        W, geoms = self.size[1], []
        for x0, x1, z0, z1 in zip(xs[:-1], xs[1:], zs[:-1], zs[1:]):
            dx, dz = x1 - x0, z1 - z0
            length, angle = np.hypot(dx, dz), np.arctan2(dz, dx)
            normal = np.array([-dz, dz * 0 + dx]) / length  # (x, z) of the top face's normal
            centre = np.array([(x0 + x1) / 2, (z0 + z1) / 2]) - normal * self.thickness / 2
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(length / 2, W / 2, self.thickness / 2),
                pos=(centre[0], W / 2, centre[1]))
            g.quat = np.array([np.cos(-angle / 2), 0, np.sin(-angle / 2), 0])  # pitch the box onto the segment
            geoms.append(TerrainGeometry(geom=g, color=(.45, .55, .35, 1.)))
        return TerrainOutput(origin=np.array([self.size[0] / 2, W / 2, zs[len(zs) // 2]]), geometries=geoms)


CLUTTER_COLORS = ((.80, .25, .20, 1.), (.25, .45, .80, 1.), (.90, .75, .20, 1.), (.30, .65, .35, 1.), (.15, .15, .15, 1.))


@dataclass(kw_only=True)
class ClutterTerrainCfg(SubTerrainCfg):
    """A lived-in floor: shoes, books, toy blocks, cables and a soft cushion lying around a free spawn square.
    Everything is static; the cushion is soft (solref time constant 0.05 s against the floor's 0.02 s), the rest hard.
    Difficulty adds items. Heights: cables 8 mm, books 2-4 cm, blocks 3-5 cm, shoes about 7 cm, cushion 8 cm."""
    spawn: float = SPAWN
    items: int = 24

    def function(self, difficulty, spec, rng):
        body = spec.body('terrain')
        L, W = self.size
        geoms = [TerrainGeometry(geom=make_plane(body, self.size, 0., center_zero=False)[0], color=(.55, .45, .35, 1.))]
        free = self.spawn / 2

        def spot(margin):
            for _ in range(30):
                p = np.array([rng.uniform(margin, L - margin), rng.uniform(margin, W - margin)])
                if np.abs(p - (L / 2, W / 2)).max() > free + margin:
                    return p
            return None

        for k in range(int(self.items * (.4 + .6 * difficulty))):
            kind = rng.choice(['shoe', 'book', 'block', 'cable', 'cushion'], p=[.2, .25, .3, .17, .08])
            yaw, rgba = rng.uniform(0, np.pi), CLUTTER_COLORS[rng.integers(len(CLUTTER_COLORS))]
            if kind == 'cable':
                p = spot(.5)
                if p is None:
                    continue
                g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=(.004, rng.uniform(.25, .5), 0), pos=(*p, .004))
                c, s_ = np.cos(yaw / 2), np.sin(yaw / 2)
                g.quat = np.sqrt(.5) * np.array([c, c, s_, s_])  # yaw after a 90 deg roll: the capsule lies flat
                geoms.append(TerrainGeometry(geom=g, color=(.1, .1, .1, 1.)))
                continue
            size = dict(shoe=(.26, .10, .07), book=(.24, .17, rng.uniform(.02, .04)), block=(rng.uniform(.03, .05),) * 3,
                        cushion=(.40, .40, .08))[kind]
            p = spot(max(size[:2]) / 2 + .02)
            if p is None:
                continue
            g = _box(body, size, (*p, size[2] / 2), FABRIC if kind == 'cushion' else rgba, yaw)
            if kind == 'cushion':
                g.geom.solref = np.array([.05, 1.])
            geoms.append(g)
        return TerrainOutput(origin=np.array([L / 2, W / 2, 0.]), geometries=geoms)


# Loose and articulated objects in every world (scene entities, placed around the spawn point at each reset).
# Their geoms are in group 2, so the ground-height rays (group 0) see only the floor. Collision bits: ground 1/1,
# robot 3/1 (add_world_objects), ball 1/1, door 0/2, so the door touches only the robot, never the static ground (a
# mocap door against the box tiles made over 1,300 contacts) nor the ball (the two are placed independently and could
# start inside each other).
BALL_RADIUS, BALL_KG = .07, .12  # a soft toy ball, 14 cm across


def ball_spec():
    spec = mujoco.MjSpec()
    b = spec.worldbody.add_body(name='ball')
    b.add_freejoint(name='ball_joint')
    b.add_geom(name='ball_geom', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(BALL_RADIUS, 0, 0), mass=BALL_KG,
        rgba=(.90, .20, .20, 1.), group=2, friction=(.8, .005, .0001), solref=(.03, 1.), contype=1, conaffinity=1)
    return spec


def door_spec(width=.80, height=1.0, stub=.40):
    """A light interior door (2 kg leaf, 80 cm wide), hinged, in a short wall: the robot can push it open or bump
    into it. The hinge has damping and no spring; the reset sets it ajar."""
    spec = mujoco.MjSpec()
    frame = spec.worldbody.add_body(name='door_frame')
    for side in (-1, 1):
        frame.add_geom(name=f'door_wall_{side + 1}', type=mujoco.mjtGeom.mjGEOM_BOX, size=(.05, stub / 2, height / 2),
            pos=(0, side * (width / 2 + .05 + stub / 2), height / 2), rgba=WALL, group=2, contype=0, conaffinity=2)
        frame.add_geom(name=f'door_post_{side + 1}', type=mujoco.mjtGeom.mjGEOM_BOX, size=(.05, .025, height / 2),
            pos=(0, side * (width / 2 + .025), height / 2), rgba=WOOD, group=2, contype=0, conaffinity=2)
    leaf = frame.add_body(name='door_leaf', pos=(0, -width / 2 + .005, 0))
    leaf.add_joint(name='door_hinge', type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0, 0, 1), range=(-1.7, 1.7), damping=.5,
        frictionloss=.05, armature=.01)
    leaf.add_geom(name='door_leaf_geom', type=mujoco.mjtGeom.mjGEOM_BOX, size=(.015, width / 2 - .01, height / 2 - .01),
        pos=(0, width / 2 - .005, height / 2 + .005), mass=2., rgba=(.70, .55, .38, 1.), group=2, contype=0, conaffinity=2)
    return spec


FLOOR_FRICTION_BASE = .3  # MuJoCo takes the larger of two friction values: a low floor lets the sole's value decide
SOLE_FRICTION = (.5, 1.)  # rubber sole on smooth wood or tile (0.5-0.6) to carpet (about 1.0), drawn per world
CARRIED_KG = (0., .10)  # a toy in each fist, drawn per world


def add_world_objects(cfg):
    """Ball, door and carried toys for an environment config on the obstacle world (tracking.env_cfg terrain='parcour')."""
    from dataclasses import replace
    from mjlab.entity import EntityCfg
    from mjlab.envs import mdp as envs_mdp
    from mjlab.envs.mdp import dr
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    # Start poses keep the objects clear of the robot at the origin (a door leaf on top of it made 1,300 contacts);
    # the resets add their offsets to them: the ball drops from 0.4 m, the door stands at floor level.
    cfg.scene.entities['ball'] = EntityCfg(spec_fn=ball_spec, init_state=EntityCfg.InitialStateCfg(pos=(.6, 0., .4)))
    cfg.scene.entities['door'] = EntityCfg(spec_fn=door_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(1.5, 0., 0.), joint_pos={'door_hinge': .35}))
    robot = cfg.scene.entities['robot']
    robot.collisions = tuple(replace(c, contype=3) for c in robot.collisions)
    # Ball: 0.3-1.0 m from the spawn point, sometimes rolling (up to 0.6 m/s); dropped from 0.4 m onto the ground.
    cfg.events['ball_reset'] = EventTermCfg(func=envs_mdp.reset_root_state_uniform, mode='reset', params=dict(
        pose_range=dict(x=(-.3, .4), y=(-.7, .7)), velocity_range=dict(x=(-.6, .6), y=(-.6, .6)),
        asset_cfg=SceneEntityCfg('ball')))
    # Door: 0.8-2.5 m away at any heading, so it stands in some worlds' paths and not in others.
    cfg.events['door_reset'] = EventTermCfg(func=envs_mdp.reset_root_state_uniform, mode='reset', params=dict(
        pose_range=dict(x=(-.7, 1.), y=(-1.2, 1.2), yaw=(-np.pi, np.pi)), asset_cfg=SceneEntityCfg('door')))
    cfg.events['carried_toy'] = EventTermCfg(func=dr.body_mass, mode='startup', params=dict(ranges=CARRIED_KG,
        operation='add', asset_cfg=SceneEntityCfg('robot', body_names=('R_forearm', 'L_forearm'))))
    cfg.events['foot_friction'].params['ranges'] = SOLE_FRICTION
    return cfg


PARCOUR_SUB_TERRAINS = dict(
    flat=BoxFlatTerrainCfg(proportion=.15),
    # mjlab draws each cell's height in +-bound: a bound of half the size keeps the span at the approved size.
    # Rough blocks cover the central 2 m (where nearly every episode stays); cells within 2 mm merge to keep the box
    # count, and with it the simulator's collision pairs, near the earlier heightfield world's.
    rough=BoxRandomGridTerrainCfg(proportion=.10, grid_width=.10, grid_height_range=(0., .006), platform_width=SPAWN,
        border_width=.5, merge_similar_heights=True, height_merge_threshold=.002),
    rolling=SlopeTerrainCfg(proportion=.05, shape='rolling', amplitude_range=(0., .015)),
    hill=SlopeTerrainCfg(proportion=.10, shape='hill', slope_range=(0., .18)),
    hollow=SlopeTerrainCfg(proportion=.05, shape='hollow', slope_range=(0., .18)),
    stairs=BoxPyramidStairsTerrainCfg(proportion=.10, step_height_range=(.005, .03), step_width=.15,
        platform_width=SPAWN, border_width=.25),
    box_grid=BoxRandomGridTerrainCfg(proportion=.10, grid_width=.15, grid_height_range=(0., .0075), platform_width=SPAWN,
        border_width=.25, merge_similar_heights=True, height_merge_threshold=.002),
    mats_and_sills=BoxRandomSpreadTerrainCfg(proportion=.10, num_boxes=16, box_width_range=(.3, 1.), box_length_range=(.08, 1.),
        box_height_range=(.005, .02), platform_width=SPAWN, border_width=.25),
    room=FurnitureRoomTerrainCfg(proportion=.15),
    clutter=ClutterTerrainCfg(proportion=.10),
)

# Seed 1 of 0-39 gives the tile mix closest to the proportions (6 rough, 7 hill, 6 stairs, 12 rooms, ...).
PARCOUR_TERRAIN = TerrainGeneratorCfg(seed=1, curriculum=False, size=(TILE, TILE), border_width=2., border_height=.3,
    num_rows=6, num_cols=10, sub_terrains=PARCOUR_SUB_TERRAINS, difficulty_range=(0., 1.), color_scheme='height',
    add_lights=True)


def tile_kinds(cfg=None):
    """{(row, col): (type, difficulty)} of the generated grid (deterministic for a seeded generator)."""
    from mjlab.terrains.terrain_generator import TerrainGenerator
    cfg = cfg or PARCOUR_TERRAIN
    names, kinds = list(cfg.sub_terrains), {}
    gen = TerrainGenerator(cfg)
    make = gen._create_terrain_geom

    def record(spec, pos, difficulty, sub, row, col):
        kinds[(row, col)] = (names[[id(c) for c in cfg.sub_terrains.values()].index(id(sub))], round(float(difficulty), 3))
        return make(spec, pos, difficulty, sub, row, col)
    gen._create_terrain_geom = record
    gen.compile(mujoco.MjSpec())
    return kinds


SPAWN_RADIUS = 1.0  # m: each episode starts at a random point this far from its tile centre, on the ground there
SPAWN_WINDOW = .10  # m: half-width of the patch under the robot whose ground decides the start height
SPAWN_MAX_STEP = .04  # m: largest ground range in that patch: one stair, a 10 deg slope or a book; not two stairs (an 8 cm drop), a shoe, cushion or furniture edge
SPAWN_MAX_RISE = .25  # m: largest start height against the tile base (a hill top, not a table, couch or chair seat)


def ground_map(cfg=None, res=.05, reach=TILE / 2 - .05):
    """Terrain heights (m, world z) around every tile centre on a res grid, from vertical lines against every terrain
    geom (all boxes; cables by their bounds). Returns dict(top, low, axis, origins): top and low are the highest and
    lowest ground within SPAWN_WINDOW of each point, [rows, cols, n, n] over axis offsets (x first, then y). raw is the
    height at each point, and slope the rise per run (dz/dx, dz/dy) of the top face that point stands on,
    [rows, cols, n, n, 2]: inclined boxes (hills, hollows, rolling ground) give their exact slope, level boxes (stairs,
    mats, sills, blocks) and cables zero, so a step edge is a height change and never a tilt."""
    from mjlab.terrains.terrain_generator import TerrainGenerator
    cfg = cfg or PARCOUR_TERRAIN
    gen, spec = TerrainGenerator(cfg), mujoco.MjSpec()
    gen.compile(spec)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    axis = np.arange(-reach, reach + 1e-9, res)
    origins = np.asarray(gen.terrain_origins, dtype=float)
    rows, cols = origins.shape[:2]
    gx, gy = np.meshgrid(axis, axis, indexing='ij')
    height = np.full((rows, cols, len(axis), len(axis)), -np.inf)
    slope = np.zeros((rows, cols, len(axis), len(axis), 2))
    top_z = 10.
    for g in range(m.ngeom):
        c, R, h = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g].copy()
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            corners = None
            ext = np.abs(R) @ np.array([h[0], h[0], h[0] + h[1]])  # bounds of the capsule
            lo, hi, zt = c[:2] - ext[:2], c[:2] + ext[:2], c[2] + ext[2]
        else:
            ext = np.abs(R) @ h
            lo, hi = c[:2] - ext[:2], c[:2] + ext[:2]
        for r in range(rows):
            for k in range(cols):
                o = origins[r, k]
                if hi[0] < o[0] - reach or lo[0] > o[0] + reach or hi[1] < o[1] - reach or lo[1] > o[1] + reach:
                    continue
                px, py = o[0] + gx, o[1] + gy
                inside = (px >= lo[0]) & (px <= hi[0]) & (py >= lo[1]) & (py <= hi[1])
                if not inside.any():
                    continue
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    higher = inside.copy()
                    higher[inside] = zt > height[r, k][inside]
                    height[r, k][higher], slope[r, k][higher] = zt, 0.
                    continue
                # A vertical line from top_z down, in the box frame: slab test, entry depth gives the top surface.
                p0 = (np.stack([px[inside], py[inside], np.full(inside.sum(), top_z)], -1) - c) @ R
                v = -R[2]  # world -z in box coordinates
                with np.errstate(divide='ignore', invalid='ignore'):
                    t1, t2 = (-h - p0) / v, (h - p0) / v
                lo_t = np.where(np.abs(v) < 1e-12, np.where(np.abs(p0) <= h, -np.inf, np.inf), np.minimum(t1, t2))
                hi_t = np.where(np.abs(v) < 1e-12, np.where(np.abs(p0) <= h, np.inf, -np.inf), np.maximum(t1, t2))
                enter, leave = lo_t.max(-1), hi_t.min(-1)
                z = np.where((enter <= leave) & (leave >= 0), top_z - enter, -np.inf)
                face = lo_t.argmax(-1)  # the slab the line enters last is the face it lands on
                n = R[:, face].T * -np.sign(v[face])[:, None]  # that face's outward normal (world)
                # A line on a box's edge can land on a side face: that is a step, not a tilt.
                grad = np.where(n[:, 2:] > .5, -n[:, :2] / np.maximum(n[:, 2:], 1e-6), 0.)
                sel = z > height[r, k][inside]
                higher = inside.copy()
                higher[inside] = sel
                height[r, k][higher], slope[r, k][higher] = z[sel], grad[sel]
    for _ in range(2):  # a line exactly on the seam of two blocks misses both (floating point): take the neighbours'
        miss = np.isinf(height)
        if not miss.any():
            break
        p = np.pad(height, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
        shifted = np.stack([p[:, :, 1 + i:1 + i + len(axis), 1 + j:1 + j + len(axis)] for i in (-1, 0, 1) for j in (-1, 0, 1)])
        ps = np.pad(slope, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)), mode='edge')
        best = shifted.argmax(0)
        near_slope = np.stack([ps[:, :, 1 + i:1 + i + len(axis), 1 + j:1 + j + len(axis)] for i in (-1, 0, 1) for j in (-1, 0, 1)])
        near_slope = np.take_along_axis(near_slope, best[None, ..., None], 0)[0]
        slope = np.where(miss[..., None], near_slope, slope)
        height = np.where(miss, shifted.max(0), height)
    w = int(round(SPAWN_WINDOW / res))
    pad = np.pad(height,((0, 0), (0, 0), (w, w), (w, w)), mode='edge')
    n = len(axis)
    shifts = [pad[:, :, i:i + n, j:j + n] for i in range(2 * w + 1) for j in range(2 * w + 1)]
    return dict(top=np.max(shifts, 0), low=np.min(shifts, 0), raw=height, slope=slope, axis=axis, origins=origins, res=res)


def spawn_valid(top, low, base):
    """Start patches the robot can stand on: a small ground range, and near the tile base height."""
    return (top - low < SPAWN_MAX_STEP) & (np.abs(top - base) < SPAWN_MAX_RISE)


def place_eval_worlds(env):
    """Put evaluation worlds on fixed tiles, the same in every run: per tile type its hardest tile, then a middle one,
    in the order of PARCOUR_SUB_TERRAINS (20 worlds cover both). Returns each world's tile."""
    import torch
    kinds = tile_kinds()
    by = {}
    for (r, c), (k, d) in kinds.items():
        by.setdefault(k, []).append((d, r, c))
    tiles = [sorted(by[k]) for k in PARCOUR_SUB_TERRAINS if k in by]
    picks = [t[-1] for t in tiles] + [t[(len(t) - 1) // 2] for t in tiles]
    terrain = env.scene.terrain
    command = env.command_manager.get_term('motion')
    gm = command.ground if getattr(command, 'ground', None) is not None else None
    out = []
    for w in range(env.num_envs):
        d, r, c = picks[w % len(picks)]
        terrain.terrain_levels[w], terrain.terrain_types[w] = r, c
        terrain.env_origins[w] = terrain.terrain_origins[r, c]
        record = dict(tile=kinds[(r, c)][0], difficulty=d, row=r, col=c)
        if gm is not None:  # a fixed start point per world, drawn from its own seed, on ground the robot can stand on
            rng = np.random.default_rng(1000 + w)
            top, low, axis = gm['top'][r, c], gm['low'][r, c], gm['axis']
            ok = spawn_valid(top, low, gm['origins'][r, c, 2])
            ok &= (np.abs(axis)[:, None] <= SPAWN_RADIUS) & (np.abs(axis)[None, :] <= SPAWN_RADIUS)
            cells = np.argwhere(ok)
            i, j = cells[rng.integers(len(cells))] if len(cells) else (len(axis) // 2, len(axis) // 2)
            command.fixed_spawn[w] = torch.tensor([axis[i], axis[j]], device=command.fixed_spawn.device)
            record.update(spawn_xy_m=[round(float(axis[i]), 2), round(float(axis[j]), 2)],
                spawn_ground_m=round(float(top[i, j] - gm['origins'][r, c, 2]), 3))
        out.append(record)
    return out


def render(out, size=900):
    """PNG views for review: the whole grid, then each element at its hardest setting with the long-arm APE0
    standing on the spawn square (kinematic stand pose, not a controller)."""
    import sys
    from pathlib import Path
    from PIL import Image
    from mjlab.terrains.terrain_generator import TerrainGenerator
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tracking
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    views = []
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = spec.visual.global_.offheight = size
    TerrainGenerator(PARCOUR_TERRAIN).compile(spec)
    views.append(('grid', None))
    names = list(PARCOUR_SUB_TERRAINS)
    hard = TerrainGeneratorCfg(seed=3, curriculum=True, size=(TILE, TILE), num_rows=1, num_cols=len(names), border_width=0.,
        sub_terrains=PARCOUR_SUB_TERRAINS, difficulty_range=(1., 1.), color_scheme='height')
    gen = TerrainGenerator(hard)
    single = mujoco.MjSpec()
    single.visual.global_.offwidth = single.visual.global_.offheight = size
    gen.compile(single)
    robot = mujoco.MjSpec.from_file(str(tracking.MODEL / 'cove50-rigid.xml'))
    robot = tracking.add_fists(tracking.scale_arms(tracking.scale_feet(robot, (1.3, 1.3)), (1.6, 1.27)), .03, .01)
    for c, name in enumerate(names):
        o = gen.terrain_origins[0, c]
        frame = single.worldbody.add_frame(pos=o)  # the robot's root body carries its stand height
        frame.attach_body(robot.copy().worldbody.first_body(), f'r{c}_', '')
    for sp in (single, spec):
        sp.worldbody.add_light(pos=(0, 0, 12), dir=(0, 0, -1), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=(.6, .6, .6))
    model, grid_model = single.compile(), spec.compile()
    for m in (model, grid_model):
        m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:] = (.45, .45, .45), (.5, .5, .5)
        m.vis.map.zfar = 200.
    shots = []
    with mujoco.Renderer(grid_model, size, size) as r:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (0, 0, 0)
        cam.distance, cam.azimuth, cam.elevation = 30., 45., -50.
        grid_data = mujoco.MjData(grid_model)
        mujoco.mj_forward(grid_model, grid_data)
        r.update_scene(grid_data, cam)
        Image.fromarray(r.render()).save(out / 'grid.png')
        shots.append('grid.png')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, size, size) as r:
        for c, name in enumerate(names):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = gen.terrain_origins[0, c] + np.array([0, 0, .2])
            cam.distance, cam.azimuth, cam.elevation = (3.2 if name == 'room' else 2.2), 135., -30.
            r.update_scene(data, cam)
            Image.fromarray(r.render()).save(out / f'{name}.png')
            shots.append(f'{name}.png')
    from PIL import ImageDraw
    tiles = [Image.open(out / f).resize((size // 2, size // 2)) for f in shots]
    sheet = Image.new('RGB', (5 * (size // 2), 2 * (size // 2)), 'white')
    for i, (f, im) in enumerate(zip(shots, tiles)):
        sheet.paste(im, ((i % 5) * (size // 2), (i // 5) * (size // 2)))
        ImageDraw.Draw(sheet).text(((i % 5) * (size // 2) + 10, (i // 5) * (size // 2) + 10), f[:-4].replace('_', ' '), fill='white')
    sheet.save(out / 'sheet.png')
    return shots + ['sheet.png']


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Render the APE0 obstacle world for review')
    p.add_argument('--render', required=True, help='output folder for PNG views')
    print(render(p.parse_args().render))
