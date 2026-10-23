"""Engineering interchange generated from the common tree, never hand edited."""

import json
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, indent, tostring
from .robot import Robot, inertia


def vec(values):
    return " ".join(f"{v:.9g}" for v in values)


def write_xml(root, path):
    indent(root)
    Path(path).write_text(tostring(root, encoding="unicode") + "\n")


def mjcf(robot: Robot, path, supported=False):
    c = robot.config
    root = Element("mujoco", model=c.name)
    SubElement(root, "compiler", angle="radian", autolimits="true")
    SubElement(
        root,
        "option",
        timestep=str(1 / c.control.physics_hz),
        gravity="0 0 -9.81",
        integrator="implicitfast",
    )
    visual = SubElement(root, "visual")
    SubElement(visual, "global", offwidth="960", offheight="720")
    SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    default = SubElement(root, "default")
    SubElement(default, "joint", damping="0.04", armature="0.002")
    SubElement(
        default,
        "geom",
        friction="0.8 0.01 0.001",
        condim="3",
        material="matte",
    )
    asset = SubElement(root, "asset")
    SubElement(asset, "material", name="matte", specular="0.12", shininess="0.12")
    SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="flat",
        rgb1="0.88 0.90 0.88",
        width="512",
        height="512",
    )
    world = SubElement(root, "worldbody")
    SubElement(
        world,
        "light",
        pos="1 -1 2",
        dir="-0.3 0.3 -1",
        diffuse="0.9 0.9 0.9",
        ambient="0.3 0.3 0.3",
    )
    SubElement(
        world,
        "light",
        pos="-1 1 1",
        dir="0 0 -1",
        diffuse="0.5 0.5 0.5",
        castshadow="false",
    )
    SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="2 2 0.1",
        rgba="0.65 0.70 0.69 1",
    )
    bodies = {}
    for link in robot.links:
        parent = bodies[link.parent] if link.parent else world
        pos = link.pos if link.parent else (0, 0, robot.base_height + 0.002)
        body = SubElement(parent, "body", name=link.name, pos=vec(pos))
        bodies[link.name] = body
        if not link.parent and not supported:
            SubElement(body, "freejoint", name="root")
        if link.joint:
            j = link.joint
            attr = dict(name=j.name, type="hinge", axis=vec(j.axis))
            attr.update(limited="false") if j.continuous else attr.update(
                range=vec(j.limits)
            )
            SubElement(body, "joint", **attr)
        if link.mass_properties is None:
            center, diag = inertia(link)
            inertial = {
                "pos": vec(center),
                "mass": str(link.mass),
                "diaginertia": vec(diag),
            }
        else:
            prop = link.mass_properties
            tensor = prop["tensor_kg_m2"]
            inertial = {
                "pos": vec(prop["center_m"]),
                "mass": str(link.mass),
                "fullinertia": vec(
                    [
                        tensor[0][0],
                        tensor[1][1],
                        tensor[2][2],
                        tensor[0][1],
                        tensor[0][2],
                        tensor[1][2],
                    ]
                ),
            }
        SubElement(body, "inertial", **inertial)
        for i, g in enumerate(link.geometry):
            size = (
                tuple(v / 2 for v in g.size)
                if g.shape == "box"
                else (g.size[0], g.size[1] / 2)
                if g.shape == "cylinder"
                else g.size
            )
            mesh = Path(path).parent / "meshes" / f"{link.name}_{i}.stl"
            if mesh.exists():
                mesh_name = f"{link.name}_{i}_cad"
                SubElement(
                    asset,
                    "mesh",
                    name=mesh_name,
                    file=f"meshes/{mesh.name}",
                    scale="0.001 0.001 0.001",
                )
                SubElement(
                    body,
                    "geom",
                    name=mesh_name,
                    type="mesh",
                    mesh=mesh_name,
                    pos=vec(g.pos),
                    euler=vec(g.rpy),
                    rgba=vec(g.color),
                    contype="0",
                    conaffinity="0",
                    mass="0",
                )
            # Physics keeps simple primitives; visuals use the actual CAD mesh when available.
            SubElement(
                body,
                "geom",
                name=f"{link.name}_{i}",
                type=g.shape,
                size=vec(size),
                pos=vec(g.pos),
                euler=vec(g.rpy),
                rgba=vec((*g.color[:3], 0) if mesh.exists() else g.color),
                contype="1" if g.collision else "0",
                conaffinity="1" if g.collision else "0",
            )
        if link.name == "head":
            # MuJoCo camera looks along local -Z; rotate it to face robot +X.
            SubElement(
                body,
                "camera",
                name="head_camera",
                pos=vec(robot.camera_position),
                xyaxes="0 -1 0 0 0 1",
                fovy=str(robot.camera_fovy_deg),
            )
        if link.name == "torso":
            SubElement(
                body, "site", name="imu", pos=vec(robot.imu_position), size="0.004"
            )
    # Ignore only nonphysical neutral overlap of colocated joint housings.
    contact = SubElement(root, "contact")
    for link in robot.links:
        if link.parent:
            SubElement(contact, "exclude", body1=link.name, body2=link.parent)
        if link.name.endswith("upper_arm"):
            SubElement(contact, "exclude", body1=link.name, body2="torso")
        if link.name.endswith("thigh") and c.morphology == "walking":
            SubElement(contact, "exclude", body1=link.name, body2="torso")
    actuator = SubElement(root, "actuator")
    for j in robot.joints:
        limit = c.actuators[j.family].effort_limit_nm
        SubElement(
            actuator,
            "motor",
            name=j.name + "_motor",
            joint=j.name,
            gear="1",
            ctrlrange=vec((-limit, limit)),
        )
    sensors = SubElement(root, "sensor")
    SubElement(sensors, "gyro", name="imu_gyro", site="imu")
    SubElement(sensors, "accelerometer", name="imu_accel", site="imu")
    for j in robot.joints:
        SubElement(sensors, "jointpos", name=j.name + "_position", joint=j.name)
        SubElement(sensors, "jointvel", name=j.name + "_velocity", joint=j.name)
    write_xml(root, path)


def urdf(robot: Robot, path):
    root = Element("robot", name=robot.config.name)
    for link in robot.links:
        elem = SubElement(root, "link", name=link.name)
        if link.mass_properties is None:
            center, diag = inertia(link)
            tensor = [[diag[0], 0, 0], [0, diag[1], 0], [0, 0, diag[2]]]
        else:
            center, tensor = (
                link.mass_properties["center_m"],
                link.mass_properties["tensor_kg_m2"],
            )
        mass = SubElement(elem, "inertial")
        SubElement(mass, "origin", xyz=vec(center))
        SubElement(mass, "mass", value=str(link.mass))
        SubElement(
            mass,
            "inertia",
            ixx=str(tensor[0][0]),
            iyy=str(tensor[1][1]),
            izz=str(tensor[2][2]),
            ixy=str(tensor[0][1]),
            ixz=str(tensor[0][2]),
            iyz=str(tensor[1][2]),
        )
        for i, g in enumerate(link.geometry):
            for kind in ["visual", "collision"] if g.collision else ["visual"]:
                entry = SubElement(elem, kind, name=f"{link.name}_{i}")
                SubElement(entry, "origin", xyz=vec(g.pos), rpy=vec(g.rpy))
                geom = SubElement(entry, "geometry")
                if g.shape == "box":
                    SubElement(geom, "box", size=vec(g.size))
                elif g.shape == "sphere":
                    SubElement(geom, "sphere", radius=str(g.size[0]))
                elif g.shape == "ellipsoid":
                    # URDF has no ellipsoid primitive. Keep exact CAD visual mesh when built;
                    # conservative box fallback is explicit in the interface fidelity notes.
                    mesh = Path(path).parent / "meshes" / f"{link.name}_{i}.stl"
                    if kind == "visual" and mesh.exists():
                        SubElement(
                            geom,
                            "mesh",
                            filename=f"meshes/{mesh.name}",
                            scale="0.001 0.001 0.001",
                        )
                    else:
                        SubElement(geom, "box", size=vec(tuple(2 * v for v in g.size)))
                else:
                    SubElement(
                        geom, "cylinder", radius=str(g.size[0]), length=str(g.size[1])
                    )
                if kind == "visual":
                    SubElement(
                        SubElement(entry, "material", name=f"{link.name}_{i}_color"),
                        "color",
                        rgba=vec(g.color),
                    )
        if link.parent:
            j = link.joint
            joint = SubElement(
                root,
                "joint",
                name=j.name,
                type="continuous" if j.continuous else "revolute",
            )
            SubElement(joint, "parent", link=link.parent)
            SubElement(joint, "child", link=link.name)
            SubElement(joint, "origin", xyz=vec(link.pos), rpy="0 0 0")
            SubElement(joint, "axis", xyz=vec(j.axis))
            a = robot.config.actuators[j.family]
            limits = dict(
                effort=str(a.effort_limit_nm), velocity=str(a.velocity_limit_rad_s)
            )
            if not j.continuous:
                limits.update(lower=str(j.limits[0]), upper=str(j.limits[1]))
            SubElement(joint, "limit", **limits)
    write_xml(root, path)


def export_all(robot, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved-config.json").write_text(robot.config.model_dump_json(indent=2))
    (output / "interfaces.json").write_text(json.dumps(robot.interface(), indent=2))
    mjcf(robot, output / "robot.xml")
    mjcf(robot, output / "supported.xml", supported=True)
    urdf(robot, output / "robot.urdf")
