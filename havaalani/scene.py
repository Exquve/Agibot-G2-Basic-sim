"""Ayırma holünü MuJoCo XML'ine döker. Robot modeli ayrıca dahil edilir."""

import math
from pathlib import Path

import numpy as np

from layout import (
    BAG_HALF,
    BAG_MASSES,
    BELT_TOP,
    GATES,
    INCOMING_HALF_W,
    INCOMING_X0,
    INCOMING_X1,
    belt_direction,
    hex_vertices,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "vendor" / "mujoco"
SCENE_PATH = MODEL_DIR / "airport.xml"


def _quat_z(yaw):
    half = 0.5 * float(yaw)
    return f"{np.cos(half):.6f} 0 0 {np.sin(half):.6f}"


def _rgba(color):
    return " ".join(f"{float(c):.4f}" for c in color)


def _box(name, center, half, yaw, rgba, collide=False):
    quat = _quat_z(yaw)
    ctype = 'contype="1" conaffinity="1"' if collide else 'contype="0" conaffinity="0"'
    return (
        f'    <geom name="{name}" type="box" pos="{center[0]:.4f} {center[1]:.4f} {center[2]:.4f}" '
        f'quat="{quat}" size="{half[0]:.4f} {half[1]:.4f} {half[2]:.4f}" rgba="{_rgba(rgba)}" {ctype}/>'
    )


def _belt_geom(name, p0, p1, half_w, rgba, top=BELT_TOP, thick=0.05):
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    yaw = math.atan2(float(delta[1]), float(delta[0]))
    center = np.array([(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, top - thick / 2])
    return _box(name, center, (length / 2, half_w, thick / 2), yaw, rgba, collide=True)


def _chevrons(name, p0, p1, half_w, spacing=0.52):
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    direction = delta / length
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    lines = []
    ids = []
    cursor = 0.35
    index = 0
    while cursor < length - 0.25:
        pos = p0 + direction * cursor
        center = np.array([pos[0], pos[1], BELT_TOP + 0.006])
        gid = f"{name}_{index}"
        lines.append(_box(gid, center, (0.07, min(0.10, half_w * 0.45), 0.004), yaw, (0.95, 0.78, 0.15, 1)))
        ids.append(gid)
        cursor += spacing
        index += 1
    return lines, ids


def _arch(name, end, direction, color):
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    side = np.array([-direction[1], direction[0]])
    lines = []
    span = 0.42
    for sign, tag in ((-1, "l"), (1, "r")):
        foot = end + side * sign * span
        lines.append(
            f'    <geom name="{name}_{tag}" type="cylinder" pos="{foot[0]:.3f} {foot[1]:.3f} 0.70" '
            f'size="0.045 0.70" rgba="{_rgba(color)}"/>'
        )
    lines.append(_box(f"{name}_top", np.array([end[0], end[1], 1.42]), (span + 0.08, 0.055, 0.045), yaw + np.pi / 2, color))
    panel = end - direction * 0.02
    lines.append(_box(f"{name}_panel", np.array([panel[0], panel[1], 1.68]), (0.02, 0.28, 0.16), yaw, color))
    return lines


def _bag(index, x, mass):
    z = BELT_TOP + BAG_HALF[2] + 0.02
    shell = _rgba((0.7, 0.7, 0.72, 1))
    return f"""
    <body name="bag{index}" pos="{x:.3f} 0 {z:.3f}">
      <freejoint name="bag{index}_free"/>
      <inertial pos="0 0 0" mass="{mass:.1f}" diaginertia="0.08 0.12 0.14"/>
      <geom name="bag{index}_shell" type="box" size="{BAG_HALF[0]:.3f} {BAG_HALF[1]:.3f} {BAG_HALF[2]:.3f}" rgba="{shell}" friction="0.9 0.02 0.001" solref="0.02 1" contype="1" conaffinity="1"/>
      <geom name="bag{index}_lid" type="box" pos="0 0 {BAG_HALF[2] + 0.008:.3f}" size="{BAG_HALF[0] - 0.012:.3f} {BAG_HALF[1] - 0.008:.3f} 0.006" rgba="0.12 0.12 0.13 1" contype="0" conaffinity="0"/>
      <geom name="bag{index}_handle" type="capsule" fromto="-{BAG_HALF[0] * 0.28:.3f} 0 {BAG_HALF[2] + 0.032:.3f} {BAG_HALF[0] * 0.28:.3f} 0 {BAG_HALF[2] + 0.032:.3f}" size="0.012" rgba="0.18 0.18 0.2 1" contype="0" conaffinity="0"/>
      <geom name="bag{index}_tag" type="box" pos="{BAG_HALF[0] + 0.012:.3f} 0 {BAG_HALF[2] * 0.15:.3f}" size="0.012 0.055 0.045" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>
    </body>"""


def _hex_vertex_text():
    top = BELT_TOP
    bottom = BELT_TOP - 0.10
    coords = []
    for vertex in hex_vertices():
        coords.append(f"{vertex[0]:.4f} {vertex[1]:.4f} {bottom:.4f}")
    for vertex in hex_vertices():
        coords.append(f"{vertex[0]:.4f} {vertex[1]:.4f} {top:.4f}")
    return " ".join(coords)


def build_scene_xml(bag_count=4):
    parts = []
    chevron_names = []

    parts.append(
        _belt_geom(
            "belt_in",
            np.array([INCOMING_X1, 0.0]),
            np.array([INCOMING_X0, 0.0]),
            INCOMING_HALF_W,
            (0.07, 0.075, 0.08, 1),
        )
    )
    incoming_chevrons, names = _chevrons(
        "chev_in",
        np.array([INCOMING_X1 - 0.3, 0.0]),
        np.array([INCOMING_X0 + 0.15, 0.0]),
        INCOMING_HALF_W,
    )
    parts.extend(incoming_chevrons)
    chevron_names.extend(names)

    parts.append(
        '    <geom name="sort_deck" type="mesh" mesh="hexdeck" rgba="0.90 0.91 0.92 1" contype="1" conaffinity="1"/>'
    )
    # Altıncı kenar: robotun durduğu yer. Bant yok, bavul buradan geçmez.
    west = float(np.min(hex_vertices()[:, 0]))
    parts.append(
        _box(
            "station",
            np.array([(0.02 + west) / 2, 0.0, BELT_TOP - 0.055]),
            ((west - 0.02) / 2, 0.36, 0.045),
            0.0,
            (0.70, 0.72, 0.74, 1),
            collide=True,
        )
    )

    for gate in GATES:
        direction = belt_direction(gate)
        parts.append(_belt_geom(f"belt_{gate['key']}", gate["p0"], gate["p1"], gate["half_w"], (0.08, 0.085, 0.09, 1)))
        lines, names = _chevrons(f"chev_{gate['key']}", gate["p0"], gate["p1"], gate["half_w"])
        parts.extend(lines)
        chevron_names.extend(names)
        parts.extend(_arch(f"gate_{gate['key']}", gate["p1"], direction, gate["color"]))
        # Yan korkuluk, ağzın 0.55 m ilerisinden başlar ki bavul banda girebilsin.
        yaw = math.atan2(float(direction[1]), float(direction[0]))
        side = np.array([-direction[1], direction[0]])
        length = float(np.linalg.norm(gate["p1"] - gate["p0"]))
        rail_from = 1.25
        rail_to = length - 0.35
        if rail_to > rail_from:
            mid = gate["p0"] + direction * ((rail_from + rail_to) / 2)
            half_len = (rail_to - rail_from) / 2
            for sign, tag in ((-1, "a"), (1, "b")):
                pos = mid + side * sign * (gate["half_w"] + 0.03)
                parts.append(
                    _box(
                        f"rail_{gate['key']}_{tag}",
                        np.array([pos[0], pos[1], BELT_TOP + 0.09]),
                        (half_len, 0.025, 0.08),
                        yaw,
                        (0.85, 0.72, 0.18, 1),
                        collide=True,
                    )
                )

    # Giriş bandı korkuluğu, ayırma masasına kadar gelmez.
    rail_y = INCOMING_HALF_W + 0.03
    rail_center_x = (INCOMING_X0 + 0.25 + INCOMING_X1) / 2
    rail_half_x = (INCOMING_X1 - (INCOMING_X0 + 0.25)) / 2
    for sign, tag in ((1, "l"), (-1, "r")):
        parts.append(
            _box(
                f"rail_in_{tag}",
                np.array([rail_center_x, sign * rail_y, BELT_TOP + 0.09]),
                (rail_half_x, 0.025, 0.08),
                0.0,
                (0.85, 0.72, 0.18, 1),
                collide=True,
            )
        )

    # Salon
    parts.append(_box("wall_back", np.array([-5.15, 0.0, 1.45]), (0.08, 6.4, 1.45), 0.0, (0.62, 0.64, 0.66, 1)))
    for index, y in enumerate(np.linspace(-3.2, 3.2, 5)):
        parts.append(_box(f"window_{index}", np.array([-5.05, y, 1.75]), (0.02, 0.48, 0.42), 0.0, (0.55, 0.72, 0.82, 0.55)))
    parts.append(_box("station_mark", np.array([-0.02, 0.0, 0.004]), (0.16, 0.22, 0.003), 0.0, (0.90, 0.75, 0.12, 0.9)))

    bags = []
    masses = BAG_MASSES
    spawn_x = [2.15, 3.57, 4.99, 6.41]
    for index in range(bag_count):
        bags.append(_bag(index, spawn_x[index], masses[index % len(masses)]))

    body = "\n".join(parts)
    bag_xml = "\n".join(bags)
    xml = f"""<mujoco model="g2_airport_sort">
  <include file="G2_place_workpiece.xml"/>

  <statistic center="0.8 0 0.8" extent="5.5"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.38 0.40 0.42" specular="0.05 0.05 0.05"/>
    <rgba haze="0.55 0.62 0.68 1"/>
    <global azimuth="135" elevation="-24" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.62 0.70 0.78" rgb2="0.90 0.92 0.94" width="512" height="512"/>
    <texture name="floortex" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.20 0.22 0.24" rgb2="0.15 0.16 0.18" mark="edge" markrgb="0.30 0.32 0.34"/>
    <material name="floor" texture="floortex" texuniform="true" texrepeat="16 16" reflectance="0.04"/>
    <mesh name="hexdeck" vertex="{_hex_vertex_text()}"/>
  </asset>

  <worldbody>
    <light name="hall" pos="2.5 0 6.5" dir="0 0 -1" directional="true" diffuse="0.7 0.7 0.68"/>
    <light name="sort_spot" pos="0.9 0 3.4" dir="0.1 0 -1" diffuse="0.9 0.9 0.85" specular="0.2 0.2 0.2" cutoff="55"/>
    <geom name="floor" type="plane" size="14 14 0.1" material="floor" condim="3" friction="1 0.005 0.0001"/>
{body}
{bag_xml}
  </worldbody>
</mujoco>
"""
    return xml, chevron_names


def write_scene():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    xml, chevrons = build_scene_xml()
    SCENE_PATH.write_text(xml)
    return SCENE_PATH, chevrons
