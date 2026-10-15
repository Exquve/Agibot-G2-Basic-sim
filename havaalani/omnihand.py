"""G2 bileğine Agibot OmniHand takar.

GenieSim G2 paketinde yalnızca iş parçası kıskacı var. Beş parmaklı elin
mesh'leri AgibotTech/agibot_x2_urdf içindeki OmniHand modelinden gelir
(Mulan PSL v2). Parmaklar sabit, hafif açık bir duruşta pişirilir; ek
eklem eklenmez, böylece G2'nin qpos düzeni yalnızca kıskaç kızaklarını kaybeder.
"""

from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "vendor" / "mujoco"
HAND = MODEL / "robot" / "parts" / "G2" / "omnihand"
MESH_DIR = MODEL / "robot" / "parts" / "meshes" / "omnihand"
URDF_URL = (
    "https://raw.githubusercontent.com/AgibotTech/agibot_x2_urdf/main/"
    "X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf"
)
MESH_URL = (
    "https://raw.githubusercontent.com/AgibotTech/agibot_x2_urdf/main/"
    "X2_URDF-v1.4.0/meshes/{name}"
)
LICENSE_URL = "https://raw.githubusercontent.com/AgibotTech/agibot_x2_urdf/main/LICENSE"
MESHES = (
    "l_palm.stl",
    "r_palm.stl",
    "l_thumb_roll.stl",
    "r_thumb_roll.stl",
    "finger_abad.stl",
    "finger_pip.stl",
    "finger_dip.stl",
    "thumb_abad.stl",
    "thumb_mcp.stl",
    "thumb_pip.stl",
    "thumb_dip.stl",
)

# Bilek flanşı link7'nin +X ucunda, x=0.095. Elin arkası flanşa değer.
FLANGE_X = 0.095
# Bilek ağzı flanşın içine bir santim oturur.
GAP = -0.010


def _download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    urllib.request.urlretrieve(url, destination)


def ensure_files():
    _download(URDF_URL, HAND / "source.urdf")
    _download(LICENSE_URL, MESH_DIR / "LICENSE")
    for name in MESHES:
        _download(MESH_URL.format(name=name), MESH_DIR / name)


def _rpy(rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _transform(xyz, rpy):
    out = np.eye(4)
    out[:3, :3] = _rpy(np.asarray(rpy, dtype=float))
    out[:3, 3] = np.asarray(xyz, dtype=float)
    return out


def _axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8 or abs(angle) < 1e-8:
        return np.eye(4)
    x, y, z = axis / norm
    c, s = np.cos(angle), np.sin(angle)
    v = 1.0 - c
    rotation = np.array(
        [
            [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
        ]
    )
    out = np.eye(4)
    out[:3, :3] = rotation
    return out


def _quat(rotation):
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=float)
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat = -quat
    return quat


def _numbers(text):
    return [float(item) for item in text.split()]


def _parse(urdf_path):
    root = ET.parse(urdf_path).getroot()
    visuals = {}
    for link in root.findall("link"):
        items = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            origin = visual.find("origin")
            xyz = _numbers(origin.get("xyz", "0 0 0")) if origin is not None else [0, 0, 0]
            rpy = _numbers(origin.get("rpy", "0 0 0")) if origin is not None else [0, 0, 0]
            items.append((_transform(xyz, rpy), Path(mesh.get("filename")).name))
        visuals[link.get("name")] = items
    children = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        xyz = _numbers(origin.get("xyz", "0 0 0")) if origin is not None else [0, 0, 0]
        rpy = _numbers(origin.get("rpy", "0 0 0")) if origin is not None else [0, 0, 0]
        axis_node = joint.find("axis")
        axis = _numbers(axis_node.get("xyz", "0 0 1")) if axis_node is not None else [0, 0, 1]
        limit = joint.find("limit")
        lower = float(limit.get("lower", "0")) if limit is not None else 0.0
        upper = float(limit.get("upper", "0")) if limit is not None else 0.0
        mimic = joint.find("mimic")
        parent = joint.find("parent").get("link")
        children.setdefault(parent, []).append(
            {
                "name": joint.get("name"),
                "child": joint.find("child").get("link"),
                "origin": _transform(xyz, rpy),
                "axis": axis,
                "lower": lower,
                "upper": upper,
                "mimic": None
                if mimic is None
                else (mimic.get("joint"), float(mimic.get("multiplier", "1")), float(mimic.get("offset", "0"))),
            }
        )
    return visuals, children


def _rest(joint, solved):
    if joint["mimic"] is not None:
        source, multiplier, offset = joint["mimic"]
        return multiplier * solved[source] + offset
    name = joint["name"].lower()
    lower, upper = joint["lower"], joint["upper"]
    if "thumb_roll" in name:
        return float(np.clip(-0.55 if lower < -0.3 else 0.55, lower, upper))
    if "thumb_abad" in name:
        return float(np.clip(1.40 if upper > 1.0 else -1.40, lower, upper))
    if "thumb_mcp" in name:
        return float(np.clip(-0.32 if lower < 0.0 else 0.32, lower, upper))
    if "abad" in name:
        spread = 0.0
        for finger, amount in (("index", 0.0), ("middle", 0.04), ("ring", 0.08), ("pinky", 0.14)):
            if finger in name:
                spread = amount
        return float(np.clip(spread if upper > 0.0 else -spread, lower, upper))
    if "pip" in name or "dip" in name:
        return float(np.clip(0.18 if upper > 0.0 else -0.18, lower, upper))
    return 0.0


def _solve(children, palm):
    solved = {}

    def walk(link):
        for joint in children.get(link, []):
            solved[joint["name"]] = _rest(joint, solved)
            walk(joint["child"])

    walk(palm)
    # Mimic kaynakları çocuklardan önce gelmeyebilir; iki tur yeter.
    for _ in range(2):
        walk(palm)
    return solved


def _collect(visuals, children, palm, solved):
    # Gövde çerçevesi: +Z avuç içi (URDF +X), +X parmaklar (URDF +Z).
    # p_body = (z, -y, x)_urdf
    remap = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    geoms = []

    def walk(link, pose):
        for visual_pose, mesh_name in visuals.get(link, []):
            full = pose @ visual_pose
            geoms.append((remap @ full[:3, :3], remap @ full[:3, 3], mesh_name))
        for joint in children.get(link, []):
            child = pose @ joint["origin"] @ _axis_rotation(joint["axis"], solved[joint["name"]])
            walk(joint["child"], child)

    walk(palm, np.eye(4))
    return geoms


def _stl_vertices(path):
    data = path.read_bytes()
    count = int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0])
    triangles = np.ndarray((count, 3, 3), dtype="<f4", buffer=data, offset=96, strides=(50, 12, 4))
    return triangles.reshape(-1, 3).astype(float)


def _cloud(geoms, palm_only=False):
    points = []
    for rotation, origin, mesh_name in geoms:
        if palm_only and "palm" not in mesh_name:
            continue
        vertices = _stl_vertices(MESH_DIR / mesh_name)
        points.append(vertices @ rotation.T + origin)
    return np.concatenate(points, axis=0)


def _fmt(values):
    return " ".join(f"{float(value):.6f}" for value in values)


# Elin yerel eksenleri: +X avuç derisi (temas yüzü), +Z parmak uzantısı.
# link7 çerçevesinde kol ekseni +X, yan eksen +Y, aşağı eksen -Z'dir.
# Eskiden el birim duruşla bağlanıyordu: parmaklar yana, avuç kola bakıyordu;
# yani bileğe 90 derece ters bağlıydı. Mount matrisi eli önkola paralel dizer:
# sütunlar = [parmak ekseni, -avuç ekseni, kol ekseni] (link7 çerçevesinde).
MOUNT_L = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
# Sağ kolda yan eksen gövdenin diğer tarafına bakar; avuç yine içe dönsün diye ayna.
MOUNT_R = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _write_hand(side, body_name, geoms, mount_x, mount):
    quat = _quat(mount)
    # Geomlar artık montaj çerçevesine yazılır: r' = M^T r, t' = M^T t.
    lines = [
        f'<mujoco model="omnihand_{side}">',
        f'  <body name="{body_name}" pos="{mount_x:.4f} 0 0" quat="{_fmt(quat)}">',
        '    <inertial pos="0.04 0 0.015" mass="0.45" diaginertia="0.0012 0.0008 0.0006"/>',
    ]
    for rotation, origin, mesh_name in geoms:
        local_rotation = mount.T @ rotation
        local_origin = mount.T @ origin
        stem = mesh_name.replace(".stl", "")
        rgba = "0.08 0.08 0.09 1" if "palm" in stem else "0.14 0.14 0.15 1"
        lines.append(
            "    <geom type=\"mesh\" contype=\"0\" conaffinity=\"0\" group=\"1\" density=\"0\" "
            f'mesh="omni_{stem}" pos="{_fmt(local_origin)}" quat="{_fmt(_quat(local_rotation))}" rgba="{rgba}"/>'
        )
    lines.append("  </body>")
    lines.append("</mujoco>")
    (HAND / f"{side}.xml").write_text("\n".join(lines) + "\n")


def _write_assets():
    lines = ['<mujoco model="omnihand_asset">']
    for name in MESHES:
        stem = name.replace(".stl", "")
        lines.append(f'  <mesh name="omni_{stem}" file="../../meshes/omnihand/{name}"/>')
    lines.append("</mujoco>")
    (HAND / "asset.xml").write_text("\n".join(lines) + "\n")


def _replace_once(path, old, new):
    text = path.read_text()
    if old not in text:
        return
    path.write_text(text.replace(old, new, 1))


def _patch_g2():
    robot = MODEL / "G2_place_workpiece.xml"
    _replace_once(
        robot,
        '<include file="robot/parts/G2/hand_pw/asset.xml" />',
        '<include file="robot/parts/G2/omnihand/asset.xml" />',
    )
    _replace_once(
        robot,
        '    <include file="robot/parts/G2/hand_pw/actuator_left_sensor.xml" />\n'
        '    <include file="robot/parts/G2/hand_pw/actuator_right_sensor.xml" />\n',
        "",
    )
    _replace_once(
        robot,
        """        0 0 0 0 0 0 0
        0 0
        0 0 0 0 0 0 0
        0 0
        0 0 0""",
        """        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
        0 0 0""",
    )
    arm = MODEL / "robot/parts/G2/arm_crs_pw/link.xml"
    _replace_once(
        arm,
        '<include file="../hand_pw/gripper_left.xml" />',
        '<include file="../omnihand/left.xml" />',
    )
    _replace_once(
        arm,
        '<include file="../hand_pw/gripper_right.xml" />',
        '<include file="../omnihand/right.xml" />',
    )


def install_omnihand():
    """Mesh'leri indirir, eli üretir ve G2 XML'ine bağlar. Ölçüleri döndürür."""
    marker = HAND / "left.xml"
    script = Path(__file__).resolve()
    if marker.exists() and marker.stat().st_mtime >= script.stat().st_mtime and (HAND / "right.xml").exists():
        _patch_g2()
        return None
    ensure_files()
    visuals, children = _parse(HAND / "source.urdf")
    left_angles = _solve(children, "L_palm")
    right_angles = _solve(children, "R_palm")
    left = _collect(visuals, children, "L_palm", left_angles)
    right = _collect(visuals, children, "R_palm", right_angles)
    cloud = _cloud(left)
    palm = _cloud(left, palm_only=True)
    reach = float(cloud[:, 2].max())
    back = float(cloud[:, 2].min())
    # Bilek, parmak ekseninin kol tarafındaki ucudur.
    wrist = float(cloud[:, 0].min())
    mount_x = FLANGE_X - wrist + GAP
    _write_assets()
    _write_hand("left", "gripper_l_center_link", left, mount_x, MOUNT_L)
    _write_hand("right", "gripper_r_center_link", right, mount_x, MOUNT_R)
    _patch_g2()
    return {
        "reach": reach,
        "back": back,
        "wrist": wrist,
        "mount_x": mount_x,
        # Yeni dik bileşkede avuç derisi elin yerel +X'inde; parmaklar +Z'de.
        "lift": float(np.median(palm[:, 0])),
        "palm_reach": float(cloud[:, 0].max()),
    }


if __name__ == "__main__":
    fit = install_omnihand()
    print(
        f"reach {fit['reach']:.4f}  palm {fit['palm_reach']:.4f}  wrist {fit['wrist']:.4f}  "
        f"mount {fit['mount_x']:.4f}  lift {fit['lift']:.4f}"
    )
