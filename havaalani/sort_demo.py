"""G2 havaalanı bavul ayırma gösterisi.

Robot, Agibot Genie Sim G2 modelidir ve yerinde durur. Kollar, modelin
konum servo eklemleriyle sürülür. Giriş bandı bavulu ayırma masasına getirir;
hangi çıkışa gideceğine G2 karar verir ve kollarıyla o banda iter.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer

from layout import (
    BAG_HALF,
    BAG_MASSES,
    BAG_Z,
    FLIGHTS,
    GATES,
    GRIP_FACE,
    IN_SPEED,
    INCOMING_X1,
    OUT_SPEED,
    PALM,
    PALM_SKIN,
    QUEUE_GAP,
    STOP_XY,
    belt_direction,
    belt_length,
    hand_targets,
)
from scene import write_scene

ROOT = Path(__file__).resolve().parent

IK_JOINTS = [
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx05_body_joint5",
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]

READY_L = np.array([0.58, 0.20, 0.96])
READY_R = np.array([0.58, -0.20, 0.96])
LOOK_TIME = 0.70
PUSH_TIME = 1.85
PUSH_SPEED = 0.28
REGRASP_TIME = 0.35
RETURN_TIME = 0.42
CTRL_EVERY = 4
SLIDE_MU = 0.09
SLIDE_K = 1800.0
SLIDE_FMAX = 95.0
GRAVITY = 9.81


def smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def yaw_quat(yaw):
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def encode_png(rgb, level=1):
    """RGB uint8 dizisini PNG baytına çevirir."""
    import struct
    import zlib

    image = np.ascontiguousarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, level))
    png += chunk(b"IEND", b"")
    return png


def write_png(path, rgb):
    Path(path).write_bytes(encode_png(rgb, level=9))


class SortShow:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.scratch = mujoco.MjData(model)
        self._bind_robot()
        self._style_robot()
        self._bind_bags()
        self._bind_chevrons()
        self.phase = "boot"
        self.phase_t = 0.0
        self.push_index = 0
        self.active = None
        self.cursor = 0
        self.delivered = {gate["key"]: 0 for gate in GATES}
        self._last_report = -1
        self.q_ref_ready = False
        self.hand_error = 1.0
        for bag in self.bags:
            self._assign(bag)
        self._hold_passive_actuators()
        mujoco.mj_forward(model, data)

    def _bind_robot(self):
        model = self.model
        self.ik_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in IK_JOINTS]
        self.ik_qadr = np.array([int(model.jnt_qposadr[jid]) for jid in self.ik_jids], dtype=int)
        self.ik_dadr = np.array([int(model.jnt_dofadr[jid]) for jid in self.ik_jids], dtype=int)
        self.ik_lo = np.array([model.jnt_range[jid, 0] for jid in self.ik_jids])
        self.ik_hi = np.array([model.jnt_range[jid, 1] for jid in self.ik_jids])
        self.q_ref = self.data.qpos[self.ik_qadr].copy()
        # Dirsek aşağı ve dışarıda: omuz hafif öne, üst kol az bükük, dirsek bükülü.
        self.q_ref[5] = -0.70
        self.q_ref[6] = -0.40
        self.q_ref[7] = -1.00
        self.q_ref[12] = -0.70
        self.q_ref[13] = 0.40
        self.q_ref[14] = -1.00
        self.body_ref = self.q_ref[:3].copy()
        self.left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_l_center_link")
        self.right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_r_center_link")
        self.left_elbow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_l_link4")
        self.right_elbow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_r_link4")
        self.left_shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_l_link2")
        self.right_shoulder = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_r_link2")
        self.left_wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_l_link7")
        self.right_wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_r_link7")
        # T-pozdan başlayan çözüm bileği ters çeviriyor. Rahat dirsek duruşundan başla.
        self.data.qpos[self.ik_qadr] = self.q_ref
        self.head_yaw_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "idx11_head_joint1")
        self.head_pitch_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "idx13_head_joint3")
        self.head_yaw_qadr = int(model.jnt_qposadr[self.head_yaw_jid])
        self.head_pitch_qadr = int(model.jnt_qposadr[self.head_pitch_jid])
        self.head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head_link3")
        self.head_joints = {self.head_yaw_jid, self.head_pitch_jid}
        self.root_qpos = self.data.qpos[:7].copy()
        self.pos_acts = []
        for act_id in range(model.nu):
            if int(model.actuator_trntype[act_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            if abs(float(model.actuator_biasprm[act_id, 1])) < 1e-8:
                continue
            jid = int(model.actuator_trnid[act_id, 0])
            self.pos_acts.append((act_id, int(model.jnt_qposadr[jid]), jid))
        self.ik_act = {}
        ik_set = set(self.ik_jids)
        for act_id, qadr, jid in self.pos_acts:
            if jid in ik_set:
                self.ik_act[jid] = act_id
        self._calibrate_head()

    def _face_axis(self, data):
        return data.xmat[self.head_id].reshape(3, 3)[:, self.head_axis]

    def _face_yaw(self, data):
        axis = self._face_axis(data)
        return float(np.atan2(axis[1], axis[0]))

    def _face_depression(self, data):
        axis = self._face_axis(data)
        return float(np.atan2(-axis[2], np.hypot(axis[0], axis[1])))

    def _calibrate_head(self):
        scratch = self.scratch
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = 0
        scratch.qpos[self.head_yaw_qadr] = 0.0
        scratch.qpos[self.head_pitch_qadr] = 0.0
        mujoco.mj_forward(self.model, scratch)
        xmat = scratch.xmat[self.head_id].reshape(3, 3)
        # Yüz, robotun önü olan +X ile en çok hizalanan eksendir. Omuz ekseni +Y'dir.
        self.head_axis = int(np.argmax([abs(xmat[0, column]) for column in range(3)]))
        self.head_yaw_at_zero = self._face_yaw(scratch)
        self.head_pitch_at_zero = self._face_depression(scratch)
        scratch.qpos[self.head_yaw_qadr] = 0.4
        mujoco.mj_forward(self.model, scratch)
        self.head_yaw_gain = (self._face_yaw(scratch) - self.head_yaw_at_zero) / 0.4
        scratch.qpos[self.head_yaw_qadr] = 0.0
        scratch.qpos[self.head_pitch_qadr] = 0.25
        mujoco.mj_forward(self.model, scratch)
        self.head_pitch_gain = (self._face_depression(scratch) - self.head_pitch_at_zero) / 0.25
        if abs(self.head_yaw_gain) < 1e-3:
            self.head_yaw_gain = 1.0
        if abs(self.head_pitch_gain) < 1e-3:
            self.head_pitch_gain = 1.0
        self.head_yaw_lim = tuple(float(v) for v in self.model.jnt_range[self.head_yaw_jid])
        self.head_pitch_lim = tuple(float(v) for v in self.model.jnt_range[self.head_pitch_jid])

    def _style_robot(self):
        model = self.model
        for geom_id in range(model.ngeom):
            rgba = model.geom_rgba[geom_id]
            neon = (
                (rgba[0] > 0.9 and rgba[1] < 0.15 and rgba[2] > 0.9)
                or (rgba[0] > 0.9 and rgba[1] < 0.15 and rgba[2] < 0.15)
                or (rgba[0] > 0.9 and rgba[1] > 0.9 and rgba[2] < 0.15)
            )
            if neon:
                model.geom_rgba[geom_id] = (0.78, 0.79, 0.81, 1.0)
            elif rgba[0] > 0.7 and 0.3 < rgba[1] < 0.55 and rgba[2] < 0.15:
                model.geom_rgba[geom_id] = (0.18, 0.19, 0.21, 1.0)
            body_id = int(model.geom_bodyid[geom_id])
            if body_id == 0:
                continue
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if body_name.startswith("bag"):
                continue
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0

    def _bind_bags(self):
        self.bags = []
        spawns = (2.15, 3.57, 4.99, 6.41)
        for index in range(4):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"bag{index}_free")
            shell = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"bag{index}_shell")
            lid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"bag{index}_lid")
            tag = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"bag{index}_tag")
            qadr = int(self.model.jnt_qposadr[jid])
            self.bags.append(
                {
                    "qadr": qadr,
                    "dadr": int(self.model.jnt_dofadr[jid]),
                    "shell": shell,
                    "lid": lid,
                    "tag": tag,
                    "pos": np.array([spawns[index], 0.0, BAG_Z]),
                    "yaw": 0.0,
                    "state": "in",
                    "gate": None,
                    "flight": "",
                    "travel": 0.0,
                    "mass": BAG_MASSES[index],
                    "vel": np.zeros(2),
                }
            )
            self._write_bag(self.bags[-1])

    def _bind_chevrons(self):
        self.chevrons = []
        model = self.model
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if not name.startswith("chev_"):
                continue
            if name.startswith("chev_in"):
                direction = np.array([-1.0, 0.0])
            else:
                key = name.split("_")[1]
                gate = next(item for item in GATES if item["key"] == key)
                direction = belt_direction(gate)
            self.chevrons.append((geom_id, model.geom_pos[geom_id].copy(), direction))

    def _assign(self, bag):
        gate = GATES[self.cursor % len(GATES)]
        bag["gate"] = gate
        bag["flight"] = FLIGHTS[self.cursor % len(FLIGHTS)]
        bag["state"] = "in"
        bag["yaw"] = 0.0
        bag["travel"] = 0.0
        self.cursor += 1
        color = gate["color"]
        self.model.geom_rgba[bag["shell"]] = color
        self.model.geom_rgba[bag["lid"]] = color * np.array([0.42, 0.42, 0.42, 1.0])
        self.model.geom_rgba[bag["lid"]][3] = 1.0
        self.model.geom_rgba[bag["tag"]] = (0.96, 0.96, 0.94, 1.0)

    def _hold_passive_actuators(self):
        data = self.data
        data.ctrl[:] = 0
        for act_id, qadr, _jid in self.pos_acts:
            data.ctrl[act_id] = data.qpos[qadr]

    def _spawn_x(self):
        incoming = [bag["pos"][0] for bag in self.bags if bag["state"] == "in"]
        tail = max(incoming) if incoming else STOP_XY[0]
        return max(tail + QUEUE_GAP, INCOMING_X1 - 1.1)

    def _respawn(self, bag):
        self._assign(bag)
        bag["pos"][:] = (self._spawn_x(), 0.0, BAG_Z)
        bag["yaw"] = 0.0
        bag["vel"][:] = 0.0

    def _write_bag(self, bag):
        qadr = bag["qadr"]
        self.data.qpos[qadr : qadr + 3] = bag["pos"]
        self.data.qpos[qadr + 3 : qadr + 7] = yaw_quat(bag["yaw"])
        dadr = bag["dadr"]
        self.data.qvel[dadr : dadr + 6] = 0.0

    def _lock_base(self):
        self.data.qpos[:7] = self.root_qpos
        self.data.qvel[:6] = 0.0

    def _lead_incoming(self):
        waiting = [bag for bag in self.bags if bag["state"] == "in"]
        if not waiting:
            return None
        return min(waiting, key=lambda bag: bag["pos"][0])

    def _gaze_point(self):
        if self.active is not None:
            return self.active["pos"].copy()
        lead = self._lead_incoming()
        if lead is not None:
            return lead["pos"].copy()
        return np.array([STOP_XY[0], STOP_XY[1], BAG_Z])

    def _segment(self):
        points = [STOP_XY, *self.active["gate"]["pushes"]]
        index = self.push_index
        return points[index], points[index + 1]

    def _hand_targets(self):
        forward = (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        if self.phase in ("boot", "idle", "return") or self.active is None:
            return READY_L, READY_R, forward
        start, end = self._segment()
        direction = end - start
        if self.phase == "look":
            bag_xy = STOP_XY
        elif self.phase == "regrasp":
            bag_xy = start
        else:
            bag_xy = self.active["pos"][:2]
        left, right, palms = hand_targets(bag_xy, direction)
        # Önce yukarı kalk, bavulun üstünden yanaş, dışarı gelince in.
        if self.phase == "look" and self.phase_t < 0.34:
            high = 1.18
            return (
                np.array([self.data.xpos[self.left_id][0], self.data.xpos[self.left_id][1], high]),
                np.array([self.data.xpos[self.right_id][0], self.data.xpos[self.right_id][1], high]),
                palms,
            )
        if self.phase == "look":
            high = 1.16
            placed = []
            for hand, actual in (
                (left, self.data.xpos[self.left_id]),
                (right, self.data.xpos[self.right_id]),
            ):
                if float(np.hypot(actual[0] - hand[0], actual[1] - hand[1])) > 0.08:
                    hand = np.array([hand[0], hand[1], high])
                placed.append(hand)
            left, right = placed
        return left, right, palms

    def _ik(self, target_l, target_r, palms):
        model = self.model
        scratch = self.scratch
        scratch.qpos[:] = self.data.qpos
        scratch.qvel[:] = 0.0
        q = scratch.qpos[self.ik_qadr].copy()
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        bodies = (self.left_id, self.right_id)
        targets = (target_l, target_r)
        step_limit = 0.040
        orient = 0.35
        pos_error = 1.0
        elbows = (
            (self.left_elbow, self.left_shoulder, 1.0),
            (self.right_elbow, self.right_shoulder, -1.0),
        )
        for _ in range(10):
            for address, value in zip(self.ik_qadr, q):
                scratch.qpos[address] = value
            mujoco.mj_forward(model, scratch)
            rows = []
            errors = []
            for body_id, target, palm in zip(bodies, targets, palms):
                mujoco.mj_jacBody(model, scratch, jacp, jacr, body_id)
                delta = target - scratch.xpos[body_id]
                # Uzak hedefe sert gitmek kolu savuruyor. Yüze yaklaşınca sıkılaşır.
                gain = 4.0 if float(np.linalg.norm(delta)) < 0.08 else 1.0
                rows.append(gain * jacp[:, self.ik_dadr])
                errors.append(gain * delta)
                rotation = scratch.xmat[body_id].reshape(3, 3)
                # Avuç içi (yerel +Z) bavula bakar. Parmaklar yerel +X ile yukarı durur.
                axis = rotation[:, 2]
                palm = palm / np.linalg.norm(palm)
                tangent = palm - axis * float(np.dot(axis, palm))
                axis_jac = np.cross(jacr[:, self.ik_dadr].T, axis).T
                rows.append(orient * axis_jac)
                errors.append(orient * tangent)
                # Ağız dik kalsın; bilek kendi etrafında dönüp eli çevirmesin.
                up_axis = rotation[:, 0]
                upright = np.array([0.0, 0.0, 1.0])
                up_err = upright - up_axis * float(np.dot(up_axis, upright))
                up_jac = np.cross(jacr[:, self.ik_dadr].T, up_axis).T
                rows.append(0.16 * up_jac)
                errors.append(0.16 * up_err)
            for elbow_id, shoulder_id, sign in elbows:
                mujoco.mj_jacBody(model, scratch, jacp, jacr, elbow_id)
                elbow_jac = jacp[:, self.ik_dadr].copy()
                elbow_jac[:, :4] = 0.0
                want = scratch.xpos[shoulder_id].copy()
                want[0] += 0.12
                want[1] += sign * 0.20
                want[2] -= 0.18
                gap = want - scratch.xpos[elbow_id]
                gap[0] *= 0.4
                rows.append(0.18 * elbow_jac)
                errors.append(0.18 * gap)
            if self.active is not None and self.phase in ("look", "push"):
                bag = self.active["pos"]
                for body, weight, pad in (
                    (self.left_wrist, 0.8, 0.04),
                    (self.right_wrist, 0.8, 0.04),
                    (self.left_id, 0.9, 0.0),
                    (self.right_id, 0.9, 0.0),
                ):
                    mujoco.mj_jacBody(model, scratch, jacp, jacr, body)
                    point = scratch.xpos[body]
                    dx = float(point[0] - bag[0])
                    dy = float(point[1] - bag[1])
                    limx = float(BAG_HALF[0] + pad)
                    limy = float(BAG_HALF[1] + pad)
                    if abs(dx) >= limx or abs(dy) >= limy or abs(float(point[2]) - BAG_Z) > 0.18:
                        continue
                    penx = limx - abs(dx)
                    peny = limy - abs(dy)
                    depth = min(penx, peny)
                    gain = weight
                    if body in (self.left_id, self.right_id) and depth > 0.04:
                        gain = 1.8
                    if penx < peny:
                        outward = np.array([(1.0 if dx >= 0.0 else -1.0), 0.0, 0.0]) * penx
                    else:
                        outward = np.array([0.0, (1.0 if dy >= 0.0 else -1.0), 0.0]) * peny
                    limb_jac = jacp[:, self.ik_dadr].copy()
                    limb_jac[:, :4] = 0.0
                    rows.append(gain * limb_jac)
                    errors.append(gain * outward)
                # Avuç derisi bavulun içinde kalmasın.
                for body in (self.left_id, self.right_id):
                    mujoco.mj_jacBody(model, scratch, jacp, jacr, body)
                    rot = scratch.xmat[body].reshape(3, 3)
                    offset = rot[:, 2] * PALM_SKIN
                    tip = scratch.xpos[body] + offset
                    dx = float(tip[0] - bag[0])
                    dy = float(tip[1] - bag[1])
                    if abs(dx) >= BAG_HALF[0] or abs(dy) >= BAG_HALF[1] or abs(float(tip[2]) - BAG_Z) > 0.16:
                        continue
                    penx = float(BAG_HALF[0] - abs(dx))
                    peny = float(BAG_HALF[1] - abs(dy))
                    if penx < peny:
                        outward = np.array([(1.0 if dx >= 0.0 else -1.0), 0.0, 0.0]) * penx
                    else:
                        outward = np.array([0.0, (1.0 if dy >= 0.0 else -1.0), 0.0]) * peny
                    tip_jac = jacp[:, self.ik_dadr] + np.cross(jacr[:, self.ik_dadr].T, offset).T
                    tip_jac = tip_jac.copy()
                    tip_jac[:, :4] = 0.0
                    rows.append(4.0 * tip_jac)
                    errors.append(4.0 * outward)
            # Bilekler birbirinin içinden geçmesin.
            mujoco.mj_jacBody(model, scratch, jacp, jacr, self.left_wrist)
            left_jac = jacp[:, self.ik_dadr].copy()
            left_jac[:, :4] = 0.0
            left_wrist = scratch.xpos[self.left_wrist].copy()
            mujoco.mj_jacBody(model, scratch, jacp, jacr, self.right_wrist)
            right_jac = jacp[:, self.ik_dadr].copy()
            right_jac[:, :4] = 0.0
            right_wrist = scratch.xpos[self.right_wrist].copy()
            apart = left_wrist - right_wrist
            span = float(np.linalg.norm(apart))
            if span < 0.20:
                direc = apart / max(span, 1e-4)
                gap = 0.20 - span
                rows.append(0.9 * left_jac)
                errors.append(0.9 * direc * gap)
                rows.append(0.9 * right_jac)
                errors.append(0.9 * -direc * gap)
            body_lock = np.zeros((3, len(q)))
            body_lock[0, 0] = body_lock[1, 1] = body_lock[2, 2] = 1.0
            rows.append(1.4 * body_lock)
            errors.append(1.4 * (self.body_ref - q[:3]))
            # Dirsek ters dönüp kolu bavulun içinden geçirmesin.
            for index in (6, 13):
                angle = float(q[index])
                if abs(angle) <= 1.35:
                    continue
                row = np.zeros((1, len(q)))
                row[0, index] = 1.0
                rows.append(2.4 * row)
                errors.append(np.array([2.4 * (np.sign(angle) * 0.8 - angle)]))
            # Dirsek eklemi sınıra yapışıp kolu katlamasın.
            for index in (7, 14):
                angle = float(q[index])
                if angle > -1.45:
                    continue
                row = np.zeros((1, len(q)))
                row[0, index] = 1.0
                rows.append(2.2 * row)
                errors.append(np.array([2.2 * (-1.05 - angle)]))
            jacobian = np.vstack(rows)
            error = np.concatenate(errors)
            pos_error = float(np.linalg.norm(error[:6]))
            width = jacobian.shape[0]
            damping = 0.06
            gram = jacobian @ jacobian.T + damping * np.eye(width)
            solved = np.linalg.solve(gram, error)
            dq = jacobian.T @ solved
            nullspace = np.eye(len(q)) - jacobian.T @ np.linalg.solve(gram, jacobian)
            dq = dq + nullspace @ (0.06 * (self.q_ref - q))
            q = np.clip(q + np.clip(dq, -step_limit, step_limit), self.ik_lo, self.ik_hi)
        self.hand_error = pos_error
        return q

    def _apply_ctrl(self, q, head_yaw, head_pitch):
        data = self.data
        for jid, act_id in self.ik_act.items():
            index = self.ik_jids.index(jid)
            data.ctrl[act_id] = q[index]
        for act_id, qadr, jid in self.pos_acts:
            if jid == self.head_yaw_jid:
                data.ctrl[act_id] = head_yaw
            elif jid == self.head_pitch_jid:
                data.ctrl[act_id] = head_pitch
            elif jid not in self.ik_act:
                data.ctrl[act_id] = data.qpos[qadr]

    def _head_command(self):
        head = self.data.xpos[self.head_id]
        target = self._gaze_point()
        desired_yaw = float(np.atan2(target[1] - head[1], target[0] - head[0]))
        flat = float(np.hypot(target[0] - head[0], target[1] - head[1]))
        desired_pitch = float(np.atan2(head[2] - target[2], max(flat, 0.05)))
        current_yaw = self._face_yaw(self.data)
        current_pitch = self._face_depression(self.data)
        yaw_error = (desired_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi
        yaw = self.data.qpos[self.head_yaw_qadr] + yaw_error / self.head_yaw_gain
        pitch = self.data.qpos[self.head_pitch_qadr] + (desired_pitch - current_pitch) / self.head_pitch_gain
        return (
            float(np.clip(yaw, *self.head_yaw_lim)),
            float(np.clip(pitch, *self.head_pitch_lim)),
        )

    def control(self):
        target_l, target_r, palms = self._hand_targets()
        q = self._ik(target_l, target_r, palms)
        if self.phase == "boot" and self.hand_error < 0.08:
            self.q_ref_ready = True
        yaw, pitch = self._head_command()
        self._apply_ctrl(q, yaw, pitch)

    def _start_sort(self, bag):
        bag["state"] = "sort"
        bag["pos"][0] = STOP_XY[0]
        bag["pos"][1] = 0.0
        bag["yaw"] = 0.0
        bag["vel"][:] = 0.0
        self.push_index = 0
        self.active = bag
        self.phase = "look"
        self.phase_t = 0.0
        print(f"[{self.data.time:5.1f}s]  {bag['flight']}  {bag['mass']:.0f} kg  ->  {bag['gate']['name']}")

    def update(self, dt):
        self._advance_phase(dt)
        self._move_bags(dt)
        self._animate_belts()
        for bag in self.bags:
            self._write_bag(bag)

    def _advance_phase(self, dt):
        self.phase_t += dt
        if self.phase == "boot":
            if self.hand_error < 0.07 or self.phase_t > 3.5:
                self.phase = "idle"
                self.phase_t = 0.0
            return
        if self.phase == "idle":
            lead = self._lead_incoming()
            if lead is not None and lead["pos"][0] <= STOP_XY[0] + 0.02:
                self._start_sort(lead)
            return
        if self.phase == "look" and self.phase_t >= LOOK_TIME and (self._pusher_gap() < 0.04 or self.phase_t > 1.15):
            self.phase = "push"
            self.phase_t = 0.0
            return
        if self.phase == "push" and self._push_finished():
            if self.push_index + 1 < len(self.active["gate"]["pushes"]):
                self.push_index += 1
                self.phase = "regrasp"
                self.phase_t = 0.0
            else:
                self._release_active()
                self.phase = "return"
                self.phase_t = 0.0
            return
        if self.phase == "regrasp" and self.phase_t >= REGRASP_TIME:
            self.phase = "push"
            self.phase_t = 0.0
            return
        if self.phase == "return" and self.phase_t >= RETURN_TIME:
            self.phase = "idle"
            self.phase_t = 0.0
            self.active = None

    def _release_active(self):
        bag = self.active
        if bag is None:
            return
        gate = bag["gate"]
        direction = belt_direction(gate)
        bag["yaw"] = 0.0
        bag["state"] = "out"
        bag["travel"] = float(np.dot(bag["pos"][:2] - gate["p0"], direction))

    def _bag_on_belt(self, bag):
        gate = bag["gate"]
        direction = belt_direction(gate)
        side = np.array([-float(direction[1]), float(direction[0])])
        along = abs(float(direction[0])) * BAG_HALF[0] + abs(float(direction[1])) * BAG_HALF[1]
        across = BAG_HALF[0] * abs(side[0]) + BAG_HALF[1] * abs(side[1])
        front = bag["pos"][:2] + direction * along
        nose = float(np.dot(front - gate["p0"], direction))
        off = abs(float(np.dot(bag["pos"][:2] - gate["p0"], side)))
        # Ön kenar ağzın içinde ve gövde şeridin arasında.
        return nose > -0.10 and off + across < float(gate["half_w"]) + 0.02

    def _hold_between_rails(self, bag, gate):
        """Korkuluk duvardır. Bavul yanından geçemez, ortaya da ışınlanmaz."""
        direction = belt_direction(gate)
        side = np.array([-float(direction[1]), float(direction[0])])
        across = float(BAG_HALF[0] * abs(side[0]) + BAG_HALF[1] * abs(side[1]))
        limit = float(gate["half_w"]) - across - 0.02
        limit = max(limit, 0.01)
        off = float(np.dot(bag["pos"][:2] - gate["p0"], side))
        clamped = float(np.clip(off, -limit, limit))
        if abs(clamped - off) <= 1e-6:
            return
        bag["pos"][:2] += side * (clamped - off)
        outward = float(np.dot(bag["vel"], side))
        if outward * off > 0.0:
            bag["vel"] -= side * outward

    def _push_finished(self):
        if self._bag_on_belt(self.active):
            return True
        return self.phase_t > 6.0

    def _push_axis(self):
        start, end = self._segment()
        delta = end - start
        length = float(np.linalg.norm(delta))
        hat = delta / max(length, 1e-6)
        sign = 1.0 if hat[1] >= 0.0 else -1.0
        return start, hat, length, sign

    def _pusher_gap(self):
        """İten elin, bavulun dış yüzündeki hedeften sapması."""
        if self.active is None:
            return 1.0
        _start, _hat, _length, sign = self._push_axis()
        left, right, _palms = self._hand_targets()
        if sign > 0.0:
            return float(np.linalg.norm(self.data.xpos[self.right_id] - right))
        return float(np.linalg.norm(self.data.xpos[self.left_id] - left))

    def _slide(self, bag, dt, force):
        """Yatay kayma. Z ve yaw kilitli; bavul kalkmaz, dönmez."""
        mass = float(bag["mass"])
        vel = bag["vel"]
        hold = SLIDE_MU * mass * GRAVITY
        speed = float(np.linalg.norm(vel))
        if speed < 0.02:
            if float(np.linalg.norm(force)) <= hold:
                vel[:] = 0.0
                bag["pos"][2] = BAG_Z
                bag["yaw"] = 0.0
                return
        else:
            force = force - vel / speed * hold
        vel += force / mass * dt
        speed = float(np.linalg.norm(vel))
        # El bavulu fırlatmaz. Kayma, avucun yetişebileceği hızda kalır.
        if speed > 0.40:
            vel *= 0.40 / speed
        bag["pos"][:2] += vel * dt
        bag["pos"][2] = BAG_Z
        bag["yaw"] = 0.0

    def _carry(self, bag, dt, direction, speed):
        """Bant bavulu hızına çeker. Ağır olan daha geç o hıza çıkar."""
        mass = float(bag["mass"])
        vel = bag["vel"]
        along = float(np.dot(vel, direction))
        accel = float(np.clip(speed - along, -55.0 / mass, 55.0 / mass))
        side = np.array([-direction[1], direction[0]])
        lat = float(np.dot(vel, side))
        vel += direction * accel * dt - side * lat * 6.0 * dt
        bag["pos"][:2] += vel * dt
        bag["pos"][2] = BAG_Z
        bag["yaw"] = 0.0

    def _push_force(self, bag):
        _left, _right, palms = self._hand_targets()
        force = np.zeros(2)
        hands = (
            (self.data.xpos[self.left_id], palms[0]),
            (self.data.xpos[self.right_id], palms[1]),
        )
        for hand, palm in hands:
            if abs(float(hand[2]) - BAG_Z) > 0.22:
                continue
            for axis in (0, 1):
                if abs(float(palm[axis])) < 0.5:
                    continue
                sign = 1.0 if palm[axis] > 0.0 else -1.0
                other = 1 - axis
                if abs(float(hand[other]) - bag["pos"][other]) > BAG_HALF[other] + 0.12:
                    continue
                face = bag["pos"][axis] - sign * BAG_HALF[axis]
                gap = sign * (face - float(hand[axis]))
                # El yüzün içinden bakıyorsa bu itiş sayılmaz.
                if gap < -0.02:
                    continue
                clearance = gap - GRIP_FACE
                if clearance > 0.06:
                    continue
                force[axis] += sign * min(SLIDE_K * (0.06 - clearance), SLIDE_FMAX)
        # El, bavulu bandın tersine itemez.
        direction = belt_direction(bag["gate"])
        for axis in (0, 1):
            if force[axis] * float(direction[axis]) < 0.0:
                force[axis] = 0.0
        # İtiş, duraktan bant ağzına giden çizgide kalsın. Çizgiden uzaklaşan pay silinir.
        goal = np.asarray(bag["gate"]["pushes"][0], dtype=float)
        start = np.asarray(STOP_XY, dtype=float)
        delta = goal - start
        length = float(np.linalg.norm(delta))
        if length > 1e-4:
            hat = delta / length
            rel = bag["pos"][:2] - start
            along = float(np.dot(rel, hat))
            off = rel - hat * along
            span = float(np.linalg.norm(off))
            if along < length and span > 0.035:
                away = off / span
                extra = float(np.dot(force, away))
                if extra > 0.0:
                    force = force - away * extra
        return force

    def _move_bags(self, dt):
        bag = self.active
        if bag is not None and self.phase in ("look", "push", "regrasp"):
            if self.phase == "push":
                self._slide(bag, dt, self._push_force(bag))
            else:
                bag["pos"][:2] = STOP_XY
                bag["vel"][:] = 0.0
            bag["pos"][2] = BAG_Z
            bag["yaw"] = 0.0

        incoming = sorted((bag for bag in self.bags if bag["state"] == "in"), key=lambda bag: bag["pos"][0])
        lane_busy = any(
            other["state"] != "in" and abs(float(other["pos"][1])) < 2 * BAG_HALF[1] + 0.22
            for other in self.bags
        )
        for index, bag in enumerate(incoming):
            limit = STOP_XY[0] if index == 0 else incoming[index - 1]["pos"][0] + QUEUE_GAP
            # Sıradaki bavul, aktarılanın içine girmez. Şerit boşalınca öne gelir.
            if index == 0 and lane_busy:
                limit = max(limit, float(STOP_XY[0]) + 2 * BAG_HALF[0] + 0.55)
            if bag["pos"][0] <= limit + 0.005:
                bag["pos"][0] = min(bag["pos"][0], limit)
                bag["pos"][1] = 0.0
                bag["vel"][:] = 0.0
            else:
                self._carry(bag, dt, np.array([-1.0, 0.0]), IN_SPEED)
                if bag["pos"][0] < limit:
                    bag["pos"][0] = limit
                    bag["vel"][:] = 0.0
            bag["pos"][1] = 0.0
            bag["pos"][2] = BAG_Z
            bag["yaw"] = 0.0

        for bag in self.bags:
            if bag["state"] != "out":
                continue
            gate = bag["gate"]
            direction = belt_direction(gate)
            extent = abs(float(direction[0])) * BAG_HALF[0] + abs(float(direction[1])) * BAG_HALF[1]
            side = np.array([-float(direction[1]), float(direction[0])])
            front = bag["pos"][:2] + direction * extent
            nose = float(np.dot(front - gate["p0"], direction))
            off = abs(float(np.dot(bag["pos"][:2] - gate["p0"], side)))
            across = BAG_HALF[0] * abs(side[0]) + BAG_HALF[1] * abs(side[1])
            # Bant, bavul gerçekten şeride girince devralır. Konum zıplatılmaz.
            if nose > -0.10 and off + across < float(gate["half_w"]) + 0.02:
                travel = float(np.dot(bag["pos"][:2] - gate["p0"], direction))
                signed = float(np.dot(bag["pos"][:2] - gate["p0"], side))
                bag["vel"] -= side * float(np.clip(signed * 1.2, -0.35, 0.35)) * dt
                self._carry(bag, dt, direction, OUT_SPEED)
                self._hold_between_rails(bag, gate)
            else:
                self._slide(bag, dt, np.zeros(2))
            bag["travel"] = float(np.dot(bag["pos"][:2] - gate["p0"], direction))
            if bag["travel"] > belt_length(gate) - 0.55:
                self.delivered[gate["key"]] += 1
                self._respawn(bag)

    def _animate_belts(self):
        shift = (0.46 * self.data.time) % 0.52
        for geom_id, base, direction in self.chevrons:
            self.model.geom_pos[geom_id][0] = base[0] + direction[0] * shift
            self.model.geom_pos[geom_id][1] = base[1] + direction[1] * shift

    def step_n(self, count):
        for index in range(count):
            if index % CTRL_EVERY == 0:
                self.control()
            mujoco.mj_step(self.model, self.data)
            self._lock_base()
            self.update(self.model.opt.timestep)

    def joint_report(self):
        """Sürülen eklemlerin anlık açısı, derece. Konum servo komutunun sonucu."""
        groups = (
            ("Bel", ("idx05_body_joint5",)),
            ("Kafa", ("idx11_head_joint1", "idx13_head_joint3")),
            (
                "Sol kol",
                (
                    "idx21_arm_l_joint1",
                    "idx22_arm_l_joint2",
                    "idx23_arm_l_joint3",
                    "idx24_arm_l_joint4",
                    "idx25_arm_l_joint5",
                    "idx26_arm_l_joint6",
                    "idx27_arm_l_joint7",
                ),
            ),
            (
                "Sağ kol",
                (
                    "idx61_arm_r_joint1",
                    "idx62_arm_r_joint2",
                    "idx63_arm_r_joint3",
                    "idx64_arm_r_joint4",
                    "idx65_arm_r_joint5",
                    "idx66_arm_r_joint6",
                    "idx67_arm_r_joint7",
                ),
            ),
        )
        lines = []
        for title, names in groups:
            angles = []
            for name in names:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                qadr = int(self.model.jnt_qposadr[jid])
                angles.append(f"{np.degrees(self.data.qpos[qadr]):+5.0f}°")
            lines.append(f"{title:<8} {'  '.join(angles)}")
        return "\n".join(lines)

    def report(self):
        counts = "  ".join(f"{gate['key']}:{self.delivered[gate['key']]}" for gate in GATES)
        print(f"[{self.data.time:5.1f}s]  faz {self.phase:6}  el hatası {self.hand_error:.3f} m   teslim {counts}")


def load_model():
    scene_path, _chevrons = write_scene()
    model = mujoco.MjModel.from_xml_path(str(scene_path.resolve()))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_pose")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    return model, data


def run_headless(show, seconds, shot_dir=None):
    shots = {0.6, 3.4, 5.2, 8.0, 11.0, 15.0, 20.0}
    taken = set()
    renderer = None
    camera = None
    if shot_dir is not None:
        shot_dir.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(show.model, 720, 1280)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.35, 0.0, 0.7)
        camera.distance = 11.2
        camera.azimuth = 118
        camera.elevation = -32
    next_report = 1.0
    while show.data.time < seconds:
        show.step_n(CTRL_EVERY)
        if show.data.time >= next_report:
            show.report()
            next_report += 2.0
        if renderer is not None:
            for mark in shots:
                if mark not in taken and show.data.time >= mark:
                    renderer.update_scene(show.data, camera=camera)
                    write_png(shot_dir / f"t{mark:.1f}.png", renderer.render())
                    taken.add(mark)
    if renderer is not None:
        renderer.close()
    return all(show.delivered[gate["key"]] >= 1 for gate in GATES)


WEB_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>G2 bavul ayırma</title>
<style>
  html, body { margin: 0; height: 100%; background: #121416; color: #e8e8e8;
    font: 15px/1.4 -apple-system, sans-serif; }
  body { display: flex; flex-direction: column; align-items: center; }
  header { width: min(1100px, 100%); padding: 12px 16px 0; box-sizing: border-box; }
  h1 { font-size: 18px; margin: 0 0 4px; font-weight: 600; }
  p { margin: 0; color: #b7b7b7; }
  #view { width: min(1100px, 100%); height: auto; cursor: grab; touch-action: none; background: #000; }
  #status { margin-top: 8px; font-variant-numeric: tabular-nums; }
  #joints { width: min(1100px, 100%); margin: 8px 0 16px; padding: 0 16px; box-sizing: border-box;
    color: #c8c8c8; font: 13px/1.45 ui-monospace, monospace; white-space: pre; }
</style>
<header>
  <h1>G2 havaalanı bavul ayırma</h1>
  <p>Sürükle: kamerayı çevir. Tekerlek: yakınlaş. A kırmızı sol, B turuncu düz sol, C mavi düz sağ, D yeşil sağ.</p>
  <p id="status">bağlanıyor…</p>
</header>
<img id="view" alt="simülasyon">
<pre id="joints">eklemler…</pre>
<script>
const img = document.getElementById("view");
const status = document.getElementById("status");
const joints = document.getElementById("joints");
let drag = null;
img.addEventListener("pointerdown", (e) => {
  drag = { x: e.clientX, y: e.clientY, id: e.pointerId };
  img.setPointerCapture(e.pointerId);
  img.style.cursor = "grabbing";
});
img.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  fetch(`/cam?dx=${dx}&dy=${dy}`);
});
function endDrag() { drag = null; img.style.cursor = "grab"; }
img.addEventListener("pointerup", endDrag);
img.addEventListener("pointercancel", endDrag);
img.addEventListener("wheel", (e) => {
  e.preventDefault();
  fetch(`/cam?zoom=${e.deltaY}`);
}, { passive: false });
async function tick() {
  const res = await fetch("/frame?t=" + Date.now());
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const previous = img.src;
  img.src = url;
  img.onload = () => { if (previous.startsWith("blob:")) URL.revokeObjectURL(previous); tick(); };
}
async function poll() {
  const res = await fetch("/status");
  status.textContent = await res.text();
  const motors = await fetch("/joints");
  joints.textContent = await motors.text();
  setTimeout(poll, 400);
}
tick();
poll();
</script>
"""


def run_web(show, port=8765):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse

    renderer = mujoco.Renderer(show.model, 720, 1100)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.35, 0.0, 0.7)
    camera.distance = 11.2
    camera.azimuth = 118.0
    camera.elevation = -32.0

    def nudge(dx=0.0, dy=0.0, zoom=0.0):
        camera.azimuth -= dx * 0.35
        camera.elevation = float(np.clip(camera.elevation - dy * 0.25, -89, 89))
        camera.distance = float(np.clip(camera.distance * (1.0015 ** zoom), 2.2, 28))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send(self, code, body, content_type):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(200, WEB_PAGE, "text/html; charset=utf-8")
                return
            if parsed.path == "/frame":
                show.step_n(75)
                renderer.update_scene(show.data, camera=camera)
                self._send(200, encode_png(renderer.render()), "image/png")
                return
            if parsed.path == "/status":
                bag = show.active
                if bag is None:
                    line = f"{show.data.time:5.1f}s   bekliyor"
                else:
                    line = f"{show.data.time:5.1f}s   {bag['flight']}  {bag['mass']:.0f} kg  →  {bag['gate']['name']}"
                counts = "   ".join(f"{gate['key']}:{show.delivered[gate['key']]}" for gate in GATES)
                self._send(200, f"{line}    teslim {counts}", "text/plain; charset=utf-8")
                return
            if parsed.path == "/joints":
                self._send(200, show.joint_report(), "text/plain; charset=utf-8")
                return
            if parsed.path == "/cam":
                query = parse_qs(parsed.query)
                nudge(
                    float(query.get("dx", ["0"])[0]),
                    float(query.get("dy", ["0"])[0]),
                    float(query.get("zoom", ["0"])[0]),
                )
                self._send(204, b"", "text/plain")
                return
            self._send(404, "yok", "text/plain")

    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Canlı görüntü: {url}")
    print("Tarayıcıda açılır. Sekmeyi kapatmak simülasyonu durdurmaz; bu terminali durdurun.")
    import threading
    import webbrowser

    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        renderer.close()


def run_viewer(show):
    print("Pencere MuJoCo görüntüleyicisidir. Kamerayı sürükleyerek gezinin, kapatınca biter.")
    print("Çıkışlar: A kırmızı SOL, B turuncu DÜZ SOL, C mavi DÜZ SAĞ, D yeşil SAĞ.")
    with mujoco.viewer.launch_passive(show.model, show.data) as viewer:
        viewer.cam.lookat[:] = (0.35, 0.0, 0.7)
        viewer.cam.distance = 11.2
        viewer.cam.azimuth = 118
        viewer.cam.elevation = -32
        while viewer.is_running():
            started = time.time()
            show.step_n(CTRL_EVERY)
            viewer.sync()
            second = int(show.data.time)
            if second != show._last_report and second % 2 == 0:
                show.report()
                show._last_report = second
            remaining = CTRL_EVERY * show.model.opt.timestep - (time.time() - started)
            if remaining > 0:
                time.sleep(remaining)


def _ensure_mjpython():
    """macOS'ta canlı pencere yalnızca mjpython ile açılır."""
    if sys.platform != "darwin":
        return
    if Path(sys.executable).name == "mjpython":
        return
    launcher = Path(sys.executable).parent / "mjpython"
    if not launcher.exists():
        raise SystemExit(
            "Canlı pencere için mjpython gerekli. MuJoCo kurulunca .venv/bin/mjpython gelir."
        )
    os.execv(str(launcher), [str(launcher), *sys.argv])


def main():
    parser = argparse.ArgumentParser(description="G2 havaalanı bavul ayırma simülasyonu")
    parser.add_argument("--seconds", type=float, default=0.0, help="Bu süre boyunca görüntülemeden çalıştır")
    parser.add_argument("--shots", type=Path, default=None, help="Belirtilen klasöre kare PNG kaydet")
    parser.add_argument("--native", action="store_true", help="Ayrı MuJoCo penceresi aç (Mac'te mjpython gerekir)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.native and args.seconds <= 0:
        _ensure_mjpython()

    from download_g2 import ensure_model

    ensure_model()
    model, data = load_model()
    show = SortShow(model, data)
    show.q_ref_ready = False
    print(f"G2 yüklendi. Eklem {model.nq}, servo {len(show.pos_acts)}.")
    print("Robot ayırma masasının başında sabit. Bavullar giriş bandından gelir.")
    if args.seconds > 0:
        ok = run_headless(show, args.seconds, args.shots)
        show.report()
        if not ok:
            raise SystemExit("Her çıkış bandına en az bir bavul ulaşmadı.")
        return
    if args.native:
        run_viewer(show)
        return
    run_web(show, args.port)


if __name__ == "__main__":
    main()
