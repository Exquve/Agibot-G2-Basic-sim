"""Bavul sürükleme eğitimi için Gymnasium ortamı.

Robot sabit durur; iki elin avuç hedefleri politika ağının çıktısıdır.
Hedeflerden kollara giden yol, gösterideki DLS ters kinematik ile aynıdır.
Bavulun hareketi de gösterideki sürtünmeli kayma modelinden gelir; böylece
eğitim sırasında bulunan politika, sort_demo.py'nin kurallarına aykırı düşmez.

Gösterimden farkı: her epizotta bavul ve çıkış rastgele seçilir, bavulun
boyutu/ağırlığı/sürtünmesi ve başlangıç açısı rastgele üretilir (domain
randomization). Politika tek bir sabit yola ezberlenemez; gelen bavulun
konumunu ve boyutunu gözlemden okuyup ona göre davranmak zorundadır.
"""

import re
import sys
from pathlib import Path

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent))

from layout import BAG_Z, GATES, STOP_XY, belt_direction  # noqa: E402
from omnihand import install_omnihand  # noqa: E402
from scene import write_scene  # noqa: E402
from sort_demo import IK_JOINTS, SortShow, smoothstep  # noqa: E402

DT = 0.02           # politika adımı (s) — 5 Hz karar
CTRL_EVERY = 4      # simülasyon tik sayısı (model timestep ~5 ms)
ACTION_SCALE = 0.10  # hedef delta limiti (m)
PUSH_SPEED = 0.28
PALM_SKIN = 0.045
MAX_STEPS = 300     # 6 saniyelik deneme


def _flatten_includes(xml_path):
    """<include> ağacını tek dosyada birleştirir.

    MuJoCo include yollarını filedir/relative olarak çözüyor; bu modeldeki
    '../../../..omnihand/left.xml' gibi yazımlar patlıyor. Biz yolu Python'da
    doğru çözüp içeriği olduğu yere gömüyoruz. Derinlik sırasıyla body/arm/hand
    iç içe geçer; default/class tanımları en dıştaki dosyada zaten var.
    """
    xml_path = Path(xml_path).resolve()
    text = xml_path.read_text()

    def repl(m):
        rel = m.group(1)
        # bozuk '../..' yazılımını temizle: son iki bileşeni koru
        parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".")]
        if ".." in rel:
            tail = rel.split("..")[-1].strip("/\\")
            sub = tail.split("/")
            if len(sub) == 1:
                name = "G2/" + sub[0]          # link.xml -> G2/<hangisi>/link.xml aşağıda aranır
                names = [sub[0]]
            else:
                names = ["/".join(sub)]
        else:
            names = [rel]
        # Adayları bul: dosya ağacında ada göre ara (parça sayısı ile ayırt et)
        root = Path(__file__).resolve().parent / "vendor" / "mujoco"
        candidates = []
        for nm in names:
            base = nm.split("/")[-1]
            depth = nm.count("/") + 1
            for cand in root.rglob(base):
                relparts = cand.relative_to(root).parts
                if len(relparts) >= depth and str(cand.relative_to(root)).endswith(nm):
                    candidates.append(cand)
        if not candidates:
            raise FileNotFoundError(f"include bulunamadı: {rel} ({xml_path})")
        # Aynı addan birkaç tane varsa (link.xml gibi), çağranın kardeş dizinine bak
        if len(candidates) > 1:
            here = xml_path.parent.name
            preferred = [c for c in candidates if here in str(c.parent)]
            if preferred:
                candidates = preferred
        target = candidates[0]
        return _flatten_includes(target)

    pattern = re.compile('<include\\s+file="([^"]+)"\\s*/>')
    while True:
        m = pattern.search(text)
        if not m:
            break
        text = text[: m.start()] + repl(m) + text[m.end() :]
    return text


def load_train_model():
    """Eğitim sahnesi: gösterim sahnesi + gerçek serbest eklemli bavul."""
    root = Path(__file__).resolve().parent
    path, _chevrons = write_scene()
    xml = _flatten_includes(path)
    xml = xml.replace('<mujoco model="g2_airport_sort">',
                      '<mujoco model="g2_airport_sort"\n  <compiler meshdir="vendor/mujoco/robot/parts/meshes"/>')
    # G2 dosyasının kendi compiler bloğu meshdir'ı robot/parts/meshes yapıyor;
    # düzleştirilmiş metinde o blok en başta kalır, sorun yok — ama include
    # tagları temizlendi. asset path'leri meshdir üzerinden çözülür.
    if "bag0_free" not in xml:
        bag = f"""    <body name="bag0" pos="{STOP_XY[0]:.3f} 0 {BAG_Z:.3f}">
      <freejoint name="bag0_free"/>
      <geom name="bag0_shell" type="box" size="0.25 0.135 0.12" mass="12"
            contype="1" conaffinity="1" rgba="0.8 0.2 0.2 1" friction="0.9 0.05 0.05"/>
      <geom name="bag0_lid" type="box" size="0.25 0.135 0.01" pos="0 0 0.13"
            contype="0" conaffinity="0" mass="0" rgba="0.3 0.3 0.3 1"/>
      <geom name="bag0_tag" type="box" size="0.05 0.001 0.03" pos="0 -0.136 0.02"
            contype="0" conaffinity="0" mass="0" rgba="0.95 0.95 0.9 1"/>
    </body>
"""
        insert_at = xml.rfind("</worldbody>")
        xml = xml[:insert_at] + bag + xml[insert_at:]
    tmp = root / "vendor" / "mujoco" / "train_flat.xml"
    tmp.write_text(xml)
    model = mujoco.MjModel.from_xml_path(str(tmp))
    return model


class BagSlideEnv(gym.Env):
    """Tek bavulu rastgele bir çıkış bandına sürekleyen eğitim ortamı."""

    metadata = {"render_modes": []}

    def __init__(self, seed=None):
        self.model = load_train_model()
        self.data = mujoco.MjData(self.model)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "init_pose")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self.show = SortShow(self.model, self.data)
        self.rng = np.random.default_rng(seed)

        hand_dim = 3 * 2  # iki el, x/y/z deltası
        self.action_space = spaces.Box(-1.0, 1.0, (hand_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (self._obs_dim(),), dtype=np.float32
        )
        self.timestep = float(self.model.opt.timestep)
        self.steps = 0
        self.goal = None
        self.bag_half = None
        self.mu = None
        self.mass = None
        self.prev_dist = None
        self.target_l = None
        self.target_r = None
        self.palms = None

    # ---------- yardımcı ----------
    def _obs_dim(self):
        n_joints = len(IK_JOINTS)
        return 7 + n_joints + n_joints + 6 + 2 + 4 + 2 + 1  # aşağıda sayılıyor

    def bag_qadr(self):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "bag0_free")
        return int(self.model.jnt_qposadr[jid])

    def _quat_yaw(self, quat):
        w, x, y, z = quat
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng = self.rng
        data = self.data
        mujoco.mj_resetDataKeyframe(
            self.model, data, max(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "init_pose"), 0)
        )
        # Kollar rahat duruşa
        data.qpos[self.show.ik_qadr] = self.show.q_ref

        # Rastgele görev: çıkış kapısı
        gate = GATES[int(rng.integers(len(GATES)))]
        direction = belt_direction(gate)
        goal_xy = gate["pushes"][0]

        # Rastgele bavul: boyut, kütle, sürtünme, başlangıç konumu ve açısı
        scale = float(rng.uniform(0.75, 1.25))
        half = np.array([0.25, 0.135, 0.12]) * scale
        self.bag_half = half
        self.mass = float(rng.uniform(8.0, 25.0))
        self.mu = float(rng.uniform(0.05, 0.14))
        start = STOP_XY + rng.normal(0.0, 0.05, 2)
        yaw = float(rng.uniform(-0.35, 0.35))

        qadr = self.bag_qadr()
        c, s = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
        data.qpos[qadr : qadr + 3] = (start[0], start[1], half[2] + 0.008)
        data.qpos[qadr + 3 : qadr + 7] = (c, 0.0, 0.0, s)
        data.qvel[-6:] = 0.0
        shell = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "bag0_shell")
        self.model.geom_size[shell] = half
        self.model.body_mass[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bag0")] = self.mass

        self.goal = gate
        self.goal_dir = direction
        self.goal_xy = goal_xy
        self.start_xy = start.copy()
        self.prev_dist = float(np.linalg.norm(goal_xy - start))
        self.steps = 0

        # Eller hazır pozisyonda, hedefler bugünkü gösterim mantığıyla
        left, right, palms = self.show._hand_targets()
        self.target_l, self.target_r, self.palms = left, right, palms
        mujoco.mj_forward(self.model, data)
        return self._observe(), {}

    def _observe(self):
        data = self.data
        qadr = self.bag_qadr()
        pos = data.qpos[qadr : qadr + 3]
        quat = data.qpos[qadr + 3 : qadr + 7]
        # Serbest eklemin qvel bloğu en sondadır (robot gövdesi kilitli).
        vel = data.qvel[-6:]
        rel_goal = self.goal_xy - pos[:2]
        dist = float(np.linalg.norm(rel_goal))
        heading = np.array([rel_goal[0], rel_goal[1]]) / max(dist, 1e-6)
        aligned = float(np.dot(heading, self.goal_dir))
        joints = data.qpos[self.show.ik_qadr].astype(float)
        dqs = data.qvel[self.show.ik_dadr].astype(float)
        hands = np.concatenate(
            [data.xpos[self.show.left_id], data.xpos[self.show.right_id]]
        ).astype(float)
        obs = np.concatenate(
            [
                np.array(
                    [
                        pos[0], pos[1], pos[2],
                        quat[3],          # yaw bileşeni
                        self.bag_half[0], self.bag_half[1], self.mass / 25.0,
                    ]
                ),
                joints,
                dqs,
                hands,
                rel_goal,
                np.array([dist, aligned, self.mu, 1.0]),
                vel[:2],
                np.array([self.steps / MAX_STEPS]),
            ]
        ).astype(np.float32)
        return obs

    # ---------- fizik ----------
    def _slide_bag(self, force2d, dt):
        """sort_demo.SortShow._slide ile aynı model; Z ve yaw serbest bırakılmaz."""
        data = self.data
        qadr = self.bag_qadr()
        pos = data.qpos[qadr : qadr + 3]
        vel = data.qvel[-6:-4]
        hold = self.mu * self.mass * 9.81
        speed = float(np.linalg.norm(vel))
        if speed < 0.02:
            if float(np.linalg.norm(force2d)) <= hold:
                vel[:] = 0.0
                return
        else:
            force2d = force2d - vel / speed * hold
        acc = force2d / self.mass
        vel += acc * dt
        speed = float(np.linalg.norm(vel))
        if speed > 0.45:
            vel *= 0.45 / speed
        pos[:2] += vel * dt
        pos[2] = self.bag_half[2] + 0.008

    def _push_force(self):
        """İki avucun temas noktalarından bavula binen yatay itiş."""
        data = self.data
        qadr = self.bag_qadr()
        pos = data.qpos[qadr : qadr + 3]
        quat = data.qpos[qadr + 3 : qadr + 7]
        yaw = self._quat_yaw(quat)
        rotz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        half = self.bag_half
        force = np.zeros(2)
        K, FMAX = 1800.0, 95.0
        for body_id in (self.show.left_id, self.show.right_id):
            hand = data.xpos[body_id]
            rot = data.xmat[body_id].reshape(3, 3)
            contact = hand + rot[:, 0] * PALM_SKIN
            local = rotz.T @ (contact[:2] - pos[:2])
            vlocal = np.array([local[0] / max(half[0], 1e-6), local[1] / max(half[1], 1e-6)])
            axis = 0 if abs(vlocal[0]) >= abs(vlocal[1]) else 1
            if abs(vlocal[axis]) < 0.85 or abs(vlocal[1 - axis]) > 1.15:
                continue  # temas bu yüzde değil
            sign = 1.0 if vlocal[axis] >= 0.0 else -1.0
            face_local = sign * half[axis]
            gap = sign * (face_local - local[axis])
            if gap < -0.02:
                continue
            clearance = gap - 0.01
            if clearance > 0.06:
                continue
            push = min(K * (0.06 - clearance), FMAX)
            outward_local = np.zeros(2)
            outward_local[axis] = sign
            force += rotz @ (outward_local * push)
        return force

    def step(self, action):
        action = np.asarray(action, dtype=float).clip(-1.0, 1.0)
        delta = action.reshape(2, 3) * ACTION_SCALE
        self.target_l = np.clip(self.target_l + delta[0], -2.0, 2.5)
        self.target_r = np.clip(self.target_r + delta[1], -2.0, 2.5)

        reward = 0.0
        info = {}
        for sub in range(CTRL_EVERY):
            q = self.show._ik(self.target_l, self.target_r, self.palms)
            head_yaw, head_pitch = self.show._head_command()
            self.show._apply_ctrl(q, head_yaw, head_pitch)
            mujoco.mj_step(self.model, self.data)
            self.show._lock_base()
            dt = self.timestep
            force = self._push_force()
            self._slide_bag(force, dt)
        self.steps += 1

        qadr = self.bag_qadr()
        pos = self.data.qpos[qadr : qadr + 3]
        dist = float(np.linalg.norm(self.goal_xy - pos[:2]))
        progress = self.prev_dist - dist
        self.prev_dist = dist

        reward += 10.0 * progress                       # hedefe yaklaşım
        if dist < 0.18:
            reward += 5.0                               # bant ağzına varış bonusu
        # Bant yönünden sapma cezası
        rel = pos[:2] - self.start_xy
        along = float(np.dot(rel, self.goal_dir))
        lateral = rel - self.goal_dir * along
        reward -= 1.0 * float(np.linalg.norm(lateral)) * DT
        # Enerji ve sarsıntı cezası
        reward -= 1e-3 * float(np.sum(action ** 2))
        reward -= 1e-4 * float(np.sum(self.data.qvel[self.show.ik_dadr] ** 2)) * DT

        terminated = False
        truncated = self.steps >= MAX_STEPS
        if dist < 0.15:
            terminated = True
            reward += 20.0
            info["success"] = True
        else:
            info["success"] = False
        info["dist"] = dist

        obs = self._observe()
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return None


gym.register(id="BagSlide-v0", entry_point=BagSlideEnv)


if __name__ == "__main__":
    env = BagSlideEnv(seed=0)
    obs, _ = env.reset(seed=0)
    print("obs boyutu:", obs.shape, "beklenen:", env.observation_space.shape)
    total = 0.0
    for i in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample() * 0.3)
        total += r
        if term or trunc:
            break
    print(f"{i+1} adım, toplam ödül {total:.2f}, kalan mesafe {info['dist']:.2f} m")
