"""Havaalanı bavul ayırma sahnesinin ölçüleri.

Robot orijinde, yüzü +X yönüne bakar. Beyaz alan düzgün bir altıgendir.
Kenarlar: giriş, dört çıkış, robotun durduğu yer. Çıkış bantları altıgenin
dışında başlar; bavul bir banda giderken başka bandın üzerinden geçmez.
"""

import numpy as np

BELT_TOP = 0.64
BAG_HALF = np.array([0.25, 0.135, 0.12])
BAG_Z = BELT_TOP + BAG_HALF[2] + 0.008
# El merkezi yüzün bu kadar dışında durur. Avuç filesi merkezden ~5.6 cm taşar.
PALM = 0.074
GRIP_FACE = 0.056

# Düz kenarlar sağda ve solda. Sağ kenar giriş, sol kenar robotun yeri.
# Çıkışlar çapraz köşelerden ayrılır; bavul başka bandın üstünden geçmez.
HEX_CENTER = np.array([0.88, 0.0])
APOTHEM = 0.62
HEX_R = float(APOTHEM / np.cos(np.deg2rad(30.0)))
SEAM = 0.04
# Bavul robotun itebileceği yerde durur. Masa merkezi daha ileridedir.
STOP_XY = np.array([0.72, 0.0])

INCOMING_X1 = 7.5
# Giriş ve dört çıkış aynı genişlik. Eksenleri sabit bavul korkuluğa sığsın.
BELT_HALF = 0.34
INCOMING_HALF_W = BELT_HALF
QUEUE_GAP = 1.35
IN_SPEED = 0.46
OUT_SPEED = 0.55
# Dolu bavul. Ağır olanı el daha çok bastırınca kayar.
BAG_MASSES = (12.0, 18.0, 23.0, 15.0)


def _normal(degrees):
    rad = np.deg2rad(degrees)
    return np.array([np.cos(rad), np.sin(rad)])


def hex_radius():
    return float(HEX_R)


def hex_vertices():
    """Saat yönünün tersine. Sağ ve sol kenarlar düzdür."""
    angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330], dtype=float))
    return np.stack(
        [
            HEX_CENTER[0] + HEX_R * np.cos(angles),
            HEX_CENTER[1] + HEX_R * np.sin(angles),
        ],
        axis=1,
    )


def _mouth(normal):
    return HEX_CENTER + normal * APOTHEM


def _gate(key, name, color, degrees, half_w, length):
    normal = _normal(degrees)
    mouth = _mouth(normal)
    p0 = mouth + normal * SEAM
    p1 = p0 + normal * length
    # Bavulun önü banda birkaç santim binince merkez hâlâ şeridin ortasındadır.
    extent = abs(float(normal[0])) * BAG_HALF[0] + abs(float(normal[1])) * BAG_HALF[1]
    push = p0 - normal * (extent - 0.04)
    return {
        "key": key,
        "name": name,
        "color": color,
        "normal": normal,
        "p0": p0,
        "p1": p1,
        "half_w": half_w,
        "pushes": [push],
    }


# Sağ düz kenar giriş. Çıkışlar 60, 120, 240 ve 300 derecede.
INCOMING_X0 = float(HEX_CENTER[0] + APOTHEM + SEAM)

GATES = [
    _gate("A", "A  SOL", np.array([0.84, 0.16, 0.14, 1.0]), 60.0, BELT_HALF, 3.6),
    _gate("B", "B  DÜZ SOL", np.array([0.95, 0.55, 0.10, 1.0]), 120.0, BELT_HALF, 3.6),
    _gate("C", "C  DÜZ SAĞ", np.array([0.16, 0.42, 0.86, 1.0]), 240.0, BELT_HALF, 3.6),
    _gate("D", "D  SAĞ", np.array([0.12, 0.62, 0.34, 1.0]), 300.0, BELT_HALF, 3.6),
]

FLIGHTS = [
    "TK2104",
    "LH441",
    "PC2188",
    "TK1986",
    "BA672",
    "XQ912",
    "TK760",
    "LH1302",
]


def belt_direction(gate):
    delta = gate["p1"] - gate["p0"]
    return delta / np.linalg.norm(delta)


def belt_length(gate):
    return float(np.linalg.norm(gate["p1"] - gate["p0"]))


def hand_targets(bag_xy, direction):
    """İki el de bavulun dışında, ayrı yüzlerde durur.

    Sol kol batıya ve kuzeye, sağ kol güneye uzanır. Hedefler itiş boyunca
    aynı yüzde kalır; kol bir anda diğerinin üzerinden geçmez.
    """
    bag_xy = np.asarray(bag_xy, dtype=float)[:2]
    direction = np.asarray(direction, dtype=float)[:2]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([0.0, 1.0])
    else:
        direction = direction / norm
    sign_y = 1.0 if direction[1] >= 0.0 else -1.0
    hand_z = BAG_Z + 0.09
    side_y = bag_xy[1] - sign_y * (BAG_HALF[1] + PALM)
    palm_y = np.array([0.0, sign_y, 0.0])
    face_x = bag_xy[0] - BAG_HALF[0] - PALM
    palm_x = np.array([1.0, 0.0, 0.0])
    # Yan el robotun yetiştiği x'te. Daha ilerisi kolu yüzden koparıyor.
    side = np.array([bag_xy[0] - 0.14, side_y, hand_z])
    if sign_y > 0.0:
        # Sol kol batı yüzünün kuzeyinde kalır; güneye uzanıp sağ kola girmez.
        end = np.array([face_x, bag_xy[1] + 0.06, hand_z])
        return end, side, (palm_x, palm_y)
    end = np.array([face_x, bag_xy[1] - 0.06, hand_z])
    return side, end, (palm_y, palm_x)
