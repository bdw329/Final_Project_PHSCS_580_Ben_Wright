import numpy as np
import matplotlib.pyplot as plt

# This function generates the capsule geometry, and outputs the x and y points of the body as panels
def build_orion_capsule(a=1.0, total_points=300):
    """
    Build a clockwise Orion capsule profile.
    Top half is constructed directly; bottom half is mirrored from it.
    """

    R = 5.0
    thetaA = np.deg2rad(32.5)
    thetaN = np.deg2rad(23.04)

    RN = 2.4 * R

    # Segment weights for the TOP half only
    weights = np.array([0.35, 0.18, 0.22, 0.15, 0.10])
    weights /= weights.sum()

    x_top = []
    y_top = []

    def append_seg(x, y):
        if len(x_top) == 0:
            x_top.extend(x.tolist())
            y_top.extend(y.tolist())
        else:
            x_top.extend(x[1:].tolist())
            y_top.extend(y[1:].tolist())

    # -----------------------------------------------------
    # 1. FRONT ARC: from LE (theta=pi) down to top fillet start
    # -----------------------------------------------------
    n = 25
    theta = np.linspace(np.pi, np.pi - thetaN, n)
    x = RN * np.cos(theta)
    y = RN * np.sin(theta)
    append_seg(x, y)

    # -----------------------------------------------------
    # 2. FRONT FILLET (TOP)
    # -----------------------------------------------------
    n = 10
    theta_ff = np.linspace(np.pi - thetaN, np.pi/2 - thetaA, n)
    xc0 = -RN*np.cos(thetaN) + 0.1*R*np.cos(thetaN)
    yc0 =  RN*np.sin(thetaN) - 0.1*R*np.sin(thetaN)
    x = xc0 + 0.1*R*np.cos(theta_ff)
    y = yc0 + 0.1*R*np.sin(theta_ff)
    append_seg(x, y)

    # -----------------------------------------------------
    # 3. TOP SIDE PANEL
    # -----------------------------------------------------
    n = 10
    x_start = x_top[-1]
    y_start = y_top[-1]
    x_end = (-(RN - 1.3153*R) - 0.1*R) + 0.1*R*np.cos(np.pi/2 - thetaA)
    y_end = (0.30633 * R) + 0.1*R*np.sin(np.pi/2 - thetaA)
    x = np.linspace(x_start, x_end, n)
    y = np.linspace(y_start, y_end, n)
    append_seg(x, y)

    # -----------------------------------------------------
    # 4. REAR FILLET (TOP)
    # -----------------------------------------------------
    n = 5
    xc_rear = (-(RN - 1.3153 * R) - 0.1 * R)
    yc_rear = 0.30633 * R
    theta_r = np.linspace(np.pi/2 - thetaA, 0.0, n)
    x = xc_rear + 0.1*R*np.cos(theta_r)
    y = yc_rear + 0.1*R*np.sin(theta_r)
    append_seg(x, y)

    # -----------------------------------------------------
    # 5. TRAILING EDGE (TOP → BOTTOM)
    # -----------------------------------------------------
    n = 10
    x_te = np.full(n, -(RN - 1.3153*R))
    y_te = np.linspace(y_top[-1], 0.0, n)
    append_seg(x_te, y_te)

    # -----------------
    # MIRROR BOTTOM HALF
    # -----------------
    x_top_arr = np.array(x_top)
    y_top_arr = np.array(y_top)

    # Exclude the very first point to avoid duplication at LE
    x_bottom = x_top_arr[-2::-1]
    y_bottom = -y_top_arr[-2::-1]

    # Combine
    xbody = np.concatenate([x_top_arr, x_bottom, [x_top_arr[0]]])
    ybody = np.concatenate([y_top_arr, y_bottom, [y_top_arr[0]]])

    return xbody, ybody


# -----------------------
# RUN & PLOT
# -----------------------
xbody, ybody = build_orion_capsule(a=1.0, total_points=300)
