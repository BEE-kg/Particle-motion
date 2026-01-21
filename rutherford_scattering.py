import numpy as np
import matplotlib.pyplot as plt

COULOMB_CONST = 8.9875517923e9
ELEM_CHARGE = 1.602176634e-19
ATOMIC_MASS = 1.66053906660e-27
MEV_TO_J = 1e6 * ELEM_CHARGE


def coulomb_acc(pos, m, q, Q, eps=1e-18):
    x, y = pos
    r2 = x*x + y*y + eps*eps
    r = np.sqrt(r2)
    factor = COULOMB_CONST * q * Q / (m * r2 * r)
    return factor * np.array([x, y])


def rhs(state, m, q, Q):
    x, y, vx, vy = state
    ax, ay = coulomb_acc((x, y), m, q, Q)
    return np.array([vx, vy, ax, ay])


def rk4(state, dt, m, q, Q):
    k1 = rhs(state, m, q, Q)
    k2 = rhs(state + 0.5 * dt * k1, m, q, Q)
    k3 = rhs(state + 0.5 * dt * k2, m, q, Q)
    k4 = rhs(state + dt * k3, m, q, Q)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def track_particle(m, q, Q, x0, b, v0, dt, t_end,
                   r_cut_min=1e-15, r_cut_max=1e-13):

    state = np.array([x0, b, v0, 0.0], dtype=float)
    steps = int(t_end / dt)

    traj_x = []
    traj_y = []

    final_velocity = None

    for _ in range(steps):
        x, y, vx, vy = state
        traj_x.append(x)
        traj_y.append(y)
        final_velocity = (vx, vy)

        r = np.hypot(x, y)
        if r < r_cut_min or r > r_cut_max:
            break

        state = rk4(state, dt, m, q, Q)

    vx, vy = final_velocity
    scatter_angle = np.degrees(np.arctan2(vy, vx))

    return np.array(traj_x), np.array(traj_y), scatter_angle


def main():
    charge_alpha = 2 * ELEM_CHARGE
    charge_gold = 79 * ELEM_CHARGE
    mass_alpha = 4 * ATOMIC_MASS

    energy = 4 * MEV_TO_J
    velocity0 = np.sqrt(2 * energy / mass_alpha)

    x_start = -5e-14
    impact_params = np.linspace(0.2e-14, 1.0e-14, 5)

    dt = 1e-23
    t_max = 5e-20

    plt.figure(figsize=(7, 6))

    for b in impact_params:
        x, y, angle = track_particle(
            mass_alpha,
            charge_alpha,
            charge_gold,
            x_start,
            b,
            velocity0,
            dt,
            t_max,
            r_cut_min=2e-15,
            r_cut_max=8e-14
        )
        plt.plot(x, y, label=f"b={b:.1e} м, θ≈{angle:.1f}°")

    plt.scatter(0, 0, s=70)
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x, м")
    plt.ylabel("y, м")
    plt.title("Кулоновское рассеяние α-частицы на ядре Au")
    plt.legend()
    plt.savefig("rutherford_refactored.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
