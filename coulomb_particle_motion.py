import numpy as np
import matplotlib.pyplot as plt


def coulomb_force_acceleration(x, y, m, q, Q, eps=1e-6):
    r_sq = x*x + y*y + eps*eps
    r = np.sqrt(r_sq)
    coef = (q * Q) / (m * r_sq * r)
    return coef * x, coef * y


def equations(state, m, q, Q):
    x, y, vx, vy = state
    ax, ay = coulomb_force_acceleration(x, y, m, q, Q)
    return np.array([vx, vy, ax, ay])


def runge_kutta_step(state, dt, m, q, Q):
    k1 = equations(state, m, q, Q)
    k2 = equations(state + 0.5 * dt * k1, m, q, Q)
    k3 = equations(state + 0.5 * dt * k2, m, q, Q)
    k4 = equations(state + dt * k3, m, q, Q)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def compute_trajectory(m, q, Q, x_start, speed, angle_deg,
                        dt, t_max, r_min=0.02, r_max=5.0):

    angle = np.deg2rad(angle_deg)

    state = np.array([
        x_start,
        0.0,
        speed * np.cos(angle),
        speed * np.sin(angle)
    ], dtype=float)

    path_x, path_y = [], []

    for _ in range(int(t_max / dt)):
        x, y, vx, vy = state
        path_x.append(x)
        path_y.append(y)

        r = np.hypot(x, y)
        if r < r_min or r > r_max:
            break

        state = runge_kutta_step(state, dt, m, q, Q)

    return np.array(path_x), np.array(path_y)


def draw_trajectories(title, m, q, Q, x0, v0, angle_list,
                      dt, t_max, filename):

    plt.figure(figsize=(7, 6))

    for angle in angle_list:
        x, y = compute_trajectory(
            m, q, Q,
            x0, v0, angle,
            dt, t_max
        )
        plt.plot(x, y, label=f"α = {angle}°")

    plt.scatter(0, 0, s=70, label="Центральный заряд")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x, м")
    plt.ylabel("y, м")
    plt.title(title)
    plt.legend()

    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()


def main():
    mass = 1e-3
    charge_particle = 1e-2
    charge_center = 5e-2

    x_start = -1.0
    velocity = 0.1
    angles = [5, 15, 30, 45]

    draw_trajectories(
        "Кулоновское отталкивание (одинаковые знаки зарядов)",
        mass,
        +charge_particle,
        +charge_center,
        x_start,
        velocity,
        angles,
        dt=1e-3,
        t_max=30,
        filename="coulomb_repulsion.png"
    )

    draw_trajectories(
        "Кулоновское притяжение (разные знаки зарядов)",
        mass,
        -charge_particle,
        +charge_center,
        x_start,
        velocity,
        angles,
        dt=1e-3,
        t_max=30,
        filename="coulomb_attraction.png"
    )


if __name__ == "__main__":
    main()


