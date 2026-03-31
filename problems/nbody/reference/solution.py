import math
import struct
import sys


def main() -> None:
    data = sys.stdin.buffer.read()
    offset = 0

    n = struct.unpack_from("<i", data, offset)[0]
    offset += 4
    steps = struct.unpack_from("<i", data, offset)[0]
    offset += 4
    dt = struct.unpack_from("<f", data, offset)[0]
    offset += 4

    # Read body data: 7 floats per body (x, y, z, vx, vy, vz, mass)
    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []
    mass = []

    for _ in range(n):
        body = struct.unpack_from("<7f", data, offset)
        offset += 7 * 4
        x.append(body[0])
        y.append(body[1])
        z.append(body[2])
        vx.append(body[3])
        vy.append(body[4])
        vz.append(body[5])
        mass.append(body[6])

    eps_sq = 1e-6

    for _step in range(steps):
        # Zero accelerations
        ax = [0.0] * n
        ay = [0.0] * n
        az = [0.0] * n

        # Compute pairwise gravitational accelerations
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dz = z[j] - z[i]
                dist_sq = dx * dx + dy * dy + dz * dz + eps_sq
                inv_dist = 1.0 / math.sqrt(dist_sq)
                inv_dist3 = inv_dist * inv_dist * inv_dist
                ax[i] += mass[j] * dx * inv_dist3
                ay[i] += mass[j] * dy * inv_dist3
                az[i] += mass[j] * dz * inv_dist3

        # Update velocities
        for i in range(n):
            vx[i] += ax[i] * dt
            vy[i] += ay[i] * dt
            vz[i] += az[i] * dt

        # Update positions
        for i in range(n):
            x[i] += vx[i] * dt
            y[i] += vy[i] * dt
            z[i] += vz[i] * dt

    # Write output: 7 floats per body
    out = bytearray()
    for i in range(n):
        out.extend(struct.pack("<7f", x[i], y[i], z[i], vx[i], vy[i], vz[i], mass[i]))

    sys.stdout.buffer.write(out)


if __name__ == "__main__":
    main()
