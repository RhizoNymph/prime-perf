#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    int n, steps;
    float dt;

    if (fread(&n, sizeof(int), 1, stdin) != 1) return 1;
    if (fread(&steps, sizeof(int), 1, stdin) != 1) return 1;
    if (fread(&dt, sizeof(float), 1, stdin) != 1) return 1;

    float *x  = malloc(n * sizeof(float));
    float *y  = malloc(n * sizeof(float));
    float *z  = malloc(n * sizeof(float));
    float *vx = malloc(n * sizeof(float));
    float *vy = malloc(n * sizeof(float));
    float *vz = malloc(n * sizeof(float));
    float *mass = malloc(n * sizeof(float));
    float *ax = malloc(n * sizeof(float));
    float *ay = malloc(n * sizeof(float));
    float *az = malloc(n * sizeof(float));

    if (!x || !y || !z || !vx || !vy || !vz || !mass || !ax || !ay || !az) return 1;

    /* Read body data: 7 floats per body (x, y, z, vx, vy, vz, mass) */
    for (int i = 0; i < n; i++) {
        float body[7];
        if (fread(body, sizeof(float), 7, stdin) != 7) return 1;
        x[i]    = body[0];
        y[i]    = body[1];
        z[i]    = body[2];
        vx[i]   = body[3];
        vy[i]   = body[4];
        vz[i]   = body[5];
        mass[i] = body[6];
    }

    const float eps_sq = 1e-6f;

    for (int step = 0; step < steps; step++) {
        /* Zero accelerations */
        memset(ax, 0, n * sizeof(float));
        memset(ay, 0, n * sizeof(float));
        memset(az, 0, n * sizeof(float));

        /* Compute pairwise gravitational accelerations */
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                float dx = x[j] - x[i];
                float dy = y[j] - y[i];
                float dz_val = z[j] - z[i];
                float dist_sq = dx * dx + dy * dy + dz_val * dz_val + eps_sq;
                float inv_dist = 1.0f / sqrtf(dist_sq);
                float inv_dist3 = inv_dist * inv_dist * inv_dist;
                ax[i] += mass[j] * dx * inv_dist3;
                ay[i] += mass[j] * dy * inv_dist3;
                az[i] += mass[j] * dz_val * inv_dist3;
            }
        }

        /* Update velocities */
        for (int i = 0; i < n; i++) {
            vx[i] += ax[i] * dt;
            vy[i] += ay[i] * dt;
            vz[i] += az[i] * dt;
        }

        /* Update positions */
        for (int i = 0; i < n; i++) {
            x[i] += vx[i] * dt;
            y[i] += vy[i] * dt;
            z[i] += vz[i] * dt;
        }
    }

    /* Write output: 7 floats per body */
    for (int i = 0; i < n; i++) {
        float body[7];
        body[0] = x[i];
        body[1] = y[i];
        body[2] = z[i];
        body[3] = vx[i];
        body[4] = vy[i];
        body[5] = vz[i];
        body[6] = mass[i];
        fwrite(body, sizeof(float), 7, stdout);
    }

    free(x);  free(y);  free(z);
    free(vx); free(vy); free(vz);
    free(mass);
    free(ax); free(ay); free(az);

    return 0;
}
