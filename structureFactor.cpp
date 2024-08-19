#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

void GRPPBRK(double* gr, double* rij, double* rList, int numR, double rho, double delta, int NA, int NB) {
    double prefactor = 0.0317468179671204848928816524673180100775269496411364620405990871 / (delta * rho);
    int i, N, rIndex;
    double r, rp, factor;
    int onOff;
    N = NA + NB;
    #pragma omp parallel for
    for (int idx = 0; idx < (NA + NB) * numR; idx++) {
        rIndex = idx % numR;
        r = rList[rIndex];
        i = idx / numR;
        #pragma omp parallel for
        for (int j = 0; j < N; j++) {
//            if (i == j) {
//                continue;
//            }
            if ((i < NA) == (j < NA)) {
                onOff = 0;
            }
            else {
                onOff = 1;
            }
            if (i < j) {
                rp = rij[((N * (N - 1)) / 2) - ((N - i) * ((N - i) - 1)) / 2 + j - i - 1];
            }
            else {
                rp = rij[((N * (N - 1)) / 2) - ((N - j) * ((N - j) - 1)) / 2 + i - j - 1];
            }
            factor = exp(-(r - rp) * (r - rp) / (2 * delta * delta)) * prefactor / (r * r);
            gr[i * 2 * numR + rIndex * 2 + onOff] += factor;
        }
    }
}

void GRPPINT(double* gr, double* rij, double* rLists, int numR, double rho, double delta, int NA, int NB) {
    double prefactor = 0.0317468179671204848928816524673180100775269496411364620405990871 / (delta * rho);
    int i, N, rIndex;
    double r, rp, factor;
    bool iA = false;
    bool jA = false;
    int onOff;
    N = NA + NB;
    #pragma omp parallel for
    for (int idx = 0; idx < (NA + NB) * numR; idx++) {
        rIndex = idx % numR;
    //    r = rList[rIndex];
        i = idx / numR;
        iA = false;
        jA = false;
        if (i < NA) {
            iA = true;
        }
        #pragma omp parallel for
        for (int j = 0; j < N; j++) {
            onOff = 0;
            if (i == j) {
                continue;
            }
            jA = false;
            if (j < NA) {
                jA = true;
            }
            if (iA) {
                if (jA) {
                    r = rLists[rIndex];
                }
                else {
                    r = rLists[numR + rIndex];
                    onOff = 1;
                }
            }
            else {
                if (jA) {
                    r = rLists[numR + rIndex];
                    onOff = 1;
                }
                else {
                    r = rLists[2 * numR + rIndex];
                }
            }
            if (i < j) {
                rp = rij[((N * (N - 1)) / 2) - ((N - i) * ((N - i) - 1)) / 2 + j - i - 1];
            }
            else {
                rp = rij[((N * (N - 1)) / 2) - ((N - j) * ((N - j) - 1)) / 2 + i - j - 1];
            }
            factor = exp(-(r - rp) * (r - rp) / (2 * delta * delta)) * prefactor / (r * r);
            gr[i * 2 * numR + rIndex * 2 + onOff] += factor;
        }
    }
}

void GRPP(double* gr, double* rij, double* rList, int numR, double rho, double delta, int NA, int NB) {
    double prefactor = 0.0317468179671204848928816524673180100775269496411364620405990871 / (delta * rho);
    int N = NA + NB;
    int c1 = NA * numR;
    int c2 = c1 + NA * numR;
    int c3 = c2 + NB * numR;
    int c4 = c3 + NB * numR;
    int i, rIndex;
    #pragma omp parallel for
    for (int idx = 0; idx < c4; idx++) {
        int cMin = c3;
        int Nmin = NA;
        int Nmax = NA + NB;
        int iMin = NA;
        if (idx < c3) {
            cMin = c2;
            Nmin = 0;
            Nmax = NA;
        }
        if (idx < c2) {
            cMin = c1;
            Nmin = NA;
            Nmax = NA + NB;
            iMin = 0;
        }
        if (idx < c1) {
            cMin = 0;
            Nmin = 0;
            Nmax = NA;
            iMin = 0;
        }
        rIndex = (idx - cMin) % numR;
        N = NA + NB;
        double r, rp, factor;
        i = (idx - cMin) / numR + iMin;
        r = rList[rIndex];
        gr[idx] = 0.0;
        #pragma omp parallel for
        for (int j = Nmin; j < Nmax; j++) {
            if (i < j) {
                rp = rij[((N * (N - 1)) / 2) - ((N - i) * ((N - i) - 1)) / 2 + j - i - 1];
            }
            else {
                rp = rij[((N * (N - 1)) / 2) - ((N - j) * ((N - j) - 1)) / 2 + i - j - 1];
            }
//            if (i == j) {
//                continue;
//            }
            factor = exp(-(r - rp) * (r - rp) / (2 * delta * delta)) * prefactor / (r * r);
            gr[idx] += factor;
        }
    }
}

void getRijGOLD(double* rij, double* positions, double L, double gamma, int N) {
    int numElements = N * (N - 1) / 2;
    #pragma omp parallel for
    for (int threadID = 0; threadID < numElements; threadID++) {
        int i = N - 2 - floor(sqrt(-8.0 * float(threadID) + 4.0 * N * (N - 1) - 7.0) / 2.0 - 0.5);
        int j = threadID + (i * (i + 3)) / 2 - N * i + 1;
        double dx, dy, dz, cy = 0.0;
        // You need box info here to get the rij right. LEShift
        dx = positions[i * 3 + 0] - positions[j * 3 + 0];
        dy = positions[i * 3 + 1] - positions[j * 3 + 1];
        dz = positions[i * 3 + 2] - positions[j * 3 + 2];
        cy = round(dy / L) * L;
        dx = dx - cy * gamma;
        dz = dz - round(dz / L) * L;
        dy = dy - cy;
        rij[threadID] = sqrt(dx * dx + dy * dy + dz * dz);
    }
}

void getRij(double* rij, const double* positions, double L, double gamma, int N) {
    int numElements = (N * (N - 1)) / 2;

    #pragma omp parallel for
    for (int threadID = 0; threadID < numElements; threadID++) {
        int i = static_cast<int>(N - 2 - std::floor(std::sqrt(-8.0 * threadID + 4.0 * N * (N - 1) - 7.0) / 2.0 - 0.5));
        int j = threadID + (i * (i + 3)) / 2 - N * i + 1;

        double dx = positions[i * 3 + 0] - positions[j * 3 + 0];
        double dy = positions[i * 3 + 1] - positions[j * 3 + 1];
        double dz = positions[i * 3 + 2] - positions[j * 3 + 2];

        double dyOverL = dy / L;
        double dzOverL = dz / L;

        double cy = std::round(dyOverL) * L;
        dx -= cy * gamma;
        dy -= cy;
        dz -= std::round(dzOverL) * L;

        rij[threadID] = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
}

void getGR(double* gr, double* rList, double* positions, int numR, double L, double gamma, double rho, double delta, int NA, int NB) {
    int N = NA + NB;
    int numElements = N * (N - 1) / 2;
    double* rij = new double[numElements];
    getRij(rij, positions, L, gamma, NA + NB);
    GRPP(gr, rij, rList, numR, rho, delta, NA, NB);
}

void getISFOverlap(double* ISFOverlap, double* positions1, double* positions2, double L, double gamma, double rmax, double cutoff, int N) {
    double* dirVec = new double[26 * 3];
    double dx, dy, dz, cy, theta, phi, qdot, dist;
    int kk;
    double pi4 = 0.78539816339744830962;
    kk = -1;
    ISFOverlap[0] = 0.0;
    ISFOverlap[1] = 0.0;
    for (int i = 0; i < 8; i++) {
        theta = i * pi4;
        for (int j = 0; j < 3; j++) {
            phi = pi4 * (j + 1);
            kk++;
            dirVec[kk * 3 + 0] = cos(theta) * sin(phi);
            dirVec[kk * 3 + 1] = sin(theta) * sin(phi);
            dirVec[kk * 3 + 2] = cos(phi);
        }
    }
    dirVec[24 * 3 + 2] = 1;
    dirVec[25 * 3 + 2] = -1;
    for (int i = 0; i < N; i++) {
        dx = positions2[i * 3 + 0] - positions1[i * 3 + 0];
        dy = positions2[i * 3 + 1] - positions1[i * 3 + 1];
        dz = positions2[i * 3 + 2] - positions1[i * 3 + 2];
        cy = round(dy / L) * L;
        dx = dx - cy * gamma;
        dz = dz - round(dz / L) * L;
        dy = dy - cy;
        if (sqrt(dx * dx + dy * dy + dz * dz) < cutoff) {
            ISFOverlap[1] += 1;
        }
        for (int j = 0; j < 26; j++) {
            ISFOverlap[0] += cos(dirVec[3 * j + 0] * pi4 * 4 / rmax * dx + dirVec[3 * j + 1] * pi4 * 4 / rmax * dy + dirVec[3 * j + 2] * pi4 * 4 / rmax * dz);
        }

    }
    ISFOverlap[0] /= 26;
    ISFOverlap[1] /= N;
}
