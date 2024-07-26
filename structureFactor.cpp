#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

void GRPP(double* gr, double* rij, double* rLists, int* rListLengths, double rho, double delta, int NA, int NB) {
    double prefactor = 0.0317468179671204848928816524673180100775269496411364620405990871 / (delta * rho);
    int N = NA + NB;
    int c1 = NA * rListLengths[0];
    int c2 = c1 + NA * rListLengths[1];
    int c3 = c2 + NB * rListLengths[1];
    int c4 = c3 + NB * rListLengths[2];
    int r1 = rListLengths[0];
    int r2 = r1 + rListLengths[1];
    int r3 = r2 + rListLengths[2];
    #pragma omp parallel for
    for (int idx = 0; idx < c4; idx++) {
        int i, N, rIndex;
        int cMin = c3;
        int dc = rListLengths[2];
        int Nmin = NA;
        int Nmax = NA + NB;
        int rMin = r2;
        int iMin = NA;
        if (idx < c3) {
            cMin = c2;
            dc = rListLengths[1];
            Nmin = 0;
            Nmax = NA;
            rMin = r1;
        }
        if (idx < c2) {
            cMin = c1;
            dc = rListLengths[1];
            Nmin = NA;
            Nmax = NA + NB;
            rMin = r1;
            iMin = 0;
        }
        if (idx < c1) {
            cMin = 0;
            dc = rListLengths[0];
            Nmin = 0;
            Nmax = NA;
            rMin = 0;
            iMin = 0;
        }
        rIndex = (idx - cMin) % dc;
        N = NA + NB;
        double r, rp, factor;
    //    r = rList[rIndex];
        i = (idx - cMin) / dc + iMin;
        for (int j = Nmin; j < Nmax; j++) {
            r = rLists[rMin + rIndex];
            if (i < j) {
                rp = rij[((N * (N - 1)) / 2) - ((N - i) * ((N - i) - 1)) / 2 + j - i - 1];
            }
            else {
                rp = rij[((N * (N - 1)) / 2) - ((N - j) * ((N - j) - 1)) / 2 + i - j - 1];
            }
            factor = exp(-(r - rp) * (r - rp) / (2 * delta * delta)) * prefactor / (r * r);
//            gr[i * 2 * numR + rIndex * 2 + onOff] += factor;
            gr[idx] += factor;
        }
    }
}

void getRij(double* rij, double* positions, double L, double gamma, int N) {
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

void getGR(double* gr, double* rLists, double* positions, int* rListLengths, double L, double gamma, double rho, double delta, int NA, int NB) {
    int N = NA + NB;
    int numElements = N * (N - 1) / 2;
    double* rij = new double[numElements];
    getRij(rij, positions, L, gamma, NA + NB);
    GRPP(gr, rij, rLists, rListLengths, rho, delta, NA, NB);
}

