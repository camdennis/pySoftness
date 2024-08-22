import libstructureFactor as sf
import numpy as np
from matplotlib import pyplot as plt
import pickle

def getGR(rList, positions, box, rho, delta, NA, NB):
    numR = len(rList)
    newPositions = np.dot(box, positions.copy().T).T
    L = box[0, 0]
    gamma = box[0, 1] / L
    # What form does gr take?
    gr = np.zeros(2 * numR * (NA + NB))
    # particle 1 takes indices 0 to 2 * numR where 0, 1 are gAA and gAB
    # particles of type be have indices gBB, gBA
    flatPositions = newPositions.reshape(np.prod(positions.shape))
    sf.getGR(gr, rList, flatPositions, numR, L, gamma, rho, delta, NA, NB)
    # Reshape this
    indicesList = np.array([0, numR * NA, (numR + numR) * NA, (numR + numR) * NA + numR * NB, (numR + numR) * NA + (numR + numR) * NB])
    grAA = gr[indicesList[0]: indicesList[1]].reshape(NA, numR)
    grAB = gr[indicesList[1]: indicesList[2]].reshape(NA, numR)
    grBA = gr[indicesList[2]: indicesList[3]].reshape(NB, numR)
    grBB = gr[indicesList[3]: indicesList[4]].reshape(NB, numR)
    return [grAA, grAB, grBA, grBB]

def getRij(positions, box, N):
    newPositions = np.dot(box, positions.copy().T).T
    L = box[0, 0]
    gamma = box[0, 1] / L
    # What form does gr take?
    rij = np.zeros(N * (N - 1) // 2)
    # particle 1 takes indices 0 to 2 * numR where 0, 1 are gAA and gAB
    # particles of type be have indices gBB, gBA
    flatPositions = newPositions.reshape(np.prod(positions.shape))
    sf.getRij(rij, flatPositions, L, gamma, N)
    rijMatrix = np.zeros((N, N))
    upperIndices = np.triu_indices(N, k = 1)
    rijMatrix[upperIndices] = rij
    rijMatrix += rijMatrix.T
    # Reshape this
    return rijMatrix

def getISFOverlap(positions1, positions2, L, gamma, rmax, cutoff, N):
    ISFOverlap = np.zeros(2)
    sf.getISFOverlap(ISFOverlap, positions1 * L, positions2 * L, L, gamma, rmax, cutoff, N)
    return ISFOverlap[0], ISFOverlap[1]

def LJPot(dr, sigma, epsilon):
    maxFactor = 0.4 / sigma
    epsFactor = 0.016316891136
    idr = (sigma / dr)
    idr[idr <= maxFactor] = 0.0
    return epsilon * epsFactor + 4 * epsilon * (idr**12 - idr**6)

#def LJPot(dr, sigma, epsilon):
#    idr = (sigma / dr)
#    return 4 * epsilon * (idr**12 - idr**6)

def WCAPot(dr, sigma, epsilon):
    maxFactor = 0.890898718140339304740226205590512523 /sigma
    idr = (sigma / dr)
#    idr = idr[idr > maxFactor]
    idr[idr <= maxFactor] = 0.0
    return epsilon + epsilon * 4 * (idr**12 - idr**6)

def HertzianPot(dr, sigma, epsilon):
    dr /= sigma
#    dr = dr[dr < 1]
    dr[dr >= 1] = 0
    return (1 - dr)**(2.5) * epsilon / 2.5

def potential(dr, sigma, epsilon, potentialType):
    if potentialType == "Hertzian":
        return HertzianPot(dr, sigma, epsilon)
    elif potentialType == "WCA":
        return WCAPot(dr, sigma, epsilon)
    elif potentialType == "LJ":
        return LJPot(dr, sigma, epsilon)
    else:
        raise Exception("potential not implemented")

def getPhiISFOverlap(rList, positions1, positions2, gamma, L, rho, delta, NA, NB, potentialType, temp, cutoff = 0.3):
    N = NA + NB
    if potentialType == "Hertzian":
        sigma11 = 1
        sigma12 = 0.8571428571428572
#        sigma22 = 0.7142857142857143
        eps11 = 1
        eps12 = 1
#        eps22 = 1
    else:
        sigma11 = 1
        sigma12 = 0.8
#        sigma22 = 0.88
        eps11 = 1
        eps12 = 1.5
#        eps22 = 0.5
    U11 = potential(rList, sigma11, eps11, potentialType)
    U12 = potential(rList, sigma12, eps12, potentialType)
#    U22 = potential(rList, sigma22, eps22, potentialType)
    box = np.zeros((3, 3))
    box[0, 0] = L
    box[1, 1] = L
    box[2, 2] = L
    box[0, 1] = L * gamma
    grAA, grAB, grBA, grBB = getGR(rList, positions2, box, rho, delta, NA, NB)
    x1 = NA / N
    x2 = NB / N
    grAA /= x1
    grAB /= x2
    grBA /= x1
    grBB /= x2
    grAA = np.sum(grAA, axis = 0) / NA
    grAB = np.sum(grAB, axis = 0) / NA
    grBA = np.sum(grBA, axis = 0) / NB
    grBB = np.sum(grBB, axis = 0) / NB
    args = np.argwhere(rList < 0.25).T[0]
    grAA[args] = 0
    grAB[args] = 0
    grBA[args] = 0
    grBB[args] = 0
    dr = rList[1] - rList[0]
    rcut11 = 2.5 * sigma11
    rcut12 = 2.5 * sigma12
    dv11 = U11
    dv12 = U12
    beta = 1 / temp
    args = np.argwhere(rList < 3).T[0]
    grAA = grAA[args]
    grAB = grAB[args]
    # Let's test it out
    MList = np.zeros(len(rList), dtype = int)
    MList[::2] = 4
    MList[1::2] = 2
    MList[0] = 1
    MList[-1] = 1
    args = np.argwhere((grAA > 1e-6) & (rList < rcut11)).T[0]
    d2u11 = np.zeros(len(rList))
    d2u11[args] = -beta * dv11[args]
    c11 = np.zeros(len(rList))
    args = np.argwhere(grAA > 1e-6).T[0]
    c11[args] = d2u11[args] + (grAA[args] - 1) - np.log(grAA[args])
    grAA[grAA <= 1e-6] = 0
    u2T11 = x1**2 * c11 * grAA
    args = np.argwhere((grAB > 1e-6) & (rList < rcut12)).T[0]
    d2u12 = np.zeros(len(rList))
    d2u12[args] = -beta * dv12[args]
    c12 = np.zeros(len(rList))
    args = np.argwhere(grAB > 1e-6).T[0]
    c12[args] = d2u12[args] + (grAB[args] - 1) - np.log(grAB[args])
    grAB[grAB <= 1e-6] = 0
    u2T12 = x1*x2 * c12 * grAB
    u2T = u2T11 + u2T12
    Phi = np.sum(u2T * rList**2 * MList)
    Phi *= 4 * np.pi * (dr / 3) * rho
    # The first gr peak probably comes from grBB. Let's go with that.
    rmax = rList[np.argmax(grBB)]
    ISF, overlap = getISFOverlap(positions1, positions2, L, gamma, rmax, cutoff, N)
    return Phi, ISF, overlap

if __name__ == "__main__":
    positions1 = np.load('pos0.npy')
    positions2 = np.load('pos2.npy')
    L = (4000 / 1.2)**(1/3)
    box = np.identity(3) * L
    rList = np.linspace(0.005, 2.995, 300)
    rho = 1.2
    NA = 3200
    NB = 800
    potentialType = 'WCA'
    temp = 5.5
    gamma = 0
    delta = 0.025
    rij = getRij(positions2, box, NA + NB)
    Phi, ISF, overlap = getPhiISFOverlap(rList, positions1, positions2, gamma, L, rho, delta, NA, NB, potentialType, temp, cutoff = 0.3)
    print(Phi, ISF, overlap)
    Phi, ISF, overlap = getPhiISFOverlap(rList, positions2, positions1, gamma, L, rho, delta, NA, NB, potentialType, temp, cutoff = 0.3)
    print(Phi, ISF, overlap)
