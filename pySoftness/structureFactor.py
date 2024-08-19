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
    maxFactor = 0.4
    epsFactor = 0.016316891136
    idr = (sigma / dr)
    idr[idr <= maxFactor] = 0.0
    return epsilon * epsFactor + 4 * epsilon * (idr**12 - idr**6)

#def LJPot(dr, sigma, epsilon):
#    idr = (sigma / dr)
#    return 4 * epsilon * (idr**12 - idr**6)

def WCAPot(dr, sigma, epsilon):
    maxFactor = 0.890898718140339304740226205590512523
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
    fileName1 = "/home/rdennis/data/timeTemperatureEquivalence/4000/equilibrated_T_1.0.pickle"
    fileName2 = "/home/rdennis/data/timeTemperatureEquivalence/4000/equilibrated_T_1.0.pickle"
#    testPositions = np.loadtxt("/home/rdennis/Documents/Code/fortranSoftnessTesting/Codes_caging_potential/pos_ljT0.47")
#    testPositions = testPositions[:, 2:]
#    fileName2 = "/home/rdennis/data/timeTemperatureEquivalence/4000HIGHT/equilibrated_T_1.0.pickle"
    with open (fileName1, 'rb') as f:
        data = pickle.load(f)
    box = data['box']
    L = box[0, 0]
    gamma = box[0, 1] / L
    positions1 = data['RLV'] 
    with open (fileName2, 'rb') as f:
        data = pickle.load(f)
    positions2 = data['RLV'] 
    rho = 1.2
    NA = 3200
    NB = 800
#    positions2 = testPositions / L
#    rij = getRij(positions2, box, NA + NB)
    delta = 0.025
    potentialType = "LJ"
    temp = 0.47
    rList = np.arange(300) * 0.01 + 0.005
    Phi, ISF, overlap = getPhiISFOverlap(rList, positions1, positions2, gamma, L, rho, delta, NA, NB, potentialType, temp, cutoff = 0.3)
#    data = np.loadtxt("/home/rdennis/Documents/Code/fortranSoftnessTesting/Codes_caging_potential/gr_lj_atT0.47.dat")
#    print(len(data[:, 0]))
#    plt.plot(data[:, 0], data[:, 1], "r")
#    plt.plot(data[:, 0], data[:, 2], "g")
#    plt.plot(data[:, 0], data[:, 3], "k")
#    plt.show()

    print(Phi, ISF, overlap)
    exit()
    gr = np.loadtxt("/home/rdennis/Documents/Code/Codes_caging_potential/gr_lj_atT0.90.dat")
    sigma11 = 1
    sigma12 = 0.8
    sigma22 = 0.88
    eps11 = 1
    eps12 = 1.5
    eps22 = 0.5
    x1 = 3200 / (4000)
    x2 = 800 / (4000)
    rho = 1.2
    rList = np.linspace(0.005, 2.995, 300)
    dr = rList[1] - rList[0]
    v11 = potential(rList, sigma11, eps11, "LJ")
    v12 = potential(rList, sigma12, eps12, "LJ")
    v22 = potential(rList, sigma22, eps22, "LJ")
    rcut11 = 2.5 * sigma11
    rcut12 = 2.5 * sigma12
    rcut22 = 2.5 * sigma22
    dv11 = v11
    dv12 = v12
    dv22 = v22
    beta = 1 / 0.90
    plt.plot(gr[:, 0], gr[:, 1], "r")
    plt.plot(gr[:, 0], gr[:, 2], "g")
    plt.plot(gr[:, 0], gr[:, 3], "b")
    gr11 = gr[:, 1]
    gr12 = gr[:, 2]
    gr22 = gr[:, 3]
    args = np.argwhere(gr[:, 0] < 3).T[0]
    gr11 = gr11[args]
    gr12 = gr12[args]
    gr22 = gr22[args]
    # Let's test it out
    MList = np.zeros(300, dtype = int)
    MList[::2] = 4
    MList[1::2] = 2
    MList[0] = 1
    MList[-1] = 1
    args = np.argwhere((gr11 > 1e-6) & (rList < rcut11)).T[0]
    d2u11 = np.zeros(300)
    d2u11[args] = -beta * dv11[args]
    c11 = np.zeros(300)
    args = np.argwhere(gr11 > 1e-6).T[0]
    c11[args] = d2u11[args] + (gr11[args] - 1) - np.log(gr11[args])
    u2T11 = x1**2 * c11 * gr11
    args = np.argwhere((gr12 > 1e-6) & (rList < rcut12)).T[0]
    d2u12 = np.zeros(300)
    d2u12[args] = -beta * dv12[args]
    c12 = np.zeros(300)
    args = np.argwhere(gr12 > 1e-6).T[0]
    c12[args] = d2u12[args] + (gr12[args] - 1) - np.log(gr12[args])
    u2T12 = x1*x2 * c12 * gr12
    u2T = u2T11 + u2T12
    Phi = np.sum(u2T * rList**2 * MList)
    Phi *= 4 * np.pi * (dr / 3) * rho

    print(Phi)
    exit()
#    plt.show()
#    print(gr[:, 0].shape)
#    exit()
    positions = np.loadtxt("pos_ljT0.47")[:, 2:].copy()
    L = (4000 / 1.2)**(1/3)
    positions /= L
    gamma = 0
    box = np.identity(3) * L
    rho = 1.2
    delta = 0.12 * 0.88
    NA = 3200
    NB = 800
    x1 = NA / (NA + NB)
    x2 = NB / (NA + NB)
    rList = np.linspace(0.005, 2.995, 300)
#    rListAA = np.linspace(0.90, 1.43, 54) - 0.9 + 0.895
#    rListAB = np.linspace(0.74, 1.26, 53) - 0.74 + 0.735
#    rListBB = np.linspace(0.77, 1.05, 29) - 0.77 + 0.765
#    exit()
    grAA, grAB, grBA, grBB = getGR(rList, rList, rList, positions, box, rho, delta, NA, NB)
    grAA /= x1
    grAB /= x2
    grBA /= x1
    grBB /= x2
    grAA = np.sum(grAA, axis = 0) / NA
    grAB = np.sum(grAB, axis = 0) / NA
    grBA = np.sum(grBA, axis = 0) / NB
    grBB = np.sum(grBB, axis = 0) / NB
    plt.plot(rList, grAA, "r--")
    plt.plot(rList, grAB, "g--")
    plt.plot(rList, grBA, "b--")
    plt.plot(rList, grBB, "k--")
    plt.show()
