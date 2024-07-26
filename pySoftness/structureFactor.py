import libstructureFactor as sf
import numpy as np
from matplotlib import pyplot as plt

def getGR(rListAA, rListAB, rListBB, positions, box, rho, delta, NA, NB):
    numAA = len(rListAA)
    numAB = len(rListAB)
    numBB = len(rListBB)
    rListLengths = np.array([numAA, numAB, numBB])
    rLists = np.concatenate((rListAA, rListAB, rListBB))
    positions = np.dot(box, positions.copy().T).T
    L = box[0, 0]
    gamma = box[0, 1]
    # What form does gr take?
    gr = np.zeros((rListLengths[0] + rListLengths[1]) * NA + (rListLengths[1] + rListLengths[2]) * NB)
    # particle 1 takes indices 0 to 2 * numR where 0, 1 are gAA and gAB
    # particles of type be have indices gBB, gBA
    flatPositions = positions.reshape(np.prod(positions.shape))
    sf.getGR(gr, rLists, flatPositions, rListLengths, L, gamma, rho, delta, NA, NB)
    # Reshape this
    indicesList = np.array([0, numAA * NA, (numAA + numAB) * NA, (numAA + numAB) * NA + numAB * NB, (numAA + numAB) * NA + (numBB + numAB) * NB])
    grAA = gr[indicesList[0]: indicesList[1]].reshape(NA, numAA)
    grAB = gr[indicesList[1]: indicesList[2]].reshape(NA, numAB)
    grBA = gr[indicesList[2]: indicesList[3]].reshape(NB, numAB)
    grBB = gr[indicesList[3]: indicesList[4]].reshape(NB, numBB)
    return [grAA, grAB, grBA, grBB]



if __name__ == "__main__":
    positions = np.loadtxt("pos_ljT0.47")[:, 2:].copy()
    L = (4000 / 1.2)**(1/3)
    positions /= L
    gamma = 0
    box = np.identity(3) * L
    rho = 1.2
    delta = 0.09
    NA = 3200
    NB = 800
    x1 = NA / (NA + NB)
    x2 = NB / (NA + NB)
    rListAA = np.linspace(0.90, 1.43, 54) - 0.9 + 0.895
    rListAB = np.linspace(0.74, 1.26, 53) - 0.74 + 0.735
    rListBB = np.linspace(0.77, 1.05, 29) - 0.77 + 0.765
    grAA, grAB, grBA, grBB = getGR(rListAA, rListAB, rListBB, positions, box, rho, delta, NA, NB)
    grAA /= x1
    grAB /= x2
    grBA /= x1
    grBB /= x2
    grAA = np.sum(grAA, axis = 0) / NA
    grAB = np.sum(grAB, axis = 0) / NA
    grBA = np.sum(grBA, axis = 0) / NB
    grBB = np.sum(grBB, axis = 0) / NB
    plt.plot(rListAA, grAA, "r")
    plt.plot(rListAB, grAB, "g")
    plt.plot(rListAB, grBA, "b")
    plt.plot(rListBB, grBB, "k")
    plt.show()
