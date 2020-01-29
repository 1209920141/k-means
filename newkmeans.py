import numpy as np
import matplotlib.pyplot as plt


# generate random cluster center from the points in the data
def randomCenterFromData(dataSet, k):
    m = dataSet.shape[0]
    n=2
    # random select points from data
    centroidList = np.zeros((k, n))
    for i in range(k):
        index = int(np.random.uniform(0, m))  #
        centroidList[i, :] = dataSet[index, :]

    print("Generated centroid from random data points：")
    print(centroidList)
    return centroidList

# generate random cluster center location
def randomCenterLoc(k):
    # generate center loc list from -10 to 10
    # centroidList = (np.random.rand(k,2)*20)-10
    centroidList = (np.random.rand(k, 2) * 10) - 5
    print("Generated centroid randomly by location：")
    print(centroidList)
    return centroidList

def KMeans(trainData, k, a):
    m = np.shape(trainData)[0]
    clusterResult = np.mat(np.zeros((m, 2)))
    changeLabel = True

    # Choose generation methods
    if a==1:
        centroidList = randomCenterFromData(trainData, k)
    elif a==2:
        centroidList = randomCenterLoc(k)
    else:
        print("A wrong choice but will choose initial from data")
        centroidList = randomCenterFromData(trainData, k)

    iterationCycle = 0
    while changeLabel:
        iterationCycle += 1
        changeLabel = False
        for i in range(m):
            minDist = 99999
            minIndex = -1
            for j in range(k):
                # calculate euclidean distance
                distance = np.sqrt(np.sum((centroidList[j, :] - trainData[i, :]) ** 2))
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            if clusterResult[i, 0] != minIndex:
                changeLabel = True
                clusterResult[i, :] = minIndex, minDist ** 2
        # refresh centroid
        for j in range(k):
            pointsInCluster = trainData[np.nonzero(clusterResult[:, 0].A == j)[0]]
            centroidList[j, :] = np.mean(pointsInCluster, axis=0)

    print("Cluster finish!")
    return centroidList, clusterResult, iterationCycle

# draw the result into plot
def draw(trainData, k, centroidList, clusterResult):
    m, n = trainData.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    mark2 = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(m):
        markIndex = int(clusterResult[i, 0])
        plt.plot(trainData[i, 0], trainData[i, 1], mark[markIndex])
    for i in range(k):
        plt.plot(centroidList[i, 0], centroidList[i, 1], mark2[i])
    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.title("K-means result")
    plt.show()

# Used in calculateACC, to find the classification of the current cluster
def getCenterType(trainData, index):
    m = trainData.shape[0]
    type = 0
    minDist = 99999
    for i in range(m):
        distance = np.sqrt(np.sum((index - trainData[i, :]) ** 2))
        if distance < minDist:
            minDist = distance
            type = data[i,2]
    return type

# calculate the true positive of the result(Only for the center is from the data)
def calculateACC(clusterResult,k,data,centroidList,trainData):

    print(centroidList)
    if np.isnan(np.min(centroidList)):
        print("Some center is abandoned, can't calculate accuracy!!")
    else:
        Tamount = 0
        m = data.shape[0]
        # print(centroidList)
        for j in range(k):
            type = getCenterType(trainData, centroidList[j,:])
            for i in range(m):
                if int(clusterResult[i,0]) == type:
                    Tamount += 1

        ACC = Tamount/m
        print("The ACC is: ",ACC)

# calculate the average distance in a cluster
def calculateTotalDistance(trainData, centroidList,clusterResult):
    distance =0
    m = trainData.shape[0]

    for i in range(m):
        for j in range(k):
            t = int(clusterResult[i,0])
            distance += np.sqrt(np.sum((centroidList[t,:] - trainData[i, :]) ** 2))
    return distance/k


def compareKAndDraw(trainData,k,a):
    distanceList = []
    for k in range(1,9):
        centroidList, clusterResult, iterationCycle = KMeans(trainData, k, a)
        distanceList.append(calculateTotalDistance(trainData, centroidList, clusterResult))
    kList = [1,2,3,4,5,6,7,8]
    plt.plot(kList,distanceList)
    plt.xlabel('avg distance per cluster')
    plt.ylabel('k value')
    plt.title("K value compare")
    plt.show()


if __name__ == '__main__':
    # read data
    data = np.loadtxt("k-means-data.txt")
    trainData = data[:,[0,1]]

    # set k value
    k = int(input("Input a integar between 1 and 8:  "))
    print("Input 1 for initial from data, input 2 for initial from anywhere")
    a = int(input("Choice: "))
    b = int(input("Wanna compare different k value? 1 for yes, others no: "))

    centroidList, clusterResult, iterationCycle = KMeans(trainData, k, a)
    print("The number of cycle that iterates is: ",iterationCycle)

    draw(trainData, k, centroidList, clusterResult)

    if k==3:
        calculateACC(clusterResult,k,data,centroidList,trainData)
    if b==1:
        compareKAndDraw(trainData,k,a)