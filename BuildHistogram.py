import numpy as np
from numpy import *
from histogram import *
#import matplotlib.pyplot as plt
from cluster import KMeansClustering
import numpy as np
from numpy import vstack,array
from scipy.cluster.vq import kmeans,vq
from cluster import KMeansClustering

WaitTimeArray = np.load('Fuck.npy')

np.vstack(WaitTimeArray)
centroids,_ = kmeans(WaitTimeArray,40, iter=50,thresh=1e-10)
clusters,_ = vq(WaitTimeArray,centroids)
print "The centroids are: " + repr(centroids)
print "The result of clustering is :" + repr(clusters)

NumCluster = len(centroids)
ClusterMap = {}
for item in clusters:
    if(ClusterMap.has_key(item)):
        newValue = ClusterMap.get(item) + 1
        ClusterMap[item] = newValue
    else:
        ClusterMap[item] = 1

print "The CLustering is " + repr(ClusterMap)
Count = list()

for key in ClusterMap:
    Count.append(ClusterMap[key])

Count = np.array(Count)
SortIndex = np.argsort(centroids)
Count[SortIndex]






'''
bins = arange(-0.001, max(WaitingTime)+0.001, 0.001)
Histogram, bin_edges = np.histogram(WaitingTime,bins)

h = histogram("Fuck",[('tof',bins)])
'''
'''
plt.hist(WaitingTime,bins)
plt.title("Fuck")
plt.xlabel("Bucket")
plt.ylabel("Frequency")
plt.show()
'''