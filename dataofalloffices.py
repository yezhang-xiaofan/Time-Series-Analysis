#Data about all the offices
import json
from bisect import bisect_left
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
class discrete_cdf:
    def __init__(self,data):
        self._data = data # must be sorted
        self._data_len = float(len(data))

    def __call__(self,point):
        return (len(self._data[:bisect_left(self._data, point)]) / 
                self._data_len)


json_data = open('my_dict.json')
data = json.load(json_data)
Max = 0
alldata = list()
AverageWaitingTime = dict()
for key in data:
    AverageWaitingTime[key] = list()
    tempSum = 0.0
    for i in data[key]:
        alldata.append(i)
    AverageWaitingTime[key].append(sum(data[key])/len(data[key]))
matplotlib.rc('xtick',labelsize=19)
matplotlib.rc('ytick',labelsize=19)    
alldata.sort()
cdf = discrete_cdf(alldata)
xvalues = np.arange(0, int(max(alldata)),0.1)
yvalues = [cdf(point) for point in xvalues]
plt.plot(xvalues, yvalues,linewidth=4)
plt.xlabel("Wait time (hours)", fontsize=19)

plt.savefig("cdf.eps")
plt.show()


print Max
print haha

