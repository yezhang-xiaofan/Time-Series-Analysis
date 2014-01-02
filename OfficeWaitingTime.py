#This file builds hashmap on office: waiting time list
#Also draw cdf for a certain office
import numpy as np
from Mocapy import *
from numpy import *
from scipy import *
import pandas
from patsy import dmatrices
from decimal import *
from pandas import *
import pandas as pd
from cluster import KMeansClustering
import numpy as np
from numpy import vstack,array
from scipy.cluster.vq import kmeans,vq
from cluster import KMeansClustering
from sklearn import hmm
from sklearn.hmm import MultinomialHMM
from sklearn import tree

import csv
import time
import re
from datetime import datetime
from pytz import timezone
import pytz
#import xlwt
import pdb
import matplotlib
import matplotlib.pyplot as plt
import pylab
from pylab import plot, show
from matplotlib.dates import date2num
from time import mktime
import matplotlib.pylab as mp
from matplotlib.dates import MinuteLocator, DateFormatter, HourLocator
from pylab import figure
from sklearn import svm
from itertools import *
import json
from bisect import bisect_left
from scipy.stats import norm
'''
#read two files
with open('20130128_offices.csv','rb') as csvfile:
    offices = csv.reader(csvfile)        
    officelist = list()
    for row in offices:
        officelist.append(row)
officelist.pop(0)
        
OfficeId = list()
for item in officelist:
    OfficeId.append(item[0])
Num_Office = len(OfficeId)
print "The number of office is ",
print Num_Office
'''
OfficeWaitingTime = dict()
with open('20130323_waiting_times.csv') as csvfile:
    waiting_times = csv.reader(csvfile)
    for row in waiting_times:
        if(row[3]!='wo_appointment'):
            office = row[1]
            waitingtime = row[3]
            if(OfficeWaitingTime.has_key(office)):
                OfficeWaitingTime[office].append(waitingtime)
            else:
                OfficeWaitingTime[office] = list()
                OfficeWaitingTime[office].append(waitingtime)
AverageWaitingTime = dict()
for key in OfficeWaitingTime:
    tempList = [float(x) for x in OfficeWaitingTime[key]]
    AverageWaitingTime[key] = list()
    AverageWaitingTime[key].append(sum(tempList)/len(tempList))

class discrete_cdf:
    def __init__(self,data):
        self._data = data # must be sorted
        self._data_len = float(len(data))

    def __call__(self,point):
        return (len(self._data[:bisect_left(self._data, point)]) / 
                self._data_len)
matplotlib.rc('xtick',labelsize=19)
matplotlib.rc('ytick',labelsize=19)
#####################################Change the office here 
officeId = '632'
waitTimeList = [float(x) for x in OfficeWaitingTime[officeId]]
waitTimeList.sort()
cdf = discrete_cdf(waitTimeList)
xvalues = np.arange(0, max(waitTimeList),0.1)
yvalues = [cdf(point) for point in xvalues]
plt.plot(xvalues, yvalues,linewidth=4)
plt.xlabel("Wait time (hours)", fontsize=19)

plt.savefig("cdfoffice"+officeId+".eps")
plt.show()

'''
waiting_timeslist.pop(0)

number = len(waiting_timeslist)
    
#define a hashfunction to map the same bucket (ignoring weekday) to a unique value
#Then put the hash value in item[6]

def Hashfunction(item):
    Hash = (str(item[4].weekday())+str(item[4].hour).zfill(2)+str((item[4].minute)/10))
    return Hash

#Define a PutInMap function : put the item with the same hash value into the same key
#the key is the has value
#the value is a list
#0:total waiting time   1: total times
def PutInMap(item,Map):
    temp = item[6]
    if(temp not in Map):
        Map[temp] = list()
        Map[temp].append(float(item[3]))               #if temp is not in hashmap yet 
        Map[temp].append(1)
    else:
        Map[temp][0] = Map[temp][0] + float(item[3])   #add the waiting time
        Map[temp][1] = Map[temp][1] + 1                #count the times

#This function deals with test data set
#The key is the hash value
#the value is a list 
#put the corresponding wo_appointment into the list
def PutInMap2(item,Map):
    temp = item[6]
    if(temp not in Map):
        
        Map[temp] = list()
        Map[temp].append(item[3])
    else:
        Map[temp].append(item[3])

def ComKeyDic(dic1,dic2):
    return (set(dic1.keys()) == set(dic2.keys()))

# Convert String to Datetime
# each id:
# 0:d 1:office_id 2:w_appointment 3:wo_appointment 4:created_at 5:updated_at 6:Hash Value

col = 4
OfficeWaitingTime = dict()
for i in xrange(number):
    
    #extract the time string and convert them to standard PST datetime
    
    #temp = datetime.strptime(waiting_timeslist[i][col], "%Y-%m-%d %H:%M:%S")   
    #utc = pytz.UTC
    #ams = pytz.timezone('US/Pacific')
    #waiting_timeslist[i][col] = utc.localize(temp)
    #waiting_timeslist[i][col] = waiting_timeslist[i][col].astimezone(ams)
    
    #temp = waiting_timeslist[i]
    #waiting_timeslist[i].append(Hashfunction(temp))
'''
