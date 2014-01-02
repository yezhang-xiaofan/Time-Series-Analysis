#This file draw waiting time plot for all days
import numpy as np
from Mocapy import *
from numpy import *
from scipy import *
#import statsmodels.api as sm
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
#from RndCoV import RndCov
#from ghmm import *
from sklearn import hmm
from sklearn.hmm import MultinomialHMM
from itertools import *
import operator 
import xlrd
import xlwt
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
from scipy.optimize import curve_fit
import pylab as P



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
        
with open('20130323_waiting_times.csv') as csvfile:
    waiting_times = csv.reader(csvfile)
    waiting_timeslist = list()
    for row in waiting_times:
        waiting_timeslist.append(row)

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

# Compare the key sets of two dictionary
# If they are equal, return true
def ComKeyDic(dic1,dic2):
    return (set(dic1.keys()) == set(dic2.keys()))

# Convert String to Datetime
# each id:
# 0:d 1:office_id 2:w_appointment 3:wo_appointment 4:created_at 5:updated_at 6:Hash Value

col = 4
for i in xrange(number):
    
    #extract the time string and convert them to standard EST datetime
    
    temp = datetime.strptime(waiting_timeslist[i][col], "%Y-%m-%d %H:%M:%S")   
    utc = pytz.UTC
    ams = pytz.timezone('US/Pacific')
    waiting_timeslist[i][col] = utc.localize(temp)
    waiting_timeslist[i][col] = waiting_timeslist[i][col].astimezone(ams)
    
    temp = waiting_timeslist[i]
    waiting_timeslist[i].append(Hashfunction(temp))
    
#Split the whole set into train_set and test_set

train_set = waiting_timeslist[1:number/2]
test_set = waiting_timeslist[number/2:]

BucketMap = dict()
    
for Officeid in map(int,OfficeId):
    WaitTimeArray = list()
    AllTimeList = list()
    WaitTimeList = list()
    for item in waiting_timeslist:
        if(int(item[1])==Officeid):
            if(item[4].weekday()==2):
                if(item[4].hour>=9 and item[4].hour<=17):
                    if(item[4].hour!=17):
                        WaitTimeArray.append(float(item[3]))
                        AllTimeList.append(item[6])
                    else:
                        if(item[4].minute==0):
                            WaitTimeArray.append(float(item[3]))
                            AllTimeList.append(item[6])
                            
            else:
                if(item[4].hour>=8 and item[4].hour<=17):
                    if(item[4].hour!=17):
                        WaitTimeArray.append(float(item[3]))
                        AllTimeList.append(item[6])
                    else:
                        if(item[4].minute==0):
                            WaitTimeArray.append(float(item[3]))
                            AllTimeList.append(item[6])
    
    #Remove the day with all zeros
    def FindRemoveDay(TempTime):
        Day = dict()
        for item in TempTime:
            if (item[0] not in Day):
                Day[item[0]] = 1
            else:
                TempCount = Day[item[0]] + 1
                Day[item[0]] = TempCount
        return max(Day.iterkeys() , key=(lambda key : Day[key]))    
    groups = []
    uniquekeys = []
    for k, g in groupby(enumerate(zip(AllTimeList,WaitTimeArray)), lambda(i,(x,y)): y==0.0):
        groups.append(list(g))
        uniquekeys.append(k)
    
    RemoveIndex = list()
    
    for i in range(len(uniquekeys)):
        if(uniquekeys[i]==True):
            if(len(groups[i])>=50):
                (Index,TempValue) = zip(*groups[i])
                (TempTime,TempWaitTime) = zip(*TempValue)
                RemoveDay = FindRemoveDay(TempTime)
                for j in range(len(TempTime)):
                    if(TempTime[j][0]==RemoveDay):
                        RemoveIndex.append(Index[j])
    tempWaitTime = [item for i, item in enumerate(WaitTimeArray) if i not in RemoveIndex]
    WaitTimeArray = tempWaitTime
    tempTime = [item for i , item in enumerate(AllTimeList) if i not in RemoveIndex]
    AllTimeList = tempTime
        
    for i in range(len(WaitTimeArray)):
        bucket = AllTimeList[i][1] + AllTimeList[i][2] + AllTimeList[i][3]
        if(BucketMap.has_key(bucket)):
            BucketMap[bucket].append(WaitTimeArray[i])
        else:
            BucketMap[bucket] = list()
            BucketMap[bucket].append(WaitTimeArray[i])

bucketList = list()
for key in sorted(BucketMap.iterkeys()):
    bucketList.append(key)

#Define model function to be used to fit to the data
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

GaussianMap = dict()
widthList = list()
meanList = list()
for key in bucketList:
    data = BucketMap[key]
    hist, bin_edges = np.histogram(data,density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    #p0 = [1.,0.,1.]
    #x = sum(bin_centres*hist)/sum(hist)
    mean = np.mean(np.array(data))
        #coeff, var_matrix = curve_fit(gauss, bin_centres, hist,p0=p0)
        #hist_fit = gauss(bin_centres, *coeff)
    #width = sqrt(abs(sum((bin_centres-x)**2*hist)/sum(hist)))
    meanList.append(mean)
    std = sqrt(sum((data-mean)**2)/(len(data)-1))
    GaussianMap[key] = list()
    GaussianMap[key].append(mean)
    GaussianMap[key].append(std)
    widthList.append(std)
widthArray = np.array(widthList)
meanArray = np.array(meanList)
#sortedBucket = sorted(bucketList.iterkeys())
#sortedBucket = np.array(int(bucketList))
xticks = list()

for bucket in bucketList:
    tempBucket = bucket[:2] + ":" + bucket[2]+"0"
    xticks.append(time.strptime(tempBucket,"%H:%M"))
#xticks = np.array(xticks)
ind = np.arange(len(bucketList))
width = 1
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(ind,meanArray,width,color='r',yerr=widthArray)
#hours = mdates.HourLocator()
Ind = np.arange(0.5,54,6)
ax.set_xticks(Ind)
ax.set_xticklabels(['8:00','9:00','10:00','11:00','12:00','13:00','14:00','15:00','16:00'])
#ax.xaxis.set_major_locator(hours)
#timeMin = datetime(1900,1,1,8,0,0)
#timeMax = datetime(1900,1,1,16,50,0)
#ax.set_xlim(timeMin,timeMax)
#formatter = DateFormatter('%H:%M')
#ax.xaxis.set_major_formatter(formatter)
#fig.autofmt_xdate()
ax.set_ylabel('Waiting Time')
ax.set_title('Waiting time by time bucket (10 minutes)')
ax.set_xlabel('Time')

plt.show()
print "haha"
print "haha"