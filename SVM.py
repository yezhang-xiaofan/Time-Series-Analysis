#This file uses SVM to classify waiting time

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
import math
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
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.regression.linear_model import WLS, GLS

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
        
with open('20130128_waiting_times.csv') as csvfile:
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

def ComKeyDic(dic1,dic2):
    return (set(dic1.keys()) == set(dic2.keys()))

# Convert String to Datetime
# each id:
# 0:d 1:office_id 2:w_appointment 3:wo_appointment 4:created_at 5:updated_at 6:Hash Value

col = 4
for i in xrange(number):
    
    #extract the time string and convert them to standard PST datetime
    
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


AllTimeList = list()
WaitTimeList = list()

#Office548 = dict()
Officeid = 632

###Only keep the qualified tuples 
WaitTimeArray = list()
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

NumElement = len(WaitTimeArray)
SortedWaitTime = sorted(WaitTimeArray)



'''
#################################################################equi-height histogram

NumElement = len(WaitTimeArray)
SortedWaitTime = sorted(WaitTimeArray)
ZeroWaitTime = list()
NonZeroWaitTime = list()
for i in range(len(SortedWaitTime)):
    if(SortedWaitTime[i]==0):
        ZeroWaitTime.append(0)
    else:
        NonZeroWaitTime.append(SortedWaitTime[i])
'''

###############################################change the number of group here 
NumGroup = 20
#NumNonZero = len(SortedWaitTime)
Count = math.ceil(float(NumElement)/NumGroup)
GroupWaitTime = dict()
GroupIndex = 0
GroupWaitTime[GroupIndex] = list()
for i in range(NumElement):
    GroupWaitTime[GroupIndex].append(SortedWaitTime[i])                 
    if((i+1) % Count == 0):
        GroupIndex = GroupIndex + 1
        GroupWaitTime[GroupIndex] = list()

centroids= list()
#centroids.append(0.0)        #add zeros to centroids
for key in GroupWaitTime:
    centroids.append(sum(GroupWaitTime[key])/float(len(GroupWaitTime[key])))
centroids = sorted(centroids)

# change the continous waiting time into discrete waiting time 
DisWaitingTime = list()
for i in range(len(WaitTimeArray)):
    '''
    if(WaitTimeArray[i]==0):
        DisWaitingTime.append(WaitTimeArray[i])
    '''       
    DisWaitingTime.append(min(centroids,key=lambda x:abs(x-WaitTimeArray[i])))

'''
WaitTimeArray = np.array(WaitTimeArray)
np.vstack(WaitTimeArray)
###############################################################change the number of clusters here 
centroids,_ = kmeans(WaitTimeArray,10, iter=30,thresh=1e-6)
clusters,_ = vq(WaitTimeArray,centroids)

print "The centroids are: " + repr(centroids)
print "The result of clustering is :" + repr(clusters)
NumCluster = len(centroids)
print "The number of centroids is :" + repr(NumCluster)

#Construct the training data after clustering 
TimeList = list()
for i in xrange(len(WaitTimeArray)/2):
    TimeList.append(AllTimeList[i])
    WaitTimeList.append(clusters[i])
    
#Construct the testing data after clustering 
TestTimeList = list()
TestWaitTimeList = list()
for i in xrange(len(WaitTimeArray)/2,len(WaitTimeArray)):
    TestTimeList.append(AllTimeList[i])
    TestWaitTimeList.append(clusters[i])

'''


######################################Convert the discrete waiting time into the index of state number, DisWaitingTime
for i in range(len(WaitTimeArray)): 
    temp = centroids.index(DisWaitingTime[i])
    DisWaitingTime[i] = temp

#Construct the training data, WaitTimeList and TimeList is training data
TimeList = list()
for i in xrange(len(WaitTimeArray)/2):
    TimeList.append(AllTimeList[i])
    WaitTimeList.append(DisWaitingTime[i])
    
#Construct the testing data(discrete waiting time)
TestTimeList = list()
TestWaitTimeList = list()
for i in xrange(len(WaitTimeArray)/2,len(WaitTimeArray)):
    TestTimeList.append(AllTimeList[i])
    TestWaitTimeList.append(DisWaitingTime[i])

DisWaitingTime = np.array(DisWaitingTime)
WaitingTimeValue = np.unique(DisWaitingTime)
NumWaitingTime = WaitingTimeValue.size

#convert the time value to integer and discard "Weekday" in the training data set
TimeList = [(x[1]+x[2]+x[3]) for x in TimeList]
TimeList = [int(x) for x in TimeList]
TimeArray = np.array(TimeList)
TimeValue = np.unique(TimeArray)
NumTime = len(TimeValue)

#convert the time value to integer and discard "Weekday" in the testing data set
TestTimeList = [(x[1]+x[2]+x[3]) for x in TestTimeList]
TestTimeList = [int(x) for x in TestTimeList]
TestTimeArray = np.array(TestTimeList)









#Construct the training sequences 
index = 0
WaitingTimeSeq = dict()
TimeSeq = dict()
OriginalWaitingTimeSeq = dict()
key = 0
while(index<len(TimeList)):
    WaitingTimeSeq[key] = list()
    TimeSeq[key] = list()
    OriginalWaitingTimeSeq[key] = list()
    print key
    while (True):    
        if((index+2)<=len(TimeList) and TimeList[index+1] >= TimeList[index]):
            WaitingTimeSeq[key].append(DisWaitingTime[index])
            TimeSeq[key].append(TimeList[index])
            OriginalWaitingTimeSeq[key].append(WaitTimeArray[index])
            index = index + 1
            print index
            continue
        else:
            WaitingTimeSeq[key].append(DisWaitingTime[index])
            TimeSeq[key].append(TimeList[index])
            OriginalWaitingTimeSeq[key].append(WaitTimeArray[index])
            key = key + 1
            index = index + 1
            break
NumSeq = key

#Construct testing sequence 
key = 0
index = 0
TestWaitTimeSeq = dict()
TestTimeSeq = dict()
#TestOriginWaitTimeSeq = dict()

while(index<len(TestTimeList)):
    TestWaitTimeSeq[key] = list()
    TestTimeSeq[key] = list()
    print key
    while (True):    
        if((index+2)<=len(TestTimeList) and TestTimeList[index+1] >= TestTimeList[index]):
            TestTimeSeq[key].append(TestTimeList[index])
            TestWaitTimeSeq[key].append(TestWaitTimeList[index])
            index = index + 1
            print index
            continue
        else:
            TestWaitTimeSeq[key].append(TestWaitTimeList[index])
            TestTimeSeq[key].append(TestTimeList[index])
            #TestOriginWaitTimeSeq[key].append(TestOriginWaitTimeList[index])
            #TestWeekSeq[key].append(TestWeekList[index])
            #TestWeekTimeSeq[key].append(TestWeekTimeList[index])
            key = key + 1
            index = index + 1
            break
    

#Construct feature list
Num_Feature = 2
FeatureWaitTime = list()
FeatureTime = list()
LabelList = list()
for key in WaitingTimeSeq:
    for i in range(len(WaitingTimeSeq[key])-1):
        FeatureWaitTime.append(WaitingTimeSeq[key][i])
        FeatureTime.append(TimeSeq[key][i+1])
        LabelList.append(WaitingTimeSeq[key][i+1])

FeatureArray = np.zeros((len(LabelList),Num_Feature))
LabelArray = np.array(LabelList)
for i in range(0, len(FeatureTime)):
    FeatureArray[i,0] = FeatureWaitTime[i]
    FeatureArray[i,1] = FeatureTime[i]
    LabelArray[i] = LabelList[i]

clf = svm.SVC()
clf.fit(FeatureArray,LabelArray)

#Test
TotalError = 0.0
TotalCount = 0.0
PredictWaitTimeList = list()
PredictTimeList = list()
TrueValueList = list()

'''
for key in TestWaitTimeSeq:
    for i in range(len(TestWaitTimeSeq[key])-1):
        newPredict = clf.predict(TestWaitTimeSeq[key][i],TestTimeList[i+1])
        PredictWaitTimeList.append(TestWaitTimeSeq[key][i])
        PredictTimeList.append(TestTimeSeq[key][i+1])
        TrueValueList.append(TestWaitTimeSeq[key][i+1])
'''

#PredictFeatureArray = np.zeros((len(TrueValueList),Num_Feature))
#TrueValueArray = np.array(TrueValueList)
for i in range(len(TestWaitTimeList)-1):
    newPredict = clf.predict([TestWaitTimeList[i],TestTimeList[i+1]])
    TotalError += abs(centroids[newPredict] - centroids[TestWaitTimeList[i+1]])
AverageError = TotalError/(len(TestWaitTimeList)-1)
print "average error : " + repr(AverageError)
##0.1636 hour
Error = 0    
for i in range(0,len(PredictValue)):
    if(PredictValue[i]!=TestWaitTimeList[i+1]):
        Error = Error + 1

ErrorPercentage = float(Error) / len(TestWaitTimeList)
Accuracy = 1 - ErrorPercentage
print ErrorPercentage
print "The accuracy is: " + repr(Accuracy)
#0.6190476190476191   