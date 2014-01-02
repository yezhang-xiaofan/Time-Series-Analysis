#This file uses DBN to model the data. (Time+WeekDay)
from decimal import *
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
from itertools import *
import operator 
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
import json
from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.pgmlearner import PGMLearner

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
                    
'''
#Compute the centroids for this office 
WaitTimeArray = list()
for item in waiting_timeslist:
    if(int(item[1])==Officeid):
        WaitTimeArray.append(float(item[3]))
        AllTimeList.append(item[6])
'''
'''
WaitTimeArray = np.array(WaitTimeArray)
np.vstack(WaitTimeArray)
###############################################################change the number of clusters here 
centroids,_ = kmeans(WaitTimeArray,5, iter=30,thresh=1e-6)
clusters,_ = vq(WaitTimeArray,centroids)

print "The centroids are: " + repr(centroids)
print "The result of clustering is :" + repr(clusters)
NumCluster = len(centroids)
print "The number of centroids is :" + repr(NumCluster)

#Construct the training data
TimeList = list()
for i in xrange(len(WaitTimeArray)/2):
    TimeList.append(AllTimeList[i])
    WaitTimeList.append(clusters[i])
    
#Construct the testing data
TestTimeList = list()
TestWaitTimeList = list()
for i in xrange(len(WaitTimeArray)/2,len(WaitTimeArray)):
    TestTimeList.append(AllTimeList[i])
    TestWaitTimeList.append(clusters[i])
'''
#################################################################equi-height histogram

NumElement = len(WaitTimeArray)
SortedWaitTime = sorted(WaitTimeArray)

###############################################change the number of group here 
NumGroup = 20
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
for key in GroupWaitTime:
    centroids.append(sum(GroupWaitTime[key])/float(len(GroupWaitTime[key])))
centroids = sorted(centroids)

# change the continous waiting time into discrete waiting time 
DisWaitingTime = list()
for i in range(len(WaitTimeArray)):
    DisWaitingTime.append(min(centroids,key=lambda x:abs(x-WaitTimeArray[i])))

######################################Convert the discrete waiting time into the index of state number, DisWaitingTime
for i in range(len(WaitTimeArray)):
    temp = centroids.index(DisWaitingTime[i])
    DisWaitingTime[i] = temp

#Construct the training data, WaitTimeList and TimeList is training data
#WeekList is the "WeekDay" varaiable sample sequence
#
TimeList = list()
WeekList = list()
WeekTimeList = list()
for i in xrange(len(WaitTimeArray)/2):
    WeekTimeList.append(AllTimeList[i])
    TimeList.append(AllTimeList[i][1:])
    WeekList.append(AllTimeList[i][0])
    WaitTimeList.append(DisWaitingTime[i])
    
#Construct the testing data
TestTimeList = list()
TestWeekList = list()
TestWaitTimeList = list()
TestWeekTimeList = list()
TestOriginWaitTimeList = list()
for i in xrange(len(WaitTimeArray)/2,len(WaitTimeArray)):
    TestWeekTimeList.append(AllTimeList[i])
    TestTimeList.append(AllTimeList[i][1:])
    TestWeekList.append(AllTimeList[i][0])
    TestWaitTimeList.append(DisWaitingTime[i])
    TestOriginWaitTimeList.append(WaitTimeArray[i])
    
DisWaitingTime = np.array(DisWaitingTime)
WaitingTimeValue = np.unique(DisWaitingTime)
NumWaitingTime = WaitingTimeValue.size

TimeValue = np.unique(np.array([x[1:] for x in AllTimeList]))
NumTime = len(TimeValue)

WeekValue = np.unique(np.array([x[0] for x in AllTimeList]))
NumWeek = len(WeekValue)

WeekTimeValue = np.unique(np.array(AllTimeList))
NumWeekTime = len(WeekTimeValue)

#Construct the training sequences 
index = 0
WaitingTimeSeq = dict()
TimeSeq = dict()
WeekSeq = dict()
OriginalWaitingTimeSeq = dict()

key = 0
while(index<len(TimeList)):
    WaitingTimeSeq[key] = list()
    TimeSeq[key] = list()
    OriginalWaitingTimeSeq[key] = list()
    WeekSeq[key] = list()
    print key
    while (True):    
        if((index+2)<=len(TimeList) and WeekTimeList[index+1] >= WeekTimeList[index]):
            WaitingTimeSeq[key].append(DisWaitingTime[index])
            TimeSeq[key].append(TimeList[index])
            WeekSeq[key].append(WeekList[index])
            OriginalWaitingTimeSeq[key].append(WaitTimeArray[index])
            index = index + 1
            print index
            continue
        else:
            WaitingTimeSeq[key].append(DisWaitingTime[index])
            TimeSeq[key].append(TimeList[index])
            OriginalWaitingTimeSeq[key].append(WaitTimeArray[index])
            WeekSeq[key].append(WeekList[index])
            key = key + 1
            index = index + 1
            break
NumSeq = key
'''
#Construct the transition matrix 
Transition = np.zeros((NumWaitingTime,NumWaitingTime),'d')
Transition = DataFrame(Transition)
index = WaitingTimeValue
columns = WaitingTimeValue
Transition = DataFrame(Transition,index=index, columns=columns)
Count = Series(np.zeros(NumWaitingTime), index=WaitingTimeValue)
for key in WaitingTimeSeq:
    print 'This is the {index}st sequence: '.format(index = key+1) + repr(WaitingTimeSeq[key])
    for i in xrange(len(WaitingTimeSeq[key])-1):
        Current = Transition.get_value(WaitingTimeSeq[key][i],WaitingTimeSeq[key][i+1])
        Current = Current + 1
        Count[WaitingTimeSeq[key][i]] = Count[WaitingTimeSeq[key][i]] + 1
        Transition.set_value(WaitingTimeSeq[key][i],WaitingTimeSeq[key][i+1],Current)
'''
'''
Emission = np.zeros((NumWaitingTime,NumTime),'d')
index = WaitingTimeValue
columns = TimeValue
Emission = DataFrame(Emission,index=index,columns=columns)
Count1 = Series(np.zeros(NumWaitingTime), index=WaitingTimeValue)

for key in WaitingTimeSeq:
    for i in xrange(len(WaitingTimeSeq[key])):
        Current = Emission.get_value(WaitingTimeSeq[key][i],TimeSeq[key][i])
        Current = Current + 1
        Count1[WaitingTimeSeq[key][i]] = Count1[WaitingTimeSeq[key][i]] + 1
        Emission.set_value(WaitingTimeSeq[key][i],TimeSeq[key][i],Current)
for row in WaitingTimeValue:
    for column in TimeValue:
        Current = Emission.get_value(row,column)
        Current = (Current+1)/(Count1[row]+NumTime)
        Emission.set_value(row,column,Current)
'''
NumWeek =5
#Construct the CPD1
CPD1 = np.zeros((NumWaitingTime,NumWeek),'d')
index = WaitingTimeValue
columns = WeekValue
CPD1 = DataFrame(CPD1,index = index, columns = columns)
Count = Series(np.zeros(NumWaitingTime), index=WaitingTimeValue)
for key in WaitingTimeSeq:
    for i in range(len(WaitingTimeSeq[key])):
        Current = CPD1.get_value(WaitingTimeSeq[key][i],WeekSeq[key][i])
        Current = Current + 1
        Count[WaitingTimeSeq[key][i]] = Count[WaitingTimeSeq[key][i]] + 1
        CPD1.set_value(WaitingTimeSeq[key][i],WeekSeq[key][i],Current)
for row in WaitingTimeValue:
    for column in WeekValue:
        Current = CPD1.get_value(row,column)
        Current = (Current+1)/(Count[row]+NumWeek)
        CPD1.set_value(row,column,Current)

#Construct the CPD2
CPD2 = np.zeros((NumWaitingTime,NumTime),'d')
index = WaitingTimeValue
columns = TimeValue
CPD2 = DataFrame(CPD2,index = index, columns = columns)
Count = Series(np.zeros(NumWaitingTime), index=WaitingTimeValue)
for key in WaitingTimeSeq:
    for i in range(len(WaitingTimeSeq[key])):
        Current = CPD2.get_value(WaitingTimeSeq[key][i],TimeSeq[key][i])
        Current = Current + 1
        Count[WaitingTimeSeq[key][i]] = Count[WaitingTimeSeq[key][i]] + 1
        CPD2.set_value(WaitingTimeSeq[key][i],TimeSeq[key][i],Current)
for row in WaitingTimeValue:
    for column in TimeValue:
        Current = CPD2.get_value(row,column)
        Current = (Current+1)/(Count[row]+NumTime)
        CPD2.set_value(row,column,Current)
        
#Construct the initial matrix
Initial = Series(np.zeros(NumWaitingTime), index = WaitingTimeValue)
for key in WaitingTimeSeq:
    if (key>0) :
        Current = Initial[WaitingTimeSeq[key][0]]
        Current = Current + 1
        Initial[WaitingTimeSeq[key][0]] = Current
Initial = (Initial+1)/(NumSeq-1+NumWaitingTime)

#Construct the transition matrix
Transition = np.zeros((NumWaitingTime,NumWaitingTime),'d')
index = WaitingTimeValue
columns = WaitingTimeValue
Transition = DataFrame(Transition,index=index, columns=columns)
Count = Series(np.zeros(NumWaitingTime), index=WaitingTimeValue)
for key in WaitingTimeSeq:
    for i in xrange(len(WaitingTimeSeq[key])-1):
        Current = Transition.get_value(WaitingTimeSeq[key][i],WaitingTimeSeq[key][i+1])
        Current = Current + 1
        Count[WaitingTimeSeq[key][i]] = Count[WaitingTimeSeq[key][i]] + 1
        Transition.set_value(WaitingTimeSeq[key][i],WaitingTimeSeq[key][i+1],Current)
for row in WaitingTimeValue:
    SumRow = sum(Transition.ix[row])
    for column in WaitingTimeValue:
        Current = Transition.get_value(row,column)
        Current = (Current+1)/(SumRow+NumWaitingTime)
        Transition.set_value(row,column,Current)

#Construct DBN
cpd1 = np.array(CPD1)
cpd1 = normalize_cpd(cpd1)
cpd2 = np.array(CPD2)
cpd2 = normalize_cpd(cpd2)
Initial = np.array(Initial)
Initial = normalize_cpd(Initial)
Transition = np.array(Transition)
Transition = normalize_cpd(Transition)

node0 = DiscreteNode(node_size=NumWaitingTime, name='WaitingTime',user_cpd=Initial)
node1 = DiscreteNode(node_size=NumWeek, name='Weekday',user_cpd=cpd1)
node2 = DiscreteNode(node_size=NumTime, name='Time',user_cpd=cpd2)
node3 = DiscreteNode(node_size=NumWaitingTime,name='WaitingTime1',user_cpd=Transition)
start_nodes = [node0,node1,node2]
end_nodes = [node3, node1,node2]
dbn = DBN(start_nodes,end_nodes)
dbn.add_intra(0,1)
dbn.add_intra(0,2)
dbn.add_inter(0,0)
dbn.construct()

'''
#Construct the testing data
TestTimeList = list()
TestWeekList = list()
TestWaitTimeList = list()
TestWeekTimeList = list()
TestOriginWaitTimeList = list()
for i in xrange(len(WaitTimeArray)/2,len(WaitTimeArray)):
    TestWeekTimeList.append(AllTimeList[i])
    TestTimeList.append(AllTimeList[i][1:])
    TestWeekList.append(AllTimeList[i][0])
    TestWaitTimeList.append(DisWaitingTime[i])
    TestOriginWaitTimeList.append(WaitTimeArray[i])
'''

#split the testing data into sequences
key = 0
index = 0
TestWaitTimeSeq = dict()
TestTimeSeq = dict()
TestWeekSeq = dict()
TestOriginWaitTimeSeq = dict()
TestWeekTimeSeq = dict()

while(index<len(TestWeekTimeList)):
    TestWaitTimeSeq[key] = list()
    TestTimeSeq[key] = list()
    TestWeekSeq[key] = list()
    TestOriginWaitTimeSeq[key] = list()
    TestWeekTimeSeq[key] = list()    
    print key
    while (True):    
        if((index+2)<=len(TestWeekTimeList) and TestWeekTimeList[index+1] >= TestWeekTimeList[index]):
            TestOriginWaitTimeSeq[key].append(TestOriginWaitTimeList[index])
            TestTimeSeq[key].append(TestTimeList[index])
            TestWeekSeq[key].append(TestWeekList[index])
            TestWaitTimeSeq[key].append(TestWaitTimeList[index])
            index = index + 1
            TestWeekTimeSeq[key].append(TestWeekTimeList[index])
            print index
            continue
        else:
            TestWaitTimeSeq[key].append(TestWaitTimeList[index])
            TestTimeSeq[key].append(TestTimeList[index])
            TestOriginWaitTimeSeq[key].append(TestOriginWaitTimeList[index])
            TestWeekSeq[key].append(TestWeekList[index])
            TestWeekTimeSeq[key].append(TestWeekTimeList[index])
            #TestWaitTimeSeq[key].append(TestWaitTimeList[index])
            key = key + 1
            index = index + 1
            break

#TestWeekList = [float(x[0]) for x in TestWeekList]
#TestTimeList = [float(list(TimeValue).index(x[1:])) for x in TestWeekTimeValue]

TotalError = 0.0
TotalCount = 0.0
for i in xrange(1,len(TestWaitTimeSeq)):
    Error = 0.0
    HiddenList = [0.0] * len(TestWeekTimeSeq[i])
    TestWeekList = [float(x) for x in TestWeekSeq[i]]
    TestTimeList = [float (list(TimeValue).index(x)) for x in TestTimeSeq[i]]
    seq = np.array([HiddenList,TestWeekList,TestTimeList])
    testseq = np.transpose(seq)
    testseq.dtype = 'float64'
    hidden_node_indices = [0]
    ie = InfEngineHMM(dbn,testseq,hidden_node_indices)
    viterbi_path, ll=ie.get_viterbi()

#convert the state index into waitingtime value
    FinalPrediction = [centroids[int(x)] for x in viterbi_path[:,0]]
#FinalPrediction = Series(FinalPrediction,index=TestWeekTimeList)

#Put the test data into hashmap. key is time bucket, value is waiting time
    
    for j in xrange(len(FinalPrediction)):
        error = abs(FinalPrediction[j]-centroids[int(TestWaitTimeSeq[i][j])])
        TotalError = TotalError + error
    TotalCount += len(FinalPrediction)
    
AverageError = TotalError/TotalCount

print "The average error is: " + repr(AverageError)
#### The average error is 0.7340 hour

'''
    
TestMap = dict()
for i in xrange(len(TestTimeList)):
    if(TestMap.has_key(int(TestWeekTimeList[i]))):
        TestMap[int(TestWeekTimeList[i])].append(centroids[TestWaitTimeList[i]])
    else:
        TestMap[int(TestWeekTimeList[i])] = list()
        TestMap[int(TestWeekTimeList[i])].append(centroids[TestWaitTimeList[i]])

DifferMap = {}
for key in FinalPrediction.index:
    a = map(float, TestMap[key])
    b = [FinalPrediction[key]]*len(TestMap[key])    
    DifferList = [a-b for a,b in zip(a,b)]
    Average_Error = sum(map(abs,DifferList))/len(DifferList)
    
    DifferMap[key] = list()
    #DifferMap[key].append(CountTime[key][2])
    DifferMap[key].append(Average_Error)
error = (sum(DifferMap.values()))/float(len(DifferMap))
print "The average error is: " + repr(error) +" hours"
'''

        
