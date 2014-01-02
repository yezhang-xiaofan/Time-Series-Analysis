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

# This file computes the average error against time using only time without weekday
#This file computes the average error over all offices in each bucket. 


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
    

count = 0
TotalError = 0.0


waiting_timeslist[1]
for Officeid in OfficeId:
    
    AllTimeList = list()
    WaitTimeList = list()

#Compute the centroids for this office 
    WaitTimeArray = list()
    for item in waiting_timeslist:
        if((item[1])==Officeid):
            WaitTimeArray.append(float(item[3]))
            AllTimeList.append(item[6])

    WaitTimeArray = np.array(WaitTimeArray)

    np.vstack(WaitTimeArray)
    centroids,_ = kmeans(WaitTimeArray,30, iter=50,thresh=1e-10)
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
    print "The length of WaitTimeList is: " + repr(len(WaitTimeList))
    print "The length of TimeList is: " + repr(len(TimeList))
    print "The waiting time list is: " + repr(WaitTimeList)
    '''
    
    DisWaitingTime = np.array(WaitTimeList)
    MaxWaitingTime = DisWaitingTime.max()
    MinWaitingTime = DisWaitingTime.min()
    WaitingTimeValue = np.unique(DisWaitingTime)
    NumWaitingTime = WaitingTimeValue.size
    
    '''
    print "The maximum value of waiting time is : " + repr(MaxWaitingTime)
    print "The minimum value of waiting time is : " + repr(MinWaitingTime)
    print "The number of possible value of waiting time is : " + repr(NumWaitingTime)
    '''
    '''
    DisWaitingTime = [float(Decimal("%.1f" % e)) for e in WaitTimeList]
    #   print DisWaitingTime 
    '''

    TimeList = [int(x) for x in TimeList]
    print "The TimeList is: " + repr(TimeList)
    TimeArray = np.array(TimeList)
    TimeValue = np.unique(TimeArray)
    print "The possible value of Time is: " + repr(TimeValue)
    NumTime = TimeValue.size
    print "The number of possible value of Time is : " + repr(NumTime)

    '''
    TimeList = [x.zfill(4) for x in TimeList]
    print TimeList
    index = 0
    '''

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
    print "The number of sequence is: " + repr(key)

    for i in range(NumSeq):
        print 'The {index}th WaitingTimesequence is: '.format(index=i+1) + repr(WaitingTimeSeq[i])

    #Construct the sequence data
    #data = np.array(WaitingTimeSeq[2],TimeSeq[2])
    #data = np.transpose(data)

    #Construct the transition and emission matrix
    Transition = np.zeros((NumWaitingTime,NumWaitingTime),'d')
    print "Transition Matrix is: " + repr(Transition)
    
    Transition = DataFrame(Transition)
    print repr(Transition)
    index = WaitingTimeValue
    print 'The index is' + repr(index)

    columns = WaitingTimeValue
    print 'The column is ' + repr(columns)
    Transition = DataFrame(Transition,index=index, columns=columns)
    #Transition = DataFrame(Transition,columns=columns)
    #print "The named transition matrix is : " + repr(Transition)
    Count = Series(np.zeros(NumWaitingTime), index=WaitingTimeValue)

    for key in WaitingTimeSeq:
        print 'This is the {index}st sequence: '.format(index = key+1) + repr(WaitingTimeSeq[key])
        for i in xrange(len(WaitingTimeSeq[key])-1):
            Current = Transition.get_value(WaitingTimeSeq[key][i],WaitingTimeSeq[key][i+1])
            Current = Current + 1
            Count[WaitingTimeSeq[key][i]] = Count[WaitingTimeSeq[key][i]] + 1
            Transition.set_value(WaitingTimeSeq[key][i],WaitingTimeSeq[key][i+1],Current)

    for row in WaitingTimeValue:
        for column in WaitingTimeValue:
            Current = Transition.get_value(row,column)
            Current = (Current+1)/(Count[row]+NumWaitingTime)
            Transition.set_value(row,column,Current)
    print Transition
    
    Emission = np.zeros((NumWaitingTime,NumTime),'d')
    #print "Transition Matrix is: " + repr(Transition)
    
    #Emission = DataFrame(Emission)
    index = WaitingTimeValue
    print 'The index of Emission Matrix is: ' + repr(index)

    columns = TimeValue
    print 'The column is ' + repr(columns)
    Emission = DataFrame(Emission,index=index,columns=columns)
    print "The named transition matrix is : " + repr(Emission)

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

    print "The emission matrix is: " + repr(Emission)

    #Initial Probability
    Initial = Series(np.zeros(NumWaitingTime), index = WaitingTimeValue)
    
    for key in WaitingTimeSeq:
        if (key>0) :
            Current = Initial[WaitingTimeSeq[key][0]]
            Current = Current + 1
            Initial[WaitingTimeSeq[key][0]] = Current

    Initial = (Initial+1)/(NumSeq-1+NumWaitingTime)
    
    #Convert pandas dataframe to numpy array
    subset = Transition[WaitingTimeValue]
    tuples = [tuple(x) for x in subset.values]
    Transition = np.array(tuples)

    subset = Emission[TimeValue]
    tuples = [tuple(x) for x in subset.values]
    Emission = np.array(tuples)


    #tuples = [tuple(x) for x in Initial.values]
    Initial = np.array(Initial)

    print "The Initial probability is: " + repr(Initial)
    print "The Emission probability is: " + repr(Emission)

    n_components = NumWaitingTime
    n_symbols = NumTime
    startprob = Initial
    transmat = Transition
    emissionprob_ = Emission

    emitter = hmm.MultinomialHMM(n_components, startprob, transmat)

    emitter.emissionprob_ = emissionprob_
    observations = arange(NumTime)
    observations = observations.reshape(len(observations),)
    Prediction = emitter.predict(observations)
    FinalPrediction = list()

    for i in xrange(len(Prediction)):
        FinalPrediction.append(centroids[Prediction[i]])
    
    print "The prediction is: " + repr(Prediction)
    print "The FinalPrediction[i]" + repr(FinalPrediction)

    '''
    plt.plot(FinalPrediction)
    plt.show()
    '''
    
    FinalPrediction = Series(FinalPrediction,index=TimeValue)

    #Compute the error
    for i in xrange(len(TestWaitTimeList)):
        TestWaitTimeList[i] = centroids[TestWaitTimeList[i]]

    TestMap = dict()
    for i in xrange(len(TestTimeList)):
        if(TestMap.has_key(int(TestTimeList[i]))):
            TestMap[int(TestTimeList[i])].append(TestWaitTimeList[i])
        else:
            TestMap[int(TestTimeList[i])] = list()
            TestMap[int(TestTimeList[i])].append(TestWaitTimeList[i])

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
    count = count + 1
    TotalError = TotalError + error 


AverageError = TotalError/count