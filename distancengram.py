#This file uses distance n-gram model
#Need to set the method to calculate transition probability, different lambda, and different method to predict, and set the name
# of the figure
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
import math
import multipolyfit
from matplotlib.ticker import MaxNLocator


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


AllTimeList = list()
WaitTimeList = list()

############################################## set the office Id here
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


'''
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
        
centroids = sorted(centroids)

# draw the histogram for all the data for each bucket
TotalMap = dict()
for i in range(len(WaitTimeArray)):
    if(TotalMap.has_key(AllTimeList[i])):
        TotalMap[AllTimeList[i]].append(WaitTimeArray[i])
    else:
        TotalMap[AllTimeList[i]] = list()
        TotalMap[AllTimeList[i]].append(WaitTimeArray[i])
'''
keyList = list()
for key in sorted(TotalMap.iterkeys()):
    keyList.append(key)


for k in range(1,13):
    plt.figure(k)
    j = 1
    for i in range((k-1)*22,(k-1)*22+22):
        plt.subplot(4,6,j)
        TempList = list(TotalMap[keyList[i]])
        plt.hist(TempList)
        #plt.title(keyList[i])
        plt.savefig(str(k) + '.jpg')
        j = j + 1

#plt.autoscale_view(True,True,True)
plt.show()
'''

'''

TotalMapList = list()
#for 
    #frequency, bin_edges = np.histogram(np.array(TotalMap[key]))
    #plt.subplot(12,22,i)
    #TempList = list(TotalMap[key])
    #plt.hist(TempList)
    #plt.title(key)
    #i = i + 1
    #TotalMapList.extend(list(frequency))
plt.hist(TotalMapList)
plt.show()
'''
    
#draw the histogram for all the data for each hour
'''
HourMap = dict()
for i in range(len(WaitTimeArray)):
    Hour = AllTimeList[i][0] + AllTimeList[i][1] + AllTimeList[i][2]
    if(HourMap.has_key(Hour)):
        HourMap[Hour].append(WaitTimeArray[i])
    else:
        HourMap[Hour] = list()
        HourMap[Hour].append(WaitTimeArray[i])

HourKeyList = list()
for key in sorted(HourMap.iterkeys()):
    HourKeyList.append(key)
'''
    
#draw the histogram for the training data for each hour
TrainWaitTimeArray = WaitTimeArray[:len(WaitTimeArray)/2]
TrainTimeList = AllTimeList[:len(AllTimeList)/2]
HourMap = dict()
for i in range(len(TrainWaitTimeArray)):
    Hour = TrainTimeList[i][0] + TrainTimeList[i][1] + TrainTimeList[i][2]
    if(HourMap.has_key(Hour)):
        HourMap[Hour].append(TrainWaitTimeArray[i])
    else:
        HourMap[Hour] = list()
        HourMap[Hour].append(TrainWaitTimeArray[i])

HourKeyList = list()
for key in sorted(HourMap.iterkeys()):
    HourKeyList.append(key)
    
#Define model function to be used to fit to the data
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

GaussianMap = dict()
for key in HourKeyList:
    data = HourMap[key]
    hist, bin_edges = np.histogram(data,density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    #p0 = [1.,0.,1.]
    x = sum(bin_centres*hist)/sum(hist)
        #coeff, var_matrix = curve_fit(gauss, bin_centres, hist,p0=p0)
        #hist_fit = gauss(bin_centres, *coeff)
    width = sqrt(abs(sum((bin_centres-x)**2*hist)/sum(hist)))
    GaussianMap[key] = list()
    GaussianMap[key].append(x)
    GaussianMap[key].append(width)
    
    
#fit Gaussian distribution to each hour's waiting time


'''
for k in range(13,24):
    plt.figure(k)
    j = 1
    for i in range((k-13)*4, (k-12)*4):
        plt.subplot(2,2,j)
        TempList = list(HourMap[HourKeyList[i]])
        plt.hist(TempList)
        plt.title(str(int(HourKeyList[i][0])+1) + HourKeyList[i][1] + HourKeyList[i][2])
        plt.savefig('Hour'+str(k) + '.jpg')
        j = j + 1
plt.show()
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

#convert the time value to integer and discard "Weekday"
TimeList = [(x[1]+x[2]+x[3]) for x in TimeList]
TimeList = [int(x) for x in TimeList]
TimeArray = np.array(TimeList)
TimeValue = np.unique(TimeArray)
NumTime = len(TimeValue)

#Calculate the probability P(wt|time)
#Convert timevalue to string
TimeString = [str(x).zfill(3) for x in TimeList]
#TimeString = [(str(int(x[0])+1) + x[1:]) for x in TimeString]
TimeValueString = [str(x).zfill(3) for x in list(TimeValue)]
#TimeValueString = [(str(int(x[0])+1) + x[1:]) for x in TimeValueString]
data = np.zeros((NumTime,NumWaitingTime),'d')
PredictWaiting = DataFrame(data,index=TimeValueString,columns=WaitingTimeValue)
for i in range(len(WaitTimeList)):
    tempindex = TimeString[i]
    tempwaitingtime = WaitTimeList[i]
    count = PredictWaiting.get_value(tempindex,tempwaitingtime)
    count = count + 1
    PredictWaiting.set_value(tempindex,tempwaitingtime,count)

for row in TimeValueString:
    SumRow = sum(PredictWaiting.ix[row])
    for column in WaitingTimeValue:
        Current = PredictWaiting.get_value(row,column)
        Current = (Current+1)/float(SumRow+NumWaitingTime)
        PredictWaiting.set_value(row,column,Current)

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
TestTimeList = [(x[1]+x[2]+x[3]) for x in TestTimeList]     #only consider time (no weekday)
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
           # TestOriginWaitTimeSeq[key].append(TestOriginWaitTimeList[index])
            #TestWeekSeq[key].append(TestWeekList[index])
           # TestWeekTimeSeq[key].append(TestWeekTimeList[index])
            key = key + 1
            index = index + 1
            break


errorList = list()
for TimeGap in range(3,19):
    ################Set the gap
    #TimeGap = j
    Transition = np.zeros((NumWaitingTime,NumWaitingTime),'d')
    Transition = DataFrame(Transition)
    index = WaitingTimeValue
    columns = WaitingTimeValue
    Transition = DataFrame(Transition,index=index, columns=columns)
    Count = Series(np.zeros(NumWaitingTime), index=WaitingTimeValue)
    
    ###############################need to convert between the following two
     #only transition and log linear and exponential lambda
    ############Compute P(wt+gap|wt)
    
    for key in WaitingTimeSeq:
        if(len(WaitingTimeSeq[key])>TimeGap):
            for i in xrange(0,len(WaitingTimeSeq[key])-TimeGap):
                Current = Transition.get_value(WaitingTimeSeq[key][i],WaitingTimeSeq[key][i+TimeGap])
                Current = Current + 1
                Count[WaitingTimeSeq[key][i]] = Count[WaitingTimeSeq[key][i]] + 1
                Transition.set_value(WaitingTimeSeq[key][i],WaitingTimeSeq[key][i+TimeGap],Current)
             
    
    '''
    #########compute P(wt-gap|wt)  useful when compute P(wt|wt-gap,ht)=P(wt-gap|wt)*P(wt|ht)    prior plus trans
    for key in WaitingTimeSeq:
        if(len(WaitingTimeSeq[key])>TimeGap):
            for i in xrange(TimeGap,len(WaitingTimeSeq[key])):
                Current = Transition.get_value(WaitingTimeSeq[key][i],WaitingTimeSeq[key][i-TimeGap])
                Current = Current + 1
                Count[WaitingTimeSeq[key][i]] = Count[WaitingTimeSeq[key][i]] + 1
                Transition.set_value(WaitingTimeSeq[key][i],WaitingTimeSeq[key][i-TimeGap],Current)
    '''
    
    for row in WaitingTimeValue:
        SumRow = sum(Transition.ix[row])
        for column in WaitingTimeValue:
            Current = Transition.get_value(row,column)
            Current = Decimal((Current+1))/Decimal((SumRow+NumWaitingTime))
            Transition.set_value(row,column,Current)
    
    # Compute the baseline prediction p(wt|wt-TimeGap) = p(wt-TimeGap|wt)(wt)/p(wt-TimeGap)
    # argmax(wt) p(wt,h|wt-TimeGap) = argmax(wt) p(wt-TimeGap|wt)*(wt,h)
    data = np.zeros((NumTime,NumWaitingTime),'d')
    BaseLinePre = DataFrame(data,index=TimeValueString,columns=WaitingTimeValue)
    
    ############################################Set Lambda Here ##############################
    #Lambda linear with TimeGap
    Lambda = 0.5 + TimeGap * 0.5 / 18
    
    #Lambda exponential to TimeGap
    #Lambda = 1 - 0.5**(TimeGap)
    
    #Equal weight log linear model 
    #Lambda = 0.5
    for items in TimeValueString:
        for columns in WaitingTimeValue:
            candidate = dict()
            for innercolumns in WaitingTimeValue:
                #Use log-linear model
                candidate[innercolumns] = Lambda * math.log((PredictWaiting.get_value(items,innercolumns))) + (1-Lambda) * math.log(Transition.get_value(columns,innercolumns))
                
                #Use just P(wt|w-gap) without P(wt|ht)
                #candidate[innercolumns] = math.log(Transition.get_value(columns,innercolumns))
                
                
                #Use both P(wt|w-gap) and P(wt|ht)
                #candidate[innercolumns] = (PredictWaiting.get_value(items,innercolumns))*Transition.get_value(innercolumns,columns)
            
            
            result = max(candidate.iterkeys(),key=(lambda k : candidate[k]))
            BaseLinePre.set_value(items,columns,int(result))       
    totalError = 0.0
    totalCount = 0.0
    for key in TestWaitTimeSeq:
        for i in range(TimeGap,len(TestWaitTimeSeq[key])):
            Predict = BaseLinePre.get_value(TestTimeSeq[key][i],TestWaitTimeSeq[key][i-TimeGap])
            totalError = totalError + abs(centroids[int(Predict)]-centroids[TestWaitTimeSeq[key][i]])
            totalCount = totalCount + 1
    errorRate = totalError/totalCount
    errorList.append(errorRate)
    print str(errorRate)
#TimeGap=3   error=0.3672 hour
#TimeGap=4   error=0.4238 hour
#TimeGap=5   error=0.4708 hour
#0.5149hour  TimeGap=6
#0.5512      TimeGap=7
#0.5863      TimeGap=8
#0.6092      TimeGap=9
#0.6480      TimeGap=10
#0.6502      TimeGap=11
#0.6750      TimeGap=12
#0.6510      TimeGap=13
#0.6921      TimeGap=14
#0.6952      TimeGap=15
#0.7125      TimeGap=16
#0.7364      TimeGap=17
#0.7576      TimeGap=18
#listA = list([0.3672,0.4238,0.4708,0.5149,0.5512,0.5863,0.6092,0.6480,0.6502,0.6750,0.6510,0.6921,0.6952,0.7125,0.7364,0.7576])
matplotlib.rc('xtick',labelsize=19)
matplotlib.rc('ytick',labelsize=19)
listB = np.arange(30,190,10)
plt.plot(listB,errorList,linewidth=4,color='b')
pylab.yticks(errorList,map(lambda x: "%.2f" %x, errorList))
plt.ylabel('Average error (hours)',fontsize=19)
plt.xlabel('Time gap (minutes)',fontsize=19)
plt.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))
#plt.title('Prediction Error with Gap (log linear model with weight linear to gap)')
#plt.title('Prediction ')
#plt.savefig('priorplustransition.eps')
#plt.savefig('onlytransition.eps')
#plt.savefig("equallambda.eps")
plt.savefig('exponentiallambda.eps')
#plt.savefig('linearlambda.eps')

plt.show()