#This file uses linear regression
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

NumElement = len(WaitTimeArray)
SortedWaitTime = sorted(WaitTimeArray)

                    
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

#convert the time value to integer and discard "Weekday" in the training data
TimeList = [(x[1]+x[2]+x[3]) for x in TimeList]
TimeList = [int(x) for x in TimeList]
TimeArray = np.array(TimeList)
TimeValue = np.unique(TimeArray)
NumTime = len(TimeValue)

###convert the time value to integer and discard "weekday" in the testing data
TestTimeList = [(x[1]+x[2]+x[3]) for x in TestTimeList]
TestTimeList = [int(x) for x in TestTimeList]

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



'''
# Convert the sequence to a new sequence based on hour series
HourSeq = dict()
for key in WaitingTimeSeq:
    HourSeq[key] = dict()
    for i in range(len(WaitingTimeSeq[key])):
        HourKey = TimeSeq[key][i][0] + TimeSeq[key][i][1] + TimeSeq[key][i][2]
        if(HourSeq[key].has_key(HourKey)):
            HourMap[key][HourKey].append(WaitingTimeSeq[key][i])
        else:
            HourMap[key][HourKey] = list()
            HourMap[key][HourKey].append(WaitingTimeSeq[key][i])
'''

for i in range(NumSeq):
    print 'The {index}th WaitingTimesequence is: '.format(index=i+1) + repr(WaitingTimeSeq[i])

#index of minimum waiting_time 
Index0 = np.argmin(centroids)

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
           # TestOriginWaitTimeSeq[key].append(TestOriginWaitTimeList[index])
            #TestWeekSeq[key].append(TestWeekList[index])
           # TestWeekTimeSeq[key].append(TestWeekTimeList[index])
            key = key + 1
            index = index + 1
            break

#Create linear regression object
NumAttribute = max([len(WaitingTimeSeq[key]) for key in WaitingTimeSeq]) - 1
OutputList = list()
InputList = list()
for key in range(len(WaitingTimeSeq)):
    for i in range(1,len(WaitingTimeSeq[key])):
        #tempOutput = centroids[WaitingTimeSeq[key][i]]
        tempOutput = WaitingTimeSeq[key][i]
        OutputList.append(tempOutput)
        tempList = WaitingTimeSeq[key][:i][:]
        #tempList = [centroids[j] for j in tempList]        
        tempList.reverse()
        tempList += [0]*(NumAttribute-len(WaitingTimeSeq[key][:i]))
        tempArray = np.array(tempList)
        InputList.append(tempArray)
InputMatrix = np.vstack((InputList))
#OutputMatrix = np.column_stack((OutputList))
OutputMatrix = np.array(OutputList)
X = sm.add_constant(InputMatrix)
rlm_model = sm.RLM(OutputMatrix,X,M=sm.robust.norms.HuberT())
rlm_results = rlm_model.fit()
print rlm_results.params
#residuals = rlm_results.resid
#MSE = np.dot(residuals,residuals)/rlm_results.df_resid
#print "The Mse of the training data is: " + repr(MSE)
#2.7096

#PredictList = list()
TrueValueList = list()
TestInputList = list()
for key in range(len(TestWaitTimeSeq)):
    for i in range(1,len(TestWaitTimeSeq[key])):
        #TrueValueList.append(centroids[TestWaitTimeSeq[key][i]])
        TrueValueList.append(TestWaitTimeSeq[key][i])
        #TestInputList.append(TestWaitTimeSeq[key][:i])
        tempList = TestWaitTimeSeq[key][:i][:]
        tempList.reverse()
        tempList += [0]*(NumAttribute-len(TestWaitTimeSeq[key][:i]))
        tempArray = np.array(tempList)
        TestInputList.append(tempArray)
TestInputMatrix = np.vstack((TestInputList))
TestInputMatrix = sm.add_constant(TestInputMatrix)
TrueValueArray = np.array(TrueValueList)
PredictArray = np.dot(TestInputMatrix,rlm_results.params)
PredictArray = map(round,PredictArray)
PredictArray = map(int,PredictArray)
for i in range(len(PredictArray)):
    if (PredictArray[i]<0):
        PredictArray[i] = 0
    if (PredictArray[i]>max(WaitingTimeValue)):
        PredictArray[i] = max(WaitingTimeValue)
PredictArray = [centroids[i] for i in PredictArray]

#Calculate the mse
Residual = np.array(PredictArray)- np.array([centroids[i] for i in TrueValueArray])
MSE = np.dot(Residual,Residual)/rlm_results.df_resid

#Calculate the average error
AverageError = sum(map(abs,Residual))/len(Residual)
print "AverageError"
#averageerror = 0.1582 hour


##5.7272
##mse=0.0762 hour
