#This file computes 3orderMarkov reverse model
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
        Current = (float(Current)+1)/(SumRow+len(WaitingTimeValue))
        PredictWaiting.set_value(row,column,Current)

writer = ExcelWriter('BaseLine.xls')
#PredictWaiting.to_excel(writer,sheet_name='sheet1')
data = zeros(NumTime,'d')
BasePre = Series(data,index=TimeValueString)
for row in TimeValueString:
    MaxValue = (PredictWaiting.ix[row]).argmax()
    BasePre[row] = MaxValue
BasePre.to_csv('BaseLine')

#Calculate the errors using P(wt|h)
#discard the weekday in the testing data
TestTimeList = [(x[1]+x[2]+x[3]) for x in TestTimeList]
#TestTimeList = [int(x) for x in TestTimeList]
totalCount = 0
totalError = 0
for i in range(len(TestTimeList)):
    totalCount = totalCount + 1
    totalError = totalError + abs(centroids[TestWaitTimeList[i]]-centroids[int(BasePre[TestTimeList[i]])])     
errorRate = float(totalError)/totalCount
#errorRate is 0.5235 hour

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

#Construct the transition and emission matrix

Transition = np.zeros((NumWaitingTime,NumWaitingTime),'d')
Transition = DataFrame(Transition)
index = WaitingTimeValue
columns = WaitingTimeValue
Transition = DataFrame(Transition,index=index, columns=columns)
Count = Series(np.zeros(NumWaitingTime), index=WaitingTimeValue)
for key in WaitingTimeSeq:
    print 'This is the {index}st sequence: '.format(index = key+1) + repr(WaitingTimeSeq[key])
    for i in xrange(1,len(WaitingTimeSeq[key])):
        Current = Transition.get_value(WaitingTimeSeq[key][i],WaitingTimeSeq[key][i-1])
        Current = Current + 1
        Count[WaitingTimeSeq[key][i]] = Count[WaitingTimeSeq[key][i]] + 1
        Transition.set_value(WaitingTimeSeq[key][i],WaitingTimeSeq[key][i-1],Current)

#Decrease the number of transition from 0 to 0

#getcontext().prec = 3

CurrentCount = Transition.get_value(Index0,Index0)
##########################################################################  change the decreasing rate here 
CurrentCount = float(CurrentCount) * 1 
Transition.set_value(Index0,Index0,CurrentCount)
for row in WaitingTimeValue:
    SumRow = sum(Transition.ix[row])
    for column in WaitingTimeValue:
        Current = Transition.get_value(row,column)
        Current = Decimal((Current+1))/Decimal((SumRow+NumWaitingTime))
        Transition.set_value(row,column,Current)
'''       
predict the value of w4, the problem is to output
argmax_w4 P(w1 w2 w3 w4)
P(w1 w2 w3 w4) can be rewritten as P(w4 w3 w2 w1) in
the reversed model and apply chain rule to decompose P(w4 w3 w2 w1) to P(w4)P(w3|w4)P(w2|w4 w3)P(w1|w2w3w4)
P(w4)P(w3|w4) is the first-order Markov order you already trained, adding P(w2|w4 w3)P(w1|w2w3w4) will consider longer
history from the data.
Using SRILM toolkit, it is easy to estimate the
above probability
if you reverse the waiting time sequence to w_n w_n-1,..., w3, w2, w1 and train an n-gram model from the data where each
line/sentence is one day's data. Once the model is trained, it will estimate the probability (with backoff)
 which can then be used to estimate the probability of P(w4 w3 w2 w1)
'''

##############construct the reverse sequence
f = open('reverseWaitingTimeSeq','w')
for key in TimeSeq:
    index = range(len(TimeSeq[key]))
    index.reverse()
    for i in index:
        f.write(str(WaitingTimeSeq[key][i])+" ")
    f.write('\n')

ConditionalMap = dict()
BackWeightMap = dict()
f1 = open('Reserse4Gram','r')
for line in f1:
    tempString = line.split('\t')
    if(len(tempString)>=2):
        Ngram = tempString[1].rstrip('\n')
        ConditionalMap[Ngram] = 10**float(tempString[0].rstrip('\n'))
        if(len(tempString)>=3):
            BackWeightMap[Ngram] = 10**float(tempString[2])

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

totalError = 0.0
totalCount = 0
for key in TestTimeSeq:
    for i in range(3,len(TestTimeSeq[key])):
        totalCount = totalCount + 1
        Prediction = dict()
        for j in WaitingTimeValue:
            Pj = PredictWaiting.get_value(TestTimeSeq[key][i],j)
            ##calculate P(w3|w4)
            P34 = Transition.get_value(j,TestWaitTimeSeq[key][i-1])
            ##calculate P(w2|w4 w3)
            tempString = str(j) + " " + str(TestWaitTimeSeq[key][i-1]) + " " + str(TestWaitTimeSeq[key][i-2])
            P243 = 1
            alpha = 1
            while(ConditionalMap.has_key(tempString)==False):
                tempString1 = " ".join(tempString.split(" ")[1:])
                tempString2 = " ".join(tempString.split(" ")[:-1])
                if(ConditionalMap.has_key(tempString2)):
                    alpha = alpha * BackWeightMap[tempString2]
                else:
                    alpha = 1 * alpha
                tempString = tempString1
            P243 = alpha * ConditionalMap[tempString]           
            
            tempString = str(j) + " " + str(TestWaitTimeSeq[key][i])+" "+str(TestWaitTimeSeq[key][i-1])+" "+str(TestWaitTimeSeq[key][i-2])
            prob = 1
            alpha = 1
            while(ConditionalMap.has_key(tempString)==False):
                tempString1 = " ".join(tempString.split(" ")[1:])
                tempString2 = " ".join(tempString.split(" ")[:-1])
                if(ConditionalMap.has_key(tempString2)):
                    alpha = alpha * BackWeightMap[tempString2]
                else:
                    alpha = 1 * alpha
                tempString = tempString1
            prob = alpha * ConditionalMap[tempString]
            finalprob = prob * P243 * P34 * Pj
            Prediction[j] = finalprob                
        predictWaiting = max(Prediction.iteritems(), key=operator.itemgetter(1))[0]
        totalError = totalError + abs(centroids[predictWaiting]-centroids[TestWaitTimeSeq[key][i]])

print str(totalError/totalCount)
#0.1698 hour







