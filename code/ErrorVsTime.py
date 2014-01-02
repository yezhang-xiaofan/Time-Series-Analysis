# This file computes the average error against time using only time without weekday
#This file computes the average error over all offices in each bucket. 


import csv
import time
import re
from datetime import datetime
from pytz import timezone
import pytz
import xlwt
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
    Hash = str(item[4].hour).zfill(2)+str((item[4].minute)/10)
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

#Compute the average waiting time and average error for all offices in the set

TotalError = 0
Effe_Office = 0

#The number of office appearing in the same bucket in Training data and Testing data
Num_Same = 0

#Average Error for each bucket
#key is bucket
#value is a list
#0: total average error over all the offices
#1: the counts
#2:average error over all the offices
AverErr = {}

for office in OfficeId:
    
    #CountTime is a dictionary
    #key is item's hash value
    #value is a list
    #0:Total waiting time   1: total counts  2: average waiting time
    
    CountTime = dict()
    
    #TestTime is a dictionary used to store test data
    #key is has value
    #value is a list of corresponding waiting time
    TestTime = {} 
    
    for item in train_set:
        if ((item[1])==office):
            PutInMap(item,CountTime)
    for item in test_set:
        if((item[1])==office):
            PutInMap2(item,TestTime)
    
    if (ComKeyDic(CountTime,TestTime)==True):
        Num_Same = Num_Same + 1
        for key in CountTime:
            CountTime[key].append(CountTime[key][0]/CountTime[key][1])    #CountTime[key][2]=Total waiting time/total counts=prediction waiting time

    #Put all the waiting time coressponding to a ceartin hash value into TestTime 
                     
    
    #DifferMap is a dictionary to store the difference between the prediction and the true waiting time
    #key is hash value
    #value is the average error for this bucket
        DifferMap = {}

        for key in CountTime:
            a = map(float, TestTime[key])
            b = [CountTime[key][2]]*len(TestTime[key])
    
    
            DifferList = [a-b for a,b in zip(a,b)]         # Compute the difference between the prediction and each test data
            Average_Error = sum(map(abs,DifferList))/len(DifferList)
    
            #DifferMap[key] = list()
            #DifferMap[key] = Average_Error
            if(AverErr.has_key(key)):
                AverErr[key][0] = AverErr[key][0] + Average_Error
                AverErr[key][1] = AverErr[key][1] + 1
            else:
                AverErr[key] = list()
                AverErr[key].append(Average_Error)
                AverErr[key].append(1)

#Compute the average error value for this office
        
        '''
        SumError = 0.0
        count = 0
        for key in DifferMap:
            SumError = DifferMap[key]+SumError
            count = count + 1
    
        AverError = SumError/count
        TotalError = AverError + TotalError
        Effe_Office = Effe_Office + 1
        '''
'''
print Effe_Office,
print TotalError
TotalAverError = TotalError/Effe_Office
print TotalAverError
'''

for key in AverErr:
    AverErr[key].append(float(AverErr[key][0])/(AverErr[key][1]))

#compute the total average error
TotalError = 0.0
for key in AverErr:
    TotalError = TotalError + AverErr[key][2]

print TotalError/len(AverErr)

NewDifferMap = {}
DateList = list()
ErrorList = list()
NewDict = list()

for key in sorted(AverErr.iterkeys()):
    
    NewDict.append((key,AverErr[key][2]))

(Time,Error)=zip(*NewDict)
print Time



TimeList = list()
my_xticks = list()
DateList = list()


# Create Timeseries List---DateList
for item in Time:
    item = item[0] + item[1] + str(int(item[2])*10).zfill(2)
    TimeObject = time.strptime(item,'%H%M')
    TimeList.append(TimeObject)

for item in TimeList:
    DateObject = datetime(2013,1,1,*item[3:6])       #set 01/07/2013 as Monday
    DateList.append(DateObject)





'''
Date1 = DateList[:66]
Date2 = DateList[66:132]
Date3 = DateList[132:198]
Date4 = DateList[198:264]
Date5 = DateList[264:]

Error1 = Error[:66]
Error2 = Error[66:132]
Error3 = Error[132:198]
Error4 = Error[198:264]
Error5 = Error[264:]
'''

#plot the figure

EachHour = HourLocator()
TenMinutes = MinuteLocator(range(60), interval=10)
Fmt = DateFormatter("%H:%M")

fig = figure()
ax = fig.add_subplot(111)
#ax.set_title("Monday")
ax.plot_date(DateList,Error,'-')
#ax.set_ylabel("Average Error")
#ax.set_xlabel("Time")
ax.xaxis.set_major_locator(EachHour)
ax.xaxis.set_minor_locator(TenMinutes)
ax.xaxis.set_major_formatter(Fmt)
ax.autoscale_view()

'''
ax1 = fig.add_subplot(512)
ax1.plot_date(Date2,Error2,'-')
ax1.xaxis.set_major_locator(EachHour)
ax1.xaxis.set_minor_locator(TenMinutes)
ax1.xaxis.set_major_formatter(Fmt)
ax1.autoscale_view()
#ax1.set_title('Tuesday')
#ax1.set_ylabel('Average Error')

ax2 = fig.add_subplot(513)
ax2.plot_date(Date3,Error3,'-')
ax2.xaxis.set_major_locator(EachHour)
ax2.xaxis.set_minor_locator(TenMinutes)
ax2.xaxis.set_major_formatter(Fmt)
ax2.autoscale_view()
#ax2.set_title("Wednesday")
#ax2.set_ylabel('Average Error')

ax3 = fig.add_subplot(514)
ax3.plot_date(Date4,Error4,'-')
ax3.xaxis.set_major_locator(EachHour)
ax3.xaxis.set_minor_locator(TenMinutes)
ax3.xaxis.set_major_formatter(Fmt)
ax3.autoscale_view()
#ax3.set_title("Thursday")
#ax3.set_ylabel("Average Error")

ax4 = fig.add_subplot(515)
ax4.plot_date(Date5,Error5,'-')
#ax4.set_title("Friday")
#ax4.set_ylabel("Average Error")
ax4.xaxis.set_major_locator(EachHour)
ax4.xaxis.set_minor_locator(TenMinutes)
ax4.xaxis.set_major_formatter(Fmt)
ax4.autoscale_view()
'''

fig.autofmt_xdate()
show()




