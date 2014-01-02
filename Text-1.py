

#This file computes the average error in each bucket using Time+Week method

import csv
import time
import re
from datetime import datetime
from pytz import timezone
import pytz

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
from matplotlib.ticker import MaxNLocator




#read two files
with open('20130128_offices.csv','rb') as csvfile:
    offices = csv.reader(csvfile)        
    officelist = list()
    for row in offices:
        officelist.append(row)   
        
with open('20130128_waiting_times.csv') as csvfile:
    waiting_times = csv.reader(csvfile)
    waiting_timeslist = list()
    for row in waiting_times:
        waiting_timeslist.append(row)

waiting_timeslist.pop(0)


#wbk = xlwt.Workbook()
#sheet = wbk.add_sheet('sheet 1')
#excel_date_fmt = 'M/D/YY h:mm'
#style = xlwt.XFStyle()
#style.num_format_str = excel_date_fmt


number = len(waiting_timeslist)
    
#c = csv.writer(open("NewData.csv", "wb"))

#define a hashfunction to map the same bucket to a unique value
def Hashfunction(item):
    Hash = (str(item[4].weekday())+str(item[4].hour).zfill(2)+str((item[4].minute)/10))
    return Hash

#Define a PutInMap function : Put all the bucket to a map to compute the average waiting time
#key is bucket
#value is the waiting time
def PutInMap(item,Map):
    temp = item[6]
    if(temp not in Map):
        Map[temp] = list()
        Map[temp].append(float(item[3]))
        Map[temp].append(1)
    else:
        Map[temp][0] = Map[temp][0] + float(item[3])
        Map[temp][1] = Map[temp][1] + 1
    
# Compute waiting time for 548 DMV 
CountTime = dict()

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
    
    
    #print temp
    #new_time = waiting_timeslist[i][col].strftime("%Y-%m-%d %H:%M:%S")
    #new_weekday = waiting_timeslist[i][col].weekday()
    #c.writerow([waiting_timeslist[i][0],waiting_timeslist[i][1],waiting_timeslist[i][2],waiting_timeslist[i][3],new_time,new_weekday])

#train_set and test_set

train_set = waiting_timeslist[1:number/2]
test_set = waiting_timeslist[number/2:]

#Compute the average waiting time for id=548 in the train_set

TestTime = {}


IdSet = set()
for item in train_set:
    i = int(item[1])
    IdSet.add(i)


#for j in IdSet: 
for item in train_set:
    if (int(item[1])==548):
        PutInMap(item,CountTime)

for key in CountTime:
    CountTime[key].append(CountTime[key][0]/CountTime[key][1])

#Compute the average error for each bucket for 548 in the test_set

def PutInMap2(item,Map):
    temp = item[6]
    if(temp not in Map):
        
        Map[temp] = list()
        Map[temp].append(item[3])
    else:
        Map[temp].append(item[3])
    
for item in test_set:
    if(int(item[1])==548):
        PutInMap2(item,TestTime)

DifferMap = {}

for key in CountTime:
    a = map(float, TestTime[key])
    b = [CountTime[key][2]]*len(TestTime[key])
    
    
    DifferList = [a-b for a,b in zip(a,b)]
    Average_Error = sum(map(abs,DifferList))/len(DifferList)
    
    DifferMap[key] = list()
    #DifferMap[key].append(CountTime[key][2])
    DifferMap[key].append(Average_Error)

#print DifferMap.keys()

NewDifferMap = {}

#for key in DifferMap:
    
    
DateList = list()
ErrorList = list()
'''
for key, error in DifferMap.iteritems():
    
    data_string = key
    data_object = datetime.strptime(data_string, '%w%H%M')
    
show()
'''
NewDict = list()

for key in sorted(DifferMap.iterkeys()):
    
    NewDict.append((key,DifferMap[key]))

(Time,Error)=zip(*NewDict)
print Time
TimeList = list()
my_xticks = list()
DateList = list()

for item in Time:
    item = str(int(item[0])+1) + item[1] + item[2] + str(int(item[3])*10).zfill(2)
    TimeObject = time.strptime(item,'%w%H%M')
    TimeList.append(TimeObject)

for item in TimeList:
    DateObject = datetime(2013,1,item[6]+7,*item[3:6])       #set 01/07/2013 as Monday
    DateList.append(DateObject)



def IntToWeek(i):
    if i==0:
        return 'Monday'
    elif i==1:
        return 'Tuesday'
    elif i==2:
        return 'Wednesday'
    elif i==3:
        return 'Thursday'
    else:
        return 'Friday'



#divide into five parts

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

EachHour = HourLocator()
TenMinutes = MinuteLocator(range(60), interval=10)
Fmt = DateFormatter("%H:%M")

fig = figure()
ax = fig.add_subplot(511)
#ax.set_title("Monday")
ax.plot_date(Date1,Error1,'-')
#ax.set_ylabel("Average Error")
#ax.set_xlabel("Time")
ax.xaxis.set_major_locator(EachHour)
ax.xaxis.set_minor_locator(TenMinutes)
ax.xaxis.set_major_formatter(Fmt)
ax.autoscale_view()

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



fig.autofmt_xdate()
#ax1.xaxis.set_major_formatter(my_xticks1)
show()
'''
mp.subplot(5,1,2)
#mp.title("Tuesday")
mp.xticks(Aix1,my_xticks2)
mp.plot(Aix1, Error2)


mp.subplot(5,1,3)
#mp.title("Wednesday")
mp.xticks(Aix1,my_xticks3)
mp.plot(Aix1,Error3)

mp.subplot(5,1,4)
#mp.title("Thursday")
mp.xticks(Aix1,my_xticks4)
mp.plot(Aix1,Error4)

mp.subplot(5,1,5)
#mp.title("Friday")
mp.xticks(Aix1,my_xticks5)
mp.plot(Aix1,Error5)

#graph = fig.add_subplot(111)
#plt.xticks(Aix , my_xticks)

#plt.plot(Aix,Error,'bo')
mp.show()
'''



