import csv
import time
import re
from datetime import datetime
from pytz import timezone
import pytz
import xlwt
import pdb

#read two files
with open('20130128_offices.csv','rb') as csvfile:
    offices = csv.reader(csvfile)        
    officelist = list()
    for row in offices:
        officelist.append(row)   
        
with open('20130128_waiting_times.csv') as csvfile:
    waiting_times = csv.reader(csvfile)
    #for row in data:
       # print row
       
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
    
for key in DifferMap:
    print DifferMap
    
    






