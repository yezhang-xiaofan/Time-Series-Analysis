fruit_available = {'apples': 25, 'oranges': 0, 'kiwi': 12}

my_satchel = {'apples': 1, 'oranges': 0, 'kiwi': 13, 'abc':15 }

if any(True for k in fruit_available if k not in my_satchel):
    
    print "False"
else:
    print "True"

'''
available = set(fruit_available.keys())
satchel = set(my_satchel.keys())
'''
# fruit not in your satchel, but that is available
#print (available.difference(satchel) != null)