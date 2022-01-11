import os

path = os.getcwd()
print(path)

x = os.listdir(path)

names = []

for i in range(len(x)):
    #print(x[i][6])
    if x[i][6] is "1" or x[i][6] is "2":
        oldname = x[i]
        print(oldname)
        names.append(oldname)
        
for i in range(len(names)):
    print("data_"+str(i)+ " = np.load(\'./input_va_master/"+names[i]+"\')")    
