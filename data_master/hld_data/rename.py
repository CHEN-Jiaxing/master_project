import os

path = os.getcwd()
print(path)

x = os.listdir(path)

for i in range(len(x)):
    if(x[i][0] is "h"):
        oldname = x[i]
        print(oldname)
        newname = oldname[0:3]+"_1"+oldname[3:]
        print(newname)
        
        old=path+ os.sep + oldname

        new=path + os.sep + newname
        print(old,"--->", new)
    
        os.rename(oldname,newname)
            
