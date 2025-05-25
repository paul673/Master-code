import os
#for root, dirs, files in os.walk("C:/Users/paulj/"):
    #print(root, dirs, files)

def bytes_to_gb(size_in_bytes):
    return size_in_bytes / (1024 ** 3)
summ=0
for dir in os.listdir("C:/Users/paulj/"):
    dir_sum=0
    for root, dirs, files in os.walk("C:/Users/paulj/"):
        for file in files:
            dir_sum += bytes_to_gb(os.path.getsize(os.path.join(root, file)))
    #size = bytes_to_gb(os.path.getsize(os.path.join("C:/Users/paulj/", dir)))
    print(dir,dir_sum)
    summ+=dir_sum
print("")
print(summ)