


# gola is to delte these files conainign resized key word 



#'/data1/data_alex/lsun/lsun_code/exported_dining/0/0/0/0/2/5/00002564ea7bb46b87e6d99ee0a0251283a1d01a_resized_256_resized_256_resized_128_resized_128.webp'

import glob

import os

count =0
for filename in glob.iglob("/data1/data_alex/lsun/lsun_code/exported_dining" + '**/**', recursive=True):
     
    if ("resized" in filename):
         print(filename)
         os.remove(filename)
    else:
        count+=1

print(count)