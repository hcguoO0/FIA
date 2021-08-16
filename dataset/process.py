from scipy import misc
import os
import numpy as np
import pandas as pd

images=os.listdir('./images/')
k=1
for image in images:
    os.rename('./images/'+image,'./images/'+'{0}.png'.format(k))
    k=k+1


# f = open('./labels.txt', 'w+')
# df = pd.read_csv('dev_dataset.csv',usecols=[0, 6]) # read ImageId and TrueLabel
# imagesid=df['ImageId'][:]
# labels=df['TrueLabel'][:]
# x=zip(imagesid,labels)
# x=sorted(x)
# for i in range(1000):
#     f.write(str(x[i][1])+'\n')
# f.close()



