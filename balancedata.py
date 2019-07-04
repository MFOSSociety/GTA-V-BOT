import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('training_data_v2.npy', allow_pickle = True)

df = pd.DataFrame(train_data)
#print(df.head())
print(Counter(df[1].apply(str)))

#for data in train_data:
   # choice = data[1]
   # print(choice)



forwards = []
frontleft = []
frontright = []
nokey = []
#shuffle(train_data)
i = 0 
for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [0,1,0]:
        forwards.append([img,choice])
    elif choice == [0, 1, 1]:
        frontright.append([img, choice])
    elif choice == [1, 1, 0]:
        frontleft.append([img, choice])

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice ==[0, 0, 0]:
        nokey.append([img, choice])



    
forwards = forwards[:len(nokey)]
frontleft = frontleft[:len(nokey)]
frontright = frontright[:len(nokey)]

new_train = forwards + frontleft + frontright + nokey
shuffle(new_train)
np.save('training_data_v3.npy', new_train)
new_train  = np.load('training_data_v3.npy', allow_pickle= True)
i = 0
df = pd.DataFrame(new_train)
#print(df.head())
print(Counter(df[1].apply(str)))