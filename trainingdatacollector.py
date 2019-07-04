import numpy as np
import cv2
from mss import mss
import time
from getkeys import key_check
import os

sct = mss()
monitor = {"top": 40, "left": 0, "width": 800, "height": 600}

def keys_to_output(keys): # one hot array me convert karna zaruri hai
    output = [0,0,0]

    if 'A' in keys:
        output[0] = 1
    elif 'W' in keys:
        output[1] = 1
    elif 'D' in keys:
        output[2] = 1

    return output

file_name = 'training_data_v4.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name, allow_pickle= True))
else:
    print('File does not exist, starting fresh!')
    training_data = []

def root():                               # where the main goods are
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    paused = False
    while 1:
        if not paused:
            last_time = time.time()
            screen = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (80,60))
            keys = key_check()
            output= keys_to_output(keys)
            training_data.append([screen, output])
            print(f"fps: {1 / (time.time() - last_time)} : training data length : {len(training_data)} \n keypress : {output}")
            if len(training_data) % 500 ==0:
                print(len(training_data))
                np.save(file_name, training_data)

            if 'T' in keys:
                if paused:
                    paused = False
                    time.sleep(1)
                else:
                    paused = True 
        else:
            q = input('start? y/n')
            if q =='y':
                paused = False
            else:
                paused = True

           


  

root()