import numpy as np
import cv2
from mss import mss
import time
from alexnet import alexnet
from directkeys import PressKey,ReleaseKey, W, A, S, D
from getkeys import key_check


WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',EPOCHS)

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def frontleft():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)

def frontright():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
def nokey():
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(W)

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)
sct = mss()
monitor = {"top": 40, "left": 0, "width": 800, "height": 600}

def root():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    paused = False
    while 1:
        if not paused:
            keys = key_check()
            last_time = time.time()
            screen = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (80, 60))

            moves = list(np.around(model.predict([screen.reshape(80, 60, 1)])[0]))
            print(moves)

            if moves == [1, 1, 0] or moves== [1,0,0]:
                #print('wa')
                frontleft()
            elif moves == [0, 1, 0]:
                #print('a')
                straight()
            elif moves == [0, 1, 1] or moves== [0,0,1]:
                #print('wd')
                frontright()
            elif moves == [0, 0, 0]:
                #print('0')
                nokey()
            if 'T' in keys:
                if paused:
                    paused = False
                    time.sleep(1)
                else:
                    paused = True

        #print(f"fps: {1 / (time.time() - last_time)}")
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            q = input('start? y/n')
            if q =='y':
                for i in list(range(4))[::-1]:
                    print('starting in :', i + 1)
                    time.sleep(1)
                    paused = False
            elif q=='n':
                break
root()

