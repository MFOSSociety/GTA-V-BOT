# GTA-V-bot

**This will work with _other games_ too**

Made by following tutorials by [Sentdex](https://pythonprogramming.net/)

# Requirements :




- Tensorflow-gpu (_pip install tensorflow-gpu_) 
(_tensorflow for cpu will work, but will take a **much** longer time, possibly days, to fit the model_)
- CUDA and CuDNN that are compatible with each other and the tensorflow version.
- mss for screen-cap (pip install mss)
- numpy (pip install numpy)
- opencv (pip install opencv-python)
  
  (Save yourself some time and do ``pip install -r requirements.txt``)



# How to use : The Self-driving car

Change the game resolution to 800x600. If you want to use a custom resolution, edit *monitor* dict in _trainingdatacollector.py_.


Position the game window in the top left corner, make sure no other window is overlapping with the game window.


Start _trainingdatacollector.py_ and play the game yourself in the manner you want the Neural Net to learn.



**If you want to register extra keys while playing, edit the _keys_to_output_ function. Change the size of the one hot array and add more statements accordingly**



(_consider automating the OHE conversion process if you have too many keys_)

After collecting the training data, use _balancedata.py_ to balance the data. You may need to edit _balancedata.py_ if you have additional keys.



**Anything less than 50k samples, post-balancing, will yield poor results. Try to reach at least a 100k samples, post-balancing**

Use _modelfit.py_ to fit everything, make sure _alexnet.py_ is in the same directory. (_you may need to edit alexnet.py if you have additional keys_)



After saving the model, start up the game again, position it correctly and use _testingfile.py_

If all goes well then the ingame character should move according to the predicted moves made by the model you made.

---

# How to use : theRidiculouslySlowMurderBot

clone https://github.com/tensorflow/models and move the .py file to models\research\object_detection

The bot is ridiculously slow, gives a shitty frame-rate and heats up your system. What more could you ask for? :) 

(_will update the bot to be faster and more accurate in the future_)


