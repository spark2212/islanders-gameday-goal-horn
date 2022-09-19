import pyaudio
import librosa
import numpy as np
import time
import subprocess
import os
import sys
import tensorflow.keras as keras
import json
import kbHitMod
import math
import scipy.io.wavfile

MODEL_PATH = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Model_2.model"

OT_GOAL_TRACK = "1 New York Islanders Overtime Goal and Win Horn || NYCB Live: Home of the Nassau Veterans Memorial Coliseum"
WIN_TRACK = "2 New York Islanders Win Horn || NYCB Live: Home of the Nassau Veterans Memorial Coliseum"
GOAL_TRACK = "3 New York Islanders Goal Horn || NYCB Live Home of the Nassau Veterans Memorial Coliseum"
QUIET_TRACK = "4 pure silence"

STOP_COMMAND = "osascript -e 'tell application \"iTunes\" to stop'"

WIN = 0
NO_GOAL = 1
GOAL = 2

model = keras.models.load_model(MODEL_PATH, compile=True)


class Counter:
    def __init__(self, start_val):
        self.value = start_val
        self.v0 = start_val
    
    def add(self):
        self.value += 1
    
    def reset(self, new_val=None):
        if new_val is None: 
            new_val = self.v0

        self.value = new_val
        self.v0 = new_val
    
    def get(self):
        return self.value


class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self,size_max):
        self.max = size_max
        self.data = []

    class __Full:
        """ class that implements a full buffer """
        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data[self.cur] = x
            self.cur = (self.cur+1) % self.max
        def get(self):
            """ return list of elements in correct order """
            return self.data[self.cur:]+self.data[:self.cur]


    def append(self,x):
        """append an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.max:
            self.cur = 0
            # Permanently change self's class from non-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data

class ShutdownTimer:
    """Class that defines circumstances that trigger the end of the program"""
    def __init__(self):
        self.count = False
        self.start = time.time()
        self.target = -1000
        self.cache = -1000
        self.isPaused = False
        self.runtime = 0
        self.calltime = self.start
    
    def startTimer(self, t1):
        self.start = time.time()
        self.target = t1
        self.count = True
        self.cache = self.target
        self.runtime = 0
    
    def pauseTimer(self):
        if self.count == True:
            self.count = False
            self.isPaused = True
            self.cache = self.start + self.target - time.time()
            self.runtime += self.target - self.cache

    def resumeTimer(self):
        if self.isPaused == True:
            self.count = True
            self.isPaused = False
            self.target = self.cache
            self.start = time.time()

    def getIsPaused(self):
        return self.isPaused

    def getTarget(self):
        return self.target

    def getTimeLeft(self):
        if self.count == True:
            return self.start + self.target - time.time()
        elif self.isPaused == True:
            return self.cache
        else:
            return 1000

    def getTimeElapsed(self):
        if self.count == False and self.isPaused == False:
            self.runtime = 0
        elif self.count == True:
            self.runtime += time.time() - self.calltime
        
        self.calltime = time.time()
        return self.runtime
    
    def getIsRunning(self):
        return self.count
    
    def stopTimer(self):
        self.count = False
        self.target = -1000
        self.cache = -1000
        self.start = time.time()
        self.runtime = 0

# ring buffer will keep the last 1 second worth of audio
ringBuffer = RingBuffer(1*22050)
ringBuffer2 = RingBuffer(21)
cancelBuffer = RingBuffer(512)

pauseTimer = ShutdownTimer()
shutdownTimer = ShutdownTimer()
saveTimer = ShutdownTimer()
logTimer = ShutdownTimer()
confirmTimer = ShutdownTimer()
helpTimer = ShutdownTimer()
freqTimer = ShutdownTimer()

INTERVAL = 600
MIN_THRESH = -14
MAX_THRESH = -8
THRESHOLD = MIN_THRESH

overtime = False
shootout = False
quit_ = False
demo = False

mode_names = ["WIN", "NO_GOAL", "GOAL"]

n = Counter(1) # logTimer
i = Counter(1) # Win
i2 = Counter(1) # Actually No Win
i3 = Counter(1) # Actually Win
j = Counter(1) # No goal
k = Counter(1) # Goal
k2 = Counter(1) # Actually No Goal
k3 = Counter(1) # Actually Goal
w = Counter(1) # TBD label 1
w2 = Counter(1) # TBD label 2
h = Counter(1) # False Negative
h2 = Counter(1) # True Positive

mode = NO_GOAL
noise_cancel = False

print("\nOvertime mode: off\nShootout mode: off\nDemo mode: off")
    
def getState():
    return subprocess.getoutput("osascript -e 'tell application \"iTunes\" to player state as string'") 

def getTrack():
    return subprocess.getoutput("osascript -e 'tell application \"iTunes\" to name of current track as string'")

def play(track_name):
    subprocess.getoutput(STOP_COMMAND)
    subprocess.getoutput("osascript -e 'tell application \"iTunes\" to play (first track of playlist \"Library\" whose name is \"" + track_name + "\")'")

def stop():
    subprocess.getoutput(STOP_COMMAND)

def callback(in_data, frame_count, time_info, flag):
        
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    audio_data = np.nan_to_num(audio_data)

    if mode != NO_GOAL:
        audio_data = np.zeros(audio_data.shape, dtype=np.float32)
    
    audio_data = librosa.resample(audio_data, 44100, 22050)
    
    for i in audio_data:
        ringBuffer.append(i)
            
    return (in_data, pyaudio.paContinue)

if __name__ == "__main__":

    pa = pyaudio.PyAudio()
    stream = pa.open(format = pyaudio.paFloat32,
                    channels = 1,
                    rate = 44100,
                    output = False,
                    input = True,
                    stream_callback=callback)

    # start the stream
    stream.start_stream()

    kb = kbHitMod.KBHit()

    print("Stream open!")

    need2save = False
    limit = 0

    Y = np.empty((44,44), dtype=np.float32)
    Z = np.empty((44,44), dtype=np.float32)

    while stream.is_active():
        JSON_WIN_PATH = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/json_datasets/WINS/win_" + str(i.get()) + ".json"
        ACTUALLY_WIN = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/json_datasets/ACTUALLY_WINS/this_IS_a_win_" + str(i3.get()) + ".json"
        ACTUALLY_NO_PATH = [
            "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/json_datasets/ACTUALLY_NOS/actually_no_win_" + str(i2.get()) + ".json", 
            "placeholder", 
            "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/json_datasets/ACTUALLY_NOS/actually_no_win_" + str(i2.get()) + ".json"
            ]
        JSON_NO_GOAL_PATH = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/json_datasets/NO_GOALS/no_goal_" + str(j.get()) + ".json"
        JSON_GOAL_PATH = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/json_datasets/GOALS/goal_" + str(k.get()) + ".json"
        ACTUALLY_GOAL = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/json_datasets/ACTUALLY_GOALS/this_IS_a_goal_" + str(k3.get()) + ".json"
        TBD_FILE = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/TBD_FILES/what_is_this_" + str(w2.get()) + "-" + str(w.get())
        FALSE_NEGATIVE = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/TBD_FILES/probable_false_negative_"
        TRUE_POSITIVE = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/TBD_FILES/just_before_the_trigger_"

        command = kb.kbhit()
        kb.off()
        keypressed = True
        
        if command == "o":
            shootout = False
            if overtime == True:
                overtime = False
                print("Overtime mode: off\nShootout mode: off\n")
            else:
                overtime = True
                print("Overtime mode: ON\nShootout mode: off\n")
        elif command == "s":
            overtime = False
            if shootout == True:
                shootout = False
                print("Overtime mode: off\nShootout mode: off\n")
            else:
                shootout = True
                print("Overtime mode: off\nShootout mode: ON\n")
        elif command == "n":
            noise_cancel = not noise_cancel
        elif command == "q":
            quit_ = True
            stop()
            print("Quitting... goodbye!\n")
        elif command == "d":
            demo = not demo
            if demo == True:
                print("Demo mode: ON")
            else:
                print("Demo mode: off")
        elif command == "g" or command == "w":
            print("Starting audible countdown...")
            play("5-second countdown")
            print("Get ready...")

            t = time.time()
            while getState() == "playing" and time.time() -t < 5.282: 1
                
            t0 = time.time()   
            print("GO!")
            while time.time() - t0 <= 1: 1
            
            new_signal = ringBuffer.get()
            new_signal = np.array(new_signal, np.float32)
            Y = librosa.feature.mfcc(new_signal, sr=22050, n_mfcc=44, hop_length=512, n_fft=2048)
            Y = Y.T
            path = " "
            if command == "g":
                path = ACTUALLY_GOAL
                k3.add()
            else: 
                path = ACTUALLY_WIN
                i3.add()

            print("\nDone! Saving to {}...".format(path))
            with open(path, "w") as fp:
                json.dump(Y.tolist(), fp, indent = 4)

            print("data saved to {}\n".format(path))
        
        elif command == "l":
            print("Final orders:")
            final_orders = input()
            if final_orders == "play taps" or final_orders == "Play taps":
                print("Playing \"Taps\"..."); play("Islanders Taps")

                while getState() == "playing": 1
                quit_ = True; print("(sigh)")

                t = time.time()
                t_diff = math.ceil( - time.time() + t)

                while time.time() - t < 2: 1
                break
            else:
                print("Never mind! We haven't lost yet!")
                continue
        elif command == " ":
            print("prediction == {}".format(prediction))  
        else:
            keypressed = False

        if pauseTimer.getIsRunning():
            if confirmTimer.getIsRunning() and demo == False:
                print("no_goal confirmation attempt failed...")
                logTimer.startTimer(0.9)
                logTimer.pauseTimer()
                confirmTimer.stopTimer()
                n.reset(0)
            if logTimer.getIsRunning() and logTimer.getTimeLeft() <= 1 and demo == False:
                logTimer.startTimer(10)
                logTimer.pauseTimer()

            if getState() == "paused":
                print("Never mind!")
                # Horn interrupted quickly, means false positive
                if pauseTimer.getTimeElapsed() <= 5 and need2save == True and (shootout == False or pauseTimer.getTimeElapsed() <= 3):
                    if mode == WIN:
                        i2.add()
                    elif mode == GOAL:
                        k2.add()

                    with open(ACTUALLY_NO_PATH[mode], "w") as fp:
                        json.dump(Y.tolist(), fp, indent = 4)

                    print("data saved to {}\n".format(ACTUALLY_NO_PATH[mode]))
                elif shutdownTimer.getIsRunning() == True and (shutdownTimer.getTimeElapsed() >= 40 or shootout) and need2save == True:
                    if mode == WIN:
                        with open(JSON_WIN_PATH, "w") as fp:
                            json.dump(Y.tolist(), fp, indent = 4)

                        print("data saved to {}".format(JSON_WIN_PATH))
                        i.add()
                    elif mode == GOAL:
                        with open(JSON_GOAL_PATH, "w") as fp:
                            json.dump(Y.tolist(), fp, indent = 4)

                        print("data saved to {}".format(JSON_GOAL_PATH))
                        k.add()
                
                need2save = False
                mode = NO_GOAL
                pauseTimer.stopTimer()
                shutdownTimer.stopTimer()
                stop()
            # Absolutely no interruptions
            elif pauseTimer.getTimeLeft() <= 0:
                print("Back to the game! \nlogTimer at T-{}".format(math.ceil(logTimer.getTimeLeft())))
                pauseTimer.stopTimer()
                stop()

                if mode == GOAL and need2save == True:
                    with open(JSON_GOAL_PATH, "w") as fp:
                        json.dump(Y.tolist(), fp, indent = 4)

                    print("data saved to {}".format(JSON_GOAL_PATH))
                    k.add()
                
                mode = NO_GOAL
                need2save = False
                logTimer.resumeTimer()
            elif shutdownTimer.getIsRunning() and shutdownTimer.getTimeLeft() <= 0:
                if need2save == True:
                    path = [JSON_WIN_PATH, JSON_NO_GOAL_PATH, JSON_GOAL_PATH]
                    with open(path[mode], "w") as fp:
                        json.dump(Y.tolist(), fp, indent = 4)
                    print("data saved to {}".format(path[mode]))

                quit_ = True     
                print("Game over! Goodbye!\n")

        if quit_ == True:
            break
        elif mode != NO_GOAL:
            freqTimer.pauseTimer()
        elif getState() == "playing" and mode == NO_GOAL:
            print("Oh, did I miss something?")
            need2save = False
            smoothTimer = ShutdownTimer()
            smoothTimer.startTimer(5)
            while getState() == "playing" and mode == NO_GOAL:
                if smoothTimer.getTimeLeft() <= 2:
                    track = getTrack()
                    if (track == GOAL_TRACK or track == OT_GOAL_TRACK) and smoothTimer.getTimeLeft() <= 0:
                        mode = GOAL
                        if overtime == False:    
                            pauseTimer.startTimer(42.353 - smoothTimer.getTimeElapsed())
                        elif overtime == True:
                            pauseTimer.startTimer(108 - smoothTimer.getTimeElapsed())
                            shutdownTimer.startTimer(105 - smoothTimer.getTimeElapsed())
                    elif track == WIN_TRACK or shootout:
                        state = getState()
                        if shootout == True and state == "paused" and smoothTimer.getTimeLeft() > 0:
                            mode = GOAL
                        elif smoothTimer.getTimeLeft() <= 0:
                            mode = WIN

                        if mode != NO_GOAL:
                            shutdownTimer.startTimer(75 - smoothTimer.getTimeElapsed())
                            pauseTimer.startTimer(79.44 - smoothTimer.getTimeElapsed())

            if mode != NO_GOAL:
                print("I guess I did!")
                data = ringBuffer2.get()
                for d in data:
                    scipy.io.wavfile.write(FALSE_NEGATIVE + str(h.get()) + ".wav", 22050, d)
                    
                    d = librosa.feature.mfcc(y=d, sr=22050, n_mfcc=44, hop_length=512, n_fft=2048)
                    d = d.T
                    with open(FALSE_NEGATIVE + str(h.get()) + ".json", "w") as fp:
                        json.dump(d.tolist(), fp, indent=4)
                    
                    print("Data saved to \n\n{} \n\nand \n\n{}\n".format(FALSE_NEGATIVE + str(h.get()) + ".wav", FALSE_NEGATIVE + str(h.get()) + ".json"))
                    
                    h.add()

                print("Data saved for manual sorting! \nmode == {}".format(mode_names[mode]))
        elif mode == NO_GOAL and keypressed == False:        
            signal = ringBuffer.get()
            signal = np.array(signal, np.float32)

            if signal.shape[0] == 22050: 

                X = librosa.feature.mfcc(signal, sr=22050, n_mfcc=44, hop_length=512, n_fft=2048)
                X = X.T

                X_new = X[np.newaxis, ...]
                prediction = model(X_new) 

                ringBuffer2.append(signal)
                    
                if np.argmax(prediction) == GOAL:
                    # GOAL!! 
                    state = getState()
                    track_name = getTrack()
                    Y = X
                    mode = GOAL
                    print("prediction == {}".format(prediction))

                    if state == "paused" or ((track_name != GOAL_TRACK and track_name != WIN_TRACK and track_name != OT_GOAL_TRACK) or track_name == QUIET_TRACK):
                        if overtime:
                            play(OT_GOAL_TRACK)
                            shutdownTimer.startTimer(105)
                            pauseTimer.startTimer(108)
                            print("Goal! We win!")
                            need2Save = not demo
                        elif shootout:
                            play(WIN_TRACK)
                            print("Shootout goal! We win?")
                            shutdownTimer.startTimer(75)
                            pauseTimer.startTimer(79.44)
                            need2Save = not demo
                        else:
                            play(GOAL_TRACK)
                            print("Goal!")
                            pauseTimer.startTimer(42.353)
                            need2save = not demo

                        print("\n")
                    
                    data = ringBuffer2.get()
                    for d in data[-3: ]:
                        scipy.io.wavfile.write(TRUE_POSITIVE + str(h2.get()) + ".wav", 22050, d)
                    
                        d = librosa.feature.mfcc(y=d, sr=22050, n_mfcc=44, hop_length=512, n_fft=2048)
                        d = d.T
                        with open(TRUE_POSITIVE + str(h2.get()) + ".json", "w") as fp:
                            json.dump(d.tolist(), fp, indent=4)
                        
                        print("Data saved to \n\n{} \n\nand \n\n{}\n".format(TRUE_POSITIVE + str(h2.get()) + ".wav", TRUE_POSITIVE + str(h2.get()) + ".json"))
                        
                        h2.add()

                        
                # decides if the last 1 second of audio contains a win
                elif np.argmax(prediction) == WIN:
                    state = getState()
                    track_name = getTrack()
                    Y = X
                    mode = WIN
                    print("prediction == {}".format(prediction))

                    if state == "paused" or ((track_name != GOAL_TRACK and track_name != WIN_TRACK and track_name != OT_GOAL_TRACK) or track_name == QUIET_TRACK):
                        play(WIN_TRACK)
                        shutdownTimer.startTimer(75)
                        pauseTimer.startTimer(79.44)
                        need2Save = not demo
            
                    print("We win!\n")

                    data = ringBuffer2.get()
                    for d in data[-3: ]:
                        scipy.io.wavfile.write(TRUE_POSITIVE + str(h2.get()) + ".wav", 22050, d)
                    
                        d = librosa.feature.mfcc(y=d, sr=22050, n_mfcc=44, hop_length=512, n_fft=2048)
                        d = d.T
                        with open(TRUE_POSITIVE + str(h2.get()) + ".json", "w") as fp:
                            json.dump(d.tolist(), fp, indent=4)
                        
                        print("Data saved to \n\n{} \n\nand \n\n{}\n".format(TRUE_POSITIVE + str(h2.get()) + ".wav", TRUE_POSITIVE + str(h2.get()) + ".json"))
                        
                        h2.add()
                # saves potential false negatives as both a json and as a wav file for later inspection
                elif prediction[0][GOAL] >= 1.0 * (10 ** THRESHOLD) or prediction[0][WIN] >= 1.0 * (10 ** (THRESHOLD)):
                    A = ringBuffer.get()
                    A = np.array(A, dtype=np.float32)

                    if prediction[0][GOAL] >= 10 ** (-5) or prediction[0][WIN] >= 10 ** (-5):
                        limit += 1
                    elif helpTimer.getTimeLeft() > 0 and helpTimer.getIsRunning() == True:
                        continue
                    else:
                        print("prediction == {}".format(prediction))

                        scipy.io.wavfile.write(TBD_FILE + ".wav", 22050, A)
                        with open(TBD_FILE + ".json", "w") as fp:
                            json.dump(X.tolist(), fp, indent=4)

                        print("Data needing classification saved to \n\n{}.wav and\n{}.json".format(TBD_FILE, TBD_FILE))
                        w.add()

                        helpTimer.startTimer(0.5)

                    if freqTimer.getIsRunning() == False and freqTimer.getIsPaused() == False:
                        freqTimer.startTimer(10)
                    elif freqTimer.getTimeLeft() <= 0:
                        freqTimer.startTimer(10)
                        if w.get() > 1:
                            w.reset(); w2.add()
                        else:
                            THRESHOLD -= 0.5
                        if THRESHOLD < MIN_THRESH:
                            THRESHOLD = MIN_THRESH
                            print("Minimum threshold reached!")
                        else:
                            print("Lowering threshold to 10^({})...".format(THRESHOLD))
                        limit = 0
                    elif w.get() > 3 + limit:
                        freqTimer.startTimer(10)
                        w.reset(); w2.add(); 
                        THRESHOLD += 1
                        if THRESHOLD > MAX_THRESH:
                            THRESHOLD = MAX_THRESH
                            print("Maximum threshold reached!")
                        else:
                            print("Increasing threshold to 10^({})...".format(THRESHOLD))
                        limit = 0
                

                elif np.argmax(prediction) == NO_GOAL and pauseTimer.getIsRunning() == False:
                    if confirmTimer.getIsRunning(): 
                        if confirmTimer.getTimeLeft() <= 0:
                            with open(JSON_NO_GOAL_PATH, "w") as fp:
                                json.dump(Z.tolist(), fp, indent = 4)

                                print("Last ten seconds had no goals. Data saved to {}\n".format(JSON_NO_GOAL_PATH))

                            j.add()
                            confirmTimer.stopTimer()
                            n.reset(0)
                            logTimer.startTimer(INTERVAL)
                    elif n.get() == 1:
                        Z = X
                        confirmTimer.startTimer(10)
                        print("Attempting to confirm 'no_goal' sample...")
                    elif logTimer.getTimeLeft() <= 0:
                        n.add()
                        logTimer.stopTimer()
                    elif logTimer.getIsRunning() == False:
                        n.reset(0)
                        logTimer.startTimer(INTERVAL)
                        print("logTimer started. T-{} seconds...".format(INTERVAL))
                    
                    if prediction[0][NO_GOAL] == 1 and freqTimer.getIsRunning() == True:
                        freqTimer.pauseTimer()
                    elif freqTimer.getIsPaused() == True:
                        freqTimer.resumeTimer()
        
    stream.close()
    pa.terminate()

    clock = -1000

    while getState() == "playing" and pauseTimer.getIsRunning():
        if pauseTimer.getTimeLeft() <= 0:
            subprocess.getoutput(STOP_COMMAND)
            pauseTimer.stopTimer()
        elif clock != math.floor(pauseTimer.getTimeLeft()):
            clock = math.floor(pauseTimer.getTimeLeft())
            if clock <= 3:
                print(str(clock) + "...\t", end=' ')

    print("\nProgram terminated. \n")



