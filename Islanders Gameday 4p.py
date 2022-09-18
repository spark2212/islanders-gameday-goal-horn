import pyaudio
import librosa
import numpy as np
import time
import subprocess
import os
import sys
import tensorflow.keras as keras
import json
import math
import scipy.io.wavfile
import pyautogui as pag
import kbHitMod
import pyttsx3
import sounddevice as sd
from datetime import datetime

## CONSTANTS ###################

now = datetime.now()

WIN = 0
NO_GOAL = 1
GOAL = 2

WIN_TIME = 196.383
OT_TIME = 222.558
GOAL_TIME = 42.353
SO_TIME = WIN_TIME
WIN_FLAG_TIME = 5.762

CHEER_TIME = 98.168
TRANSITION_TIME = 42.338

TAPS_TIME = 79.229

RECAP_TIME = 360
GAME_TIME = 1500
BASIC_TIME = RECAP_TIME * GAME_TIME

corner_size = 20

## NON_CONSTANT VARIABLES WITH DEFAULTS ##########
MAX_RIGHT = pag.size()[0]
MAX_DOWN = pag.size()[1]

overtime = False
shootout = False
quit_ = False
demo = False
remote = False
mode = NO_GOAL
skip_speech = False

## TRACK NAMES, FILE PATHS, and OTHER STRINGS ##########

OT_GOAL_TRACK = "01 New York Islanders Overtime Goal and Win Horn || NYCB Live: Home of the Nassau Veterans Memorial Coliseum"
WIN_TRACK = "02 New York Islanders Win Horn || NYCB Live: Home of the Nassau Veterans Memorial Coliseum"
GOAL_TRACK = "03 New York Islanders Goal Horn || NYCB Live: Home of the Nassau Veterans Memorial Coliseum"
QUIET_TRACK = "04 pure silence"
QUIT_TRACK = "05 Quick change to quit"
STATUS_TRACK = "06 Quick change to status check"
CALIBRATE_TRACK = "07 Quick change to calibrate volume"
OT_TRACK = "08 Quick change to overtime"
SO_TRACK = "09 Quick change to shootout"
REG_TRACK = "10 Quick change to regulation"
DEMO_TRACK = "11 Quick change to demo"
REMOTE_TRACK = "12 Quick change to remote"
LOG_TIME_TRACK = "13 Quick change to LogTime"
TAPS_TRACK = "14 Quick change to Taps"
CHEER_TRACK = "15 New York Islanders Organ"
TRANSITION_TRACK = "16 Islanders Transition Track"
SO_HORN_TRACK = "17 New York Islanders Shootout Horn || NYCB Live: Home of the Nassau Veterans Memorial Coliseum"

SAD_TRACK = "Islanders Taps"
PLAYLIST_2 = "Almost All Game Mode Controls"

suffix = "NYCB Live: Home of the Nassau Veterans Memorial Coliseum"
greeting = "Welcome to the Islanders Goal Horn Program! upper-right corner to get started. lower-left corner to quit now. "

status_strings = [
    "quit",
    "status",
    "calibrate",
    "overtime",
    "shootout",
    "regulation",
    "demo",
    "remote",
    "logTime",
    "Taps",
    "cancel",
    "more options"
]

status_tracks = [
    QUIT_TRACK,
    STATUS_TRACK,
    CALIBRATE_TRACK,
    OT_TRACK,
    SO_TRACK,
    REG_TRACK,
    DEMO_TRACK,
    REMOTE_TRACK,
    LOG_TIME_TRACK,
    TAPS_TRACK
]

QUIT_INDEX = 0
STATUS_INDEX = 1
CALIBRATE_INDEX = 2 
OT_INDEX = 3
SO_INDEX = 4
REG_INDEX = 5
DEMO_INDEX = 6
REMOTE_INDEX = 7 
LOG_TIME_INDEX = 8 
TAPS_INDEX = 9
CANCEL_INDEX = 10

num_options = len(status_strings)
on_off_status_strings = ["off", "on"]

MODEL_PATH = MODEL_PATH = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/Goal_Model_2.model"

FALSE_NEGATIVE = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/TBD_FILES/probable_false_negative_"

TRIGGER_POSITIVE = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/TBD_FILES/just_before_the_trigger_"

LOG_FILE = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/TBD_FILES/new_reoccurring_samples_"

stringtime = now.strftime("%Y_%m_%d__%p_%I_%M_%S")

if stringtime[15] == "0":
    stringtime = stringtime[:15] + "_" + stringtime[16:]

if stringtime[15:17] == "12":
        stringtime = stringtime[:14] + " " + stringtime[15:]

GAME_LOG_PATH = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/Game Logs/Islanders_Game_Log_" + stringtime + ".txt"

mode_names = ["WIN", "NO_GOAL", "GOAL"]

model = keras.models.load_model(MODEL_PATH, compile=True)
engine = pyttsx3.init()

## CLASSES ######################################

class RingBuffer:
    def __init__(self,size_max):
        self.max = size_max
        self.data = []
        self.diff = 0
        self.cur = 0
        self.isFull = False

    def append(self,x):
        if self.isFull:
            self.data[self.cur] = x
            self.cur = (self.cur+1) % self.max
            self.diff += 1
        else:
            self.data.append(x)
            self.diff += 1
            if len(self.data) == self.max:
                self.cur = 0
                self.isFull = True

    def get(self):
        if self.isFull:
            return self.data[self.cur:]+self.data[:self.cur]
        else:
            return self.data

    def get_diff(self):
        diff = self.diff
        self.diff = 0
        return diff

class Timer:
    def __init__(self):
        self.count = False
        self.start = time.time()
        self.target = -1000
        self.cache = -1000
        self.isPaused = False
        self.runtime = 0
        self.calltime = self.start
    
    def startTimer(self, t1):
        self.start = self.calltime = time.time()
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
        self.isPaused = False
        self.target = -1000
        self.cache = -1000
        self.start = time.time()
        self.runtime = 0

    def addTime(self, delta_t):
        self.target += delta_t

class Counter:
    def __init__(self, start_val=None):
        if start_val is None:
            start_val = 0
            
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
    def gets(self):
        output = str(self.value)
        return output.zfill(3)

## CLASS ITEMS INSTANTIATED #######################

ringBuffer = RingBuffer(1*22050)
ringBuffer2 = RingBuffer(25) # Was 36 
ringBuffer2a = RingBuffer(25)
ringBuffer3 = RingBuffer(25) # Was 36

pauseTimer = Timer()
cheerTimer = Timer()
logTimer = Timer()
cooldownTimer = Timer()
totalTimer = Timer()

false_num = Counter(1) 
false_num2 = Counter(1)
trigger_num = Counter(1) 
trigger_num2 = Counter(1) 
log_num = Counter(1)
log_num2 = Counter(1)
kbOverload = Counter(0)

logTime = 0
total_pause_time = 0

cooled = 0
loop_thru = False
includeMe = True
includeNum = 0

## LOGIC FUNCTIONS ###########################

def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

## TIME-RELATED FUNCTIONS #######################

def delay(t):
    t0 = time.time()

    if t < 0:
        print("negative timer!")
        t = 0
    elif t > 300:
        print("I'm not waiting more than five minutes!")
        t = 300

    while time.time() - t0 <= t: 1
    return

def convert_time(seconds, sep=None, shorten=False, forSpeech=False):
    hours = math.floor(seconds/3600)
    seconds -= hours * 3600

    mins = math.floor(seconds/60)
    seconds -= mins * 60
    seconds = math.floor(seconds)

    separator = [" hours, ", " minutes and ", " seconds"]; i = 0

    if hours == 1:
        separator[0] = " hour, "
    if mins == 1:
        separator[1] = " minute and "
    if seconds == 1:
        separator[2] == " seconds"

    if sep is not None:
        separator.append(sep)
        separator.append(sep)
        separator.append(None)
        i = 3

    time = str(hours) + separator[i]; i += 1
    
    if mins < 10 and forSpeech == False:
        time += "0"
    
    time += str(mins) + separator[i]; i += 1
    if seconds < 10 and forSpeech == False:
        time += "0"
    
    time += str(seconds)
    if separator[i] is not None:
        time += separator[i]

    if hours == 0 and shorten == True:
        if mins > 0:
            time = str(mins) + separator[i-1]
            time += str(seconds)
            if separator[i] is not None:
                time += separator[i]
        else:
            time = str(seconds)
            if separator[i] is not None:
                time += separator[i]
    return time

## MENU NAVIGATION ############################
def getMousePosition(cornersOnly=False):
    right = pag.size()[0]
    bottom = pag.size()[1]
    global corner_size
    size = corner_size
    
    if pag.position()[0] < size and pag.position()[1] < size:
        return "upper_left"
    elif pag.position()[0] > 0.4*right and pag.position()[0] < 0.6*right and pag.position()[1] < size and cornersOnly == False:
        return "north"
    elif pag.position()[0] > right-size and pag.position()[1] < size:
        return "upper_right"
    elif pag.position()[0] > right-size and pag.position()[1] > 0.4*bottom and pag.position()[1] < 0.6*bottom and cornersOnly == False:
        return "east"
    elif pag.position()[0] < size and pag.position()[1] > bottom-size:
        return "lower_left"
    elif pag.position()[0] > 0.4*right and pag.position()[0] < 0.6*right and pag.position()[1] > bottom-size and cornersOnly == False:
        return "south"
    elif pag.position()[0] > right-size and pag.position()[1] > bottom-size:
        return "lower_right"
    elif pag.position()[0] < size and pag.position()[1] > 0.4*bottom and pag.position()[1] < 0.6*bottom and cornersOnly == False:
        return "west"
    else:
        return "none"

def text_to_speech(string, interrupt=False, stopFlags=["none"]):  
    lastFlag = getMousePosition()
    def onWord(name, location, length):
       
        if interrupt == True and getMousePosition() in stopFlags:
            nonlocal lastFlag
            lastFlag = getMousePosition()
            engine.stop()

    token = engine.connect('started-word', onWord)

    engine.say(string)
    engine.runAndWait()
    engine.disconnect(token=token)
    return lastFlag

def waitForMouseReset(cornersOnly=False, message=None):
    while getMousePosition(cornersOnly=cornersOnly) != "none":
        if message is not None:
            text_to_speech(message, interrupt=True)

def getSelection(cornersOnly=False, wait_time=0.5):
    t0 = time.time()
    while getMousePosition(cornersOnly=cornersOnly) != "none" and time.time() - t0 < wait_time: 1

    return getMousePosition(cornersOnly=cornersOnly)

## OSASCRIPTS AND STATUS FUNCTIONS ########################

def runApp(app_name):
    subprocess.getoutput("osascript -e 'tell application \"" + app_name + "\" run'")

def getState():
    return subprocess.getoutput("osascript -e 'tell application \"iTunes\" to player state as string'") 

def getTrack():
    #global steady_track
    #if getState() == "playing" or steady_track == "":
        return subprocess.getoutput("osascript -e 'tell application \"iTunes\" to name of current track as string'")
    #else:
        #return steady_track

def getCurTime():
    curTime = subprocess.getoutput("osascript -e 'tell application \"iTunes\" to player position'")

    if curTime == "missing value":
        return 0
    else:
        return float(curTime)

def setCurTime(cursor):
    subprocess.getoutput("osascript -e 'tell application \"iTunes\" to set player position to " + str(cursor) + "'")
        

def stop():
    global steady_curTime, steady_track
    steady_track = ""
    steady_curTime = 0
    subprocess.getoutput("osascript -e 'tell application \"iTunes\" to stop'")
    

def pause():
    global steady_curTime
    steady_curTime = getCurTime()
    subprocess.getoutput("osascript -e 'tell application \"iTunes\" to pause'")

def play(track_name=None, playlist="All New York Islanders Goal Horns", curTime=None, skip_val=False):
    if track_name is None:
        subprocess.getoutput("osascript -e 'tell application \"iTunes\" to play'")
    else:
        subprocess.getoutput("osascript -e 'tell application \"iTunes\" to play (first track of playlist \"" + playlist + "\" whose name is \"" + track_name + "\")'")

    if skip_val == False:
        if (getState() != "playing" and track_name is None) or (getTrack() != track_name and track_name is not None):
            if track_name is None:
                print("\nUnable to begin playback!")
                text_to_speech("Unable to begin playback!")

            else:
                print("Track \"" + track_name + "\" not found in playlist \"" + playlist + "\".\nSearching in full library...")

                subprocess.getoutput("osascript -e 'tell application \"iTunes\" to play (first track of playlist \"Library\" whose name is \"" + track_name + "\")'")

                if getTrack() == track_name:            
                    print("Found it!")

                else: 
                    print("Track \"" + track_name + "\" does not exist.")
                    #print("Playing random track.")
                    #subprocess.getoutput("osascript -e 'tell application \"iTunes\" to play'")

        
    if getState() == "playing" and (curTime is not None or track_name is not None): 
        if curTime is not None:
            if getCurTime() < curTime:
                setCurTime(curTime)

    #elif getState() != "playing" and track_name is not None:
        #text_to_speech("Something is wrong. Maybe you should take a look.")


def getVolume():
    return float(subprocess.getoutput("osascript -e 'tell application \"iTunes\" to sound volume as integer'"))

def setVolume(vol, direction=None):
    delta_V = 0
    current_V = getVolume()
    if is_integer(vol) == False:
        print("Invalid input!")
        return
    elif direction is None or direction == False:
        if vol < 0 or vol > 100:
            print("Requested volume is out of range!")
            return
        else:
            delta_V = vol - current_V
    else:
        delta_V = vol

    new_V = current_V + delta_V

    if new_V < 0: new_V = 0
    elif new_V > 100: new_V = 100

    subprocess.getoutput("osascript -e 'tell application \"iTunes\" to set sound volume to " + str(new_V) + "'")

def calibrateVolume(remote=False):
    print("Calibrating volume. Upper right to increase volume, upper left to decrease volume. Lower right to confirm, lower left to cancel.")
    if remote == False:
        print("Calibrating volume. The current volume is {}".format(getVolume()))
        text_to_speech("Calibrating volume. The current volume is {}".format(getVolume()), interrupt=True)
        while getMousePosition(cornersOnly=True) != "none": 1

    loopTimer = Timer()
    curState = getState()
    curTrack = None
    curTime = None
    if curState == "playing":
        curTrack = getTrack()
        curTime = getCurTime()

        pauseTimer.pauseTimer()
        cheerTimer.pauseTimer()
        stop()

    
    loopTimer.startTimer(10)
    play(GOAL_TRACK, curTime=1.1)

    old_volume = getVolume()

    
    while remote == False or getState() != "paused":
        while getMousePosition(cornersOnly=True) == "upper_right":
            if getVolume() < 100:
                setVolume(1, True)
                delay(0.2)
            else:
                setVolume(100)
                text_to_speech("maximum volume reached.", interrupt=True)

            if loopTimer.getTimeLeft() <= 0:
                play(GOAL_TRACK, curTime=1.1)
                loopTimer.startTimer(10)

        while getMousePosition(cornersOnly=True) == "upper_left":
            if getVolume() > 0:
                setVolume(-1, True)
                delay(0.2)
            else:
                setVolume(0)
                text_to_speech("minimum volume reached.", interrupt=True)

            if loopTimer.getTimeLeft() <= 0:
                play(GOAL_TRACK, curTime=1.1)
                loopTimer.startTimer(10)

        if "lower" in getMousePosition(cornersOnly=True):
            if getMousePosition(cornersOnly=True) == "lower_left":
                setVolume(old_volume)
                text_to_speech("Cancelled.", False)
            else:
                engine.setProperty('volume', getVolume()/100)
                if getVolume() < 50:
                    engine.setProperty('volume', 0.5)
                
                print("The new volume is {}".format(getVolume()))
                text_to_speech("The new volume is {}".format(getVolume()), True)

            break

        if loopTimer.getTimeLeft() <= 0:
            play(GOAL_TRACK, curTime=1.1)
            loopTimer.startTimer(10)
    
    stop()
    if curState == "playing":
        play(curTrack, curTime=curTime)
        pauseTimer.resumeTimer()
        cheerTimer.resumeTimer()

    return
    
            
def restart_iTunes(track=None, playlist="All New York Islanders Goal Horns", cur=None):
    subprocess.getoutput("osascript -e 'tell application \"iTunes\" to quit'")

    delay(3)
    subprocess.getoutput("osascript -e 'tell application \"iTunes\" to run'")

    if track is not None:
        play(track_name=track, playlist=playlist, curTime=cur)

def printStatus():
    if overtime: print("\nOvertime mode: ON")
    else: print("\nOvertime mode: off")

    if shootout: print("Shootout mode: ON")
    else: print("Shootout mode: off")

    if demo: print("Demo mode: ON")
    else: print("Demo mode: off")

    if remote: print("Remote control: ON")
    else: print("Remote control: off")

def flash_screen(num_times=3, num_loops=5):
    script = """osascript -e '
    tell application "System Preferences"
        run
        reveal anchor "Hearing" of pane id "com.apple.preference.universalaccess"
    end tell
    tell application "System Events" to tell process "System Preferences" to click button "Test Screen Flash" of window "Accessibility" of application process "System Preferences" of application "System Events"'
    """

    for j in range(num_loops):
        for i in range(num_times):
            subprocess.getoutput(script)
            delay(0.5)
            
        if j != num_loops -1:
            delay(1)


########## CALLBACK and DATA HANDLING FUNCTIONS ####################

def callback(in_data, frame_count, time_info, flag):
        
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    audio_data = np.nan_to_num(audio_data)

    audio_data = librosa.resample(audio_data, 44100, 22050)

    for i in audio_data:
        ringBuffer.append(i)
            
    return (in_data, pyaudio.paContinue)

def save_data(tbd_path, counter1, counter2, index):  
    data = ringBuffer2.get()
    while os.path.exists(tbd_path + counter1.gets() + "_timestamps.json") or os.path.exists(tbd_path + counter1.gets() + "_" + counter2.gets() + ".wav"):
        counter1.add()
        counter2.reset()

    timestamps = ringBuffer2a.get()
    timestamps[index] = 0
    
    with open(tbd_path + counter1.gets() + "_timestamps.json", "w") as fp0:
        json.dump(timestamps, fp0, indent=4)

    print("\nTimestamps saved to \n{}\n".format(tbd_path + counter1.gets() + "_" + "timestamps.json"))

    for d in data[index:]:
        scipy.io.wavfile.write(tbd_path + counter1.gets() + "_" + counter2.gets() + ".wav", 22050, d)
        
        d = librosa.feature.mfcc(y=d, sr=22050, n_mfcc=44, hop_length=512, n_fft=2048)
        d = d.T
        with open(tbd_path + counter1.gets() + "_" + counter2.gets() + ".json", "w") as fp:
            json.dump(d.tolist(), fp, indent=4)
        
        print("Data saved to \n{} \nand {}\n".format(tbd_path + counter1.gets() + "_" + counter2.gets() + ".wav", tbd_path + counter1.gets() + "_" + counter2.gets() + ".json"))
        
        counter2.add()
    
    counter2.reset()
    counter1.add()

    print("Data saved for manual sorting! \nmode == {}".format(mode_names[mode]))   

############ MAIN #################################################

if __name__ == "__main__":

    engine.setProperty('rate', 200)
    vol = getVolume()
    if vol < 70:
        vol = 70
    engine.setProperty('volume', vol/100)

    totalTimer.startTimer(1)
    play(QUIET_TRACK)
    stop()

    pa = pyaudio.PyAudio()
    stream = pa.open(format = pyaudio.paFloat32,
                    channels = 1,
                    rate = 44100,
                    output = False,
                    input = True,
                    stream_callback=callback)

    stream.start_stream()
    print("\nStream open!")
    if getVolume() == 0:
        setVolume(70)

    hotCorner = text_to_speech(greeting, True, ["upper_right", "lower_left"])

    #waitForMouseReset(cornersOnly=True, message=greeting)
    while hotCorner != "upper_right" and hotCorner != "lower_left": 
        hotCorner = getMousePosition(cornersOnly=True)

    logTime = BASIC_TIME
    logTimer.startTimer(logTime)

    need2save = False
    alerted = False
    openingCheer = True
    silent_horn = False
    i2 = Counter(0)
    snippetTimer = Timer()
    delayTime = 0

    ringBuffer3.append(True)   
    for i in range(20-19):
        ringBuffer3.append(False)    

    if hotCorner == "upper_right":
        print("\nLet's watch some hockey!!!\n\nOvertime mode: off\nShootout mode: off\nDemo mode: off\nRemote control: off")
        
        text_to_speech("Let's go Islanders!")
    else:
        quit_ = True

    play(CHEER_TRACK)
    cheerTimer.startTimer(CHEER_TIME)

    kb = kbHitMod.KBHit()
    we_won = False
    we_lost = False
    curState = None
    curTrack = None
    curTime = None
    fadeout = False
    num_goals = 0
    
    # start of stream    
    while stream.is_active():
        status_num = CANCEL_INDEX
        command = kb.getch()
        kb.off()

        if command == "\r":
            if kbOverload.get() > 0:
                printStatus()
                
            kbOverload.reset()
        elif kbOverload.get() >= 2:
            quit_ = True
            print("Keyboard overloaded. Quitting...")
            text_to_speech("Keyboard overloaded. Quitting...")
        else:
            kbOverload.add()

        if getMousePosition() == "upper_right" != hotCorner:
            text_to_speech("Menu.")
        elif getMousePosition(cornersOnly=True) == "lower_left" != hotCorner and we_won == True:
            text_to_speech("Fade out?", True)

        getSelection(wait_time=1)

        if quit_ == True:
            quit_ = True
            break
        elif getMousePosition(cornersOnly=True) == "upper_right" and getMousePosition() != hotCorner: 
            text_to_speech("Menu open.", interrupt=True)

            menu = """
            \rPage 1:                    Page 2:
            \r*----------------------*   *----------------------* 
            \r|Cancel    OT    Status|   |Cancel   Demo   Status|
            \r|                      |   |                      |
            \r|SO                Reg.|   |Remote         LogTime|
            \r|                      |   |                      |
            \r|Quit     Vol.     More|   |Quit     Taps     More|
            \r*----------------------*   *----------------------*
            """

            northButtons = [OT_INDEX, DEMO_INDEX]
            southButtons = [CALIBRATE_INDEX, TAPS_INDEX]
            eastButtons = [REG_INDEX, LOG_TIME_INDEX]
            westButtons = [SO_INDEX, REMOTE_INDEX]

            page = 0
            print(menu)

            while True:
                newCorner = getMousePosition()

                if newCorner == "upper_left":
                    status_num = CANCEL_INDEX
                elif newCorner == "upper_right":
                    status_num = STATUS_INDEX
                elif newCorner == "lower_left":
                    status_num = QUIT_INDEX
                elif newCorner == "lower_right":
                    status_num = -1
                elif newCorner == "north":
                    status_num = northButtons[page]
                elif newCorner == "south":
                    status_num = southButtons[page]
                elif newCorner == "east":
                    status_num = eastButtons[page]
                elif newCorner == "west":
                    status_num = westButtons[page]
                elif newCorner == "none":
                    continue

                text_to_speech(status_strings[status_num], interrupt=True)
                if newCorner == getSelection(False):
                    if status_num != -1:
                        if status_num == CANCEL_INDEX: 
                            print("Cancelled.")
                            text_to_speech("Cancelled.")
                        else: 
                            print("You selected {}".format(status_strings[status_num]))
                            text_to_speech("You selected {}".format(status_strings[status_num]))
                        hotCorner = newCorner

                        break
                    else:
                        
                        print("You selected more options.")
                        text_to_speech("You selected more options.")
                        if getSelection(False) != "none":
                            text_to_speech("Return to center.", True)

                        waitForMouseReset(False)
                        status_num = CANCEL_INDEX
                        page += 1
                        page %= 2

                        continue
                else:
                    newCorner = "none"
                    status_num = CANCEL_INDEX
                    continue

        elif getMousePosition(cornersOnly=True) == "upper_left" != hotCorner or command == "p": 
            print("Paused!")

            t0 = time.time()
            
            curState = getState()
            runCheer = cheerTimer.getIsRunning()
            runPause = pauseTimer.getIsRunning()
            if runCheer:
                cheerTimer.pauseTimer()
            if runPause:
                pauseTimer.pauseTimer()
            
            logTimer.pauseTimer()
            pause()
            if command == "p":
                text_to_speech("Paused. Hold the mouse in the upper right corner to unpause, or hold in the lower left to quit.", interrupt=True)
                while getSelection() != "upper_right" and getSelection() != "lower_left": 
                    if int(time.time() - t0) % 60 == 0:
                        flash_screen(2, 1)

                if getMousePosition() == "lower_left":
                    quit_ = True
            else:
                text_to_speech("Paused. Move away from the corner to unpause.", interrupt=True)
                waitForMouseReset()

            if quit_ == False:
                print("Unpaused!")
                text_to_speech("Unpaused")
                
                if runCheer:
                    cheerTimer.resumeTimer()
                if runPause:
                    pauseTimer.resumeTimer()
                if curState == "playing":
                    play()

            t1 = time.time()
            total_pause_time += (t1-t0)

        elif getMousePosition(cornersOnly=True) == "lower_left" != hotCorner and we_won == True:
            text_to_speech("Fading out.")
            fadeout = True

        elif getMousePosition() != hotCorner:
            hotCorner = getMousePosition()
        elif getMousePosition() == hotCorner != "none":
            print("Return to center.")
            text_to_speech("Return to center.", interrupt=True)

        signal = ringBuffer.get()
        signal = np.array(signal, np.float32)
        ringBuffer2.append(signal)
        ringBuffer2a.append(ringBuffer.get_diff())

        if (mode == NO_GOAL and pauseTimer.getIsRunning() == False) or loop_thru == True:
            if signal.shape[0] == 22050:

                X = librosa.feature.mfcc(signal, sr=22050, n_mfcc=44, hop_length=512, n_fft=2048)
                X = X.T

                X_new = X[np.newaxis, ...]
                prediction = model(X_new) 

                prediction_result = np.argmax(prediction)
                
                if prediction_result != NO_GOAL or loop_thru == True:
                    i2.add()

                    if loop_thru == False:
                        snippetTime = 1
                        snippetTimer.startTimer(snippetTime)
                        mode = prediction_result
                        loop_thru = True
                        includeMe = True
                        cheerTimer.stopTimer()
                        we_won = False
                        stop()
                    elif i2.get() == 2:
                        delayTime = snippetTimer.getTimeLeft()/10
                        #print("delayTime == {}".format(delayTime))
                        if getState() != "playing" and mode != NO_GOAL:
                            silent_horn = True

                        continue
                    else:
                        loop_thru = False
                        if i2.get() < 100:
                            if prediction_result != NO_GOAL:
                                loop_thru = True
                                includeMe = True
                                if shootout and prediction_result == WIN:
                                    mode = WIN
                                
                        if loop_thru:
                            delay(delayTime)
                            continue
                        else:
                            if includeMe and not demo:
                                save_data(TRIGGER_POSITIVE, trigger_num, trigger_num2, includeNum - i2.get())
                                includeNum = i2.get()
                                includeMe = False

                            if shootout and mode != WIN and i2.get() < 100:
                                loop_thru = (getState() == "playing" and getTrack() != QUIET_TRACK)
                                delay(delayTime)
                                continue

                            snippetTimer.stopTimer()
                            if silent_horn:
                                text_to_speech("{}! Is the horn broken?".format(mode_names[mode]))
                                silent_horn = False

                            i2.reset()
                            includeNum = 0
                        
                if mode == GOAL:
                    num_goals += 1
                    pauseTimer.stopTimer()
                    print("prediction == {}".format(prediction))
                
                    if overtime:
                        pauseTimer.startTimer(OT_TIME)
                        play(OT_GOAL_TRACK, skip_val=True)
                        print("Goal! We win!")
                    elif shootout:
                        print("Shootout goal!")
                        pauseTimer.startTimer(SO_TIME)
                        play(SO_HORN_TRACK, skip_val=True)
                    else:
                        print("Goal!")
                        pauseTimer.startTimer(GOAL_TIME)
                        play(GOAL_TRACK, skip_val=True)

                    print("\n")
                    continue
                            
                elif mode == WIN:
                    print("prediction == {}".format(prediction))
                    pauseTimer.stopTimer()
                    if shootout:
                        pauseTimer.startTimer(SO_TIME)
                        play(SO_HORN_TRACK, skip_val=True)
                    else:
                        play(WIN_TRACK, skip_val=True)
                        pauseTimer.startTimer(WIN_TIME)
                    
                    print("We win!\n")
                    continue

                elif logTimer.getTimeLeft() <= 0 and demo == False:
                    save_data(LOG_FILE, log_num, log_num2, index=-10)
                    logTimeCurrent = logTime + logTimer.getTimeLeft()
                    logTimer.startTimer(logTimeCurrent)
                    
        tracks = [WIN_TRACK, QUIET_TRACK, GOAL_TRACK]
        if shootout:
            tracks = [SO_HORN_TRACK, QUIET_TRACK, SO_HORN_TRACK]
        elif overtime:
            tracks = [WIN_TRACK, QUIET_TRACK, OT_GOAL_TRACK]

        state = getState()
        track = getTrack()

        if track not in status_tracks:
            curState = state
            if state == "playing":
                curTrack = track
                if not fadeout:
                    curTime = getCurTime()
            else:
                curTrack = None
                curTime = None

        
        if track in status_tracks or status_num != CANCEL_INDEX or command != "\r":
           
            if track in status_tracks and curState == "playing":
                play(curTrack, curTime=curTime)

            if track == QUIT_TRACK or status_num == QUIT_INDEX or command == "q":
                quit_ = True
                break
            elif track == STATUS_TRACK or status_num == STATUS_INDEX or command == " ":
                printStatus()
                string = ""
                if overtime:
                    string = "The current mode is overtime."
                elif shootout:
                    string = "The current mode is shootout."
                else:
                    string = "The current mode is regulation."
                
                string += "Demo mode is {}. Remote mode is {}".format(on_off_status_strings[int(demo)], on_off_status_strings[int(remote)])

                if logTime == RECAP_TIME:
                    print("Game mode is recap")
                    string += "Game mode is recap."
                elif logTime == GAME_TIME:
                    print("Game mode is full game.")
                    string += "Game mode is full game."
                else:
                    print("Game mode is no game.")
                    string += "Game mode is no game."

                print("logTimer is at T - {}. \nThis program has been active for {}.".format(int(logTimer.getTimeLeft()), convert_time(totalTimer.getTimeElapsed())))

                string += "log timer is at T minus {}. This program has been active for {}.".format(int(logTimer.getTimeLeft()), convert_time(totalTimer.getTimeElapsed(), shorten=True, forSpeech=True))

                text_to_speech(string, interrupt=True, stopFlags=["none"])

            elif track == CALIBRATE_TRACK or status_num == CALIBRATE_INDEX or command == "c":
                calibrateVolume(remote=(track == CALIBRATE_TRACK))
            elif track == OT_TRACK or status_num == OT_INDEX or command == "o":
                overtime = True
                shootout = False
                text_to_speech("Switching to overtime mode.")
            elif track == SO_TRACK or status_num == SO_INDEX or command == "s":
                overtime = False
                shootout = True
                text_to_speech("Switching to shootout mode.")
            elif track == REG_TRACK or status_num == REG_INDEX or command == "n":
                overtime = False
                shootout = False
                text_to_speech("Switching to regulation mode.")
            elif track == DEMO_TRACK or status_num == DEMO_INDEX or command == "d":
                demo = (not demo)
                text_to_speech("Demo mode is {}".format(on_off_status_strings[int(demo)]))
            elif track == REMOTE_TRACK or status_num == REMOTE_INDEX or command == "r":
                remote = (not remote)
                text_to_speech("remote mode is {}".format(on_off_status_strings[int(remote)]))
            elif track == LOG_TIME_TRACK or status_num == LOG_TIME_INDEX or command == "m":
                string = ""
                if logTime == RECAP_TIME:
                    logTime = GAME_TIME
                    logTimer.addTime(GAME_TIME-RECAP_TIME)
                    print("The game mode is full game.")
                    string = "The game mode is full game."
                elif logTime == GAME_TIME:
                    logTime = BASIC_TIME
                    logTimer.addTime(BASIC_TIME-GAME_TIME)
                    print("The game mode is no game.")
                    string = "The game mode is no game."
                else:
                    logTime = RECAP_TIME
                    logTimer.addTime(RECAP_TIME-BASIC_TIME)
                    print("The game mode is recap.")
                    string = "The game mode is recap."
                    
                print("logTimer at T - {}".format(int(logTimer.getTimeLeft())))
                text_to_speech(string)
                text_to_speech("log timer at T minus {}".format(int(logTimer.getTimeLeft())), interrupt=True)
            elif track == TAPS_TRACK or status_num == TAPS_INDEX or command == "l":
                stop()
                text_to_speech("sigh.")
                play(SAD_TRACK, PLAYLIST_2)
                pauseTimer.startTimer(TAPS_TIME)
                we_won = False
                we_lost = True
                quit_ = True
                engine.setProperty('volume', 0.5)
                continue
            
        elif pauseTimer.getIsRunning():
            if state == "paused" or getTrack() == QUIET_TRACK:
                if pauseTimer.getTimeElapsed() <= 3:
                    print("Never mind!")
                
                print("Back to the game!")
                printStatus()

                pauseTimer.stopTimer()
                mode = NO_GOAL
                stop()
            elif (track != tracks[mode] or fadeout == True) and we_won == True:
                volume = getVolume()
                if not fadeout:
                    play(tracks[mode], curTime=pauseTimer.getTimeElapsed())

                pauseTimer.startTimer(13.1)
                while getVolume() > 0:
                    setVolume(-5, direction=True)
                    
                stop()
                setVolume(volume)
                quit_ = True
                pauseTimer.startTimer(3.1)
                break
            elif pauseTimer.getTimeElapsed() > 30 and track != SAD_TRACK and (mode == WIN or overtime or shootout):
                we_won = True
                if pauseTimer.getTimeLeft() < 4:
                    print("Game over! Goodbye!\n")
                    quit_ = True   
                    skip_speech = True 
            elif pauseTimer.getTimeLeft() <= 4 and track == SAD_TRACK:
                quit_ == True
                break
            elif pauseTimer.getTimeLeft() <= 0 or (pauseTimer.getTimeElapsed() > WIN_FLAG_TIME and shootout and mode != WIN):
                print("Back to the game!\nLet's go Islanders!!!")
                pauseTimer.stopTimer()
                mode = NO_GOAL 
                stop()

        elif cheerTimer.getIsRunning() == True:
            if cheerTimer.getTimeLeft() <= 0 or track != CHEER_TRACK or state != "playing":
                cheerTimer.stopTimer()
                openingCheer = False
                if state != "playing" or track not in tracks:
                    stop()
        
        # Trigger false negatives!!!
        elif state == "playing" and mode == NO_GOAL:
            smoothTimer = Timer()
            smoothTimer.startTimer(3)
            print("Did I miss something?")
            
            if remote == False or suffix not in track:
                if overtime and track != OT_GOAL_TRACK:
                    track = OT_GOAL_TRACK
                elif shootout and track != SO_HORN_TRACK:
                    track = SO_HORN_TRACK
                elif track != GOAL_TRACK:
                    track = GOAL_TRACK

                play(track)
                smoothTimer.startTimer(3)

            while smoothTimer.getTimeLeft() >= 0:
                if getState() == "paused" or suffix not in getTrack():
                    if track != tracks[WIN] and remote == False:
                        play(tracks[WIN])
                        track = tracks[WIN]
                        smoothTimer.startTimer(3)
                        mode = WIN
                        delay(0.1)
                    else:
                        print("I guess not!")

                        mode = NO_GOAL
                        stop()
                        smoothTimer.stopTimer()
                        break

            
            if getState() == "playing":
                print("I guess I did!")
                track = getTrack()
                if track == tracks[GOAL] and mode != WIN:
                    mode = GOAL
                    if shootout == True:
                        pauseTimer.startTimer(SO_TIME - smoothTimer.getTimeElapsed())
                    elif overtime == True:
                        pauseTimer.startTimer(OT_TIME - smoothTimer.getTimeElapsed())
                    else:    
                        pauseTimer.startTimer(GOAL_TIME - smoothTimer.getTimeElapsed())
                    
                elif mode == WIN:
                    pauseTimer.startTimer(WIN_TIME - smoothTimer.getTimeElapsed())

                if demo == False:  
                    save_data(FALSE_NEGATIVE, false_num, false_num2, index=-1*min(25, len(ringBuffer2.get())))  

    stream.close()
    pa.terminate()
    ## End of stream #########

    while pauseTimer.getIsRunning() == True and pauseTimer.getTimeLeft() > 3 and getState() == "playing": 1

    if we_won == True:
        pauseTimer.startTimer(TRANSITION_TIME)
        play(TRANSITION_TRACK)
        
        while getState() == "playing" and getTrack() == TRANSITION_TRACK and pauseTimer.getTimeLeft() > 3: 1

    if pauseTimer.getTimeLeft() <= 3 and pauseTimer.getIsRunning() == True and getTrack() == TRANSITION_TRACK:
        
        pauseTimer.startTimer(3)
        print("{}...".format(str(3), end=' ', flush=True))
        delay(pauseTimer.getTimeLeft()-2)
        print("{}...".format(str(2), end=' ', flush=True))
        delay(pauseTimer.getTimeLeft()-1)
        print("{}...".format(str(1), end=' ', flush=True))
        delay(pauseTimer.getTimeLeft())
        print("0.")
    
    stop()
    
    total_run_time = totalTimer.getTimeElapsed()
    print("\nTotal active time: \t{}\nTotal pause time: \t{} \nTotal runtime: \t\t{}\n\nProgram terminated. \n".format(convert_time(total_run_time-total_pause_time, ":"), convert_time(total_pause_time, ":"), convert_time(total_run_time, ":")))

    if getMousePosition() == "lower_left" or skip_speech == False: 
        text_to_speech("\nTotal active time: \t{}\nTotal pause time: \t{} \nTotal runtime: \t\t{}\n\nProgram terminated. Goodbye.\n".format(convert_time(total_run_time-total_pause_time, shorten=True, forSpeech=True), convert_time(total_pause_time, shorten=True, forSpeech=True), convert_time(total_run_time, shorten=True, forSpeech=True)), interrupt=True)

    with open(GAME_LOG_PATH, "w") as fp:
        endgame = None
        if we_won:
            if overtime:
                endgame = " with an overtime goal!"
            elif shootout:
                endgame = " with a shootout win!"
            else:
                endgame = " with a regulation win."
        elif we_lost:
            if overtime:
                endgame = " with an overtime loss."
            elif shootout: 
                endgame = " with a shootout loss."
            else:
                endgame = " with a regulation loss."
        else:
            endgame = " with the quit command."

        fp.write("Total Islanders Goals Detected: {}\nThe game ended {}\nTotal active time: \t{}\nTotal pause time: \t{} \nTotal runtime: \t\t{}\n\nProgram terminated. \n".format(num_goals, endgame, convert_time(total_run_time-total_pause_time, ":"), convert_time(total_pause_time, ":"), convert_time(total_run_time, ":")))

        fp.close()

    delay(1)

    ## End of main #####

    
            






