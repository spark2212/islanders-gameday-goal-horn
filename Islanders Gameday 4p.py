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

WIN = 0
NO_GOAL = 1
GOAL = 2
STATUS_TRACK_LENGTH = 6

overtime = False
shootout = False
quit_ = False
demo = False
remote = False
mode = NO_GOAL

OT_GOAL_TRACK = "01 New York Islanders Overtime Goal and Win Horn || NYCB Live: Home of the Nassau Veterans Memorial Coliseum"

WIN_TRACK = "02 New York Islanders Win Horn || NYCB Live: Home of the Nassau Veterans Memorial Coliseum"

GOAL_TRACK = "03 New York Islanders Goal Horn || NYCB Live: Home of the Nassau Veterans Memorial Coliseum"

QUIET_TRACK = "04 pure silence"

STATUS_NAME_INDEX = 0; STATUS_CHANGE_INDEX = 1; STATUS_QUICK_CHANGE_INDEX = 2; STATUS_ON_OFF = 3

CANCEL_INDEX = 0; OT_INDEX = 1; SO_INDEX = 2; REGULATION_INDEX = 3; DEMO_INDEX = 4; REMOTE_INDEX = 5; QUIT_INDEX = 6; TAPS_INDEX = 7; CALIBRATE_INDEX = 8; STATUS_CHECK_INDEX = 9

OFF = 0; ON = 1; MODE_STATUS = [OFF, ON]

STATUS_TRACKS = [
    ["Say the word 'cancel'", "Say, 'Currently in overtime mode'", "Say, 'Currently in shootout mode'", "Say, 'Currently in regulation mode'", "Say the word 'demo'", "Say the word 'remote'", "Say the word 'quit'", "Say the word 'Taps'"],
    ["Pause now to cancel", "Pause now for overtime", "Pause now for shootout", "Pause now for regulation", "Pause now for demo", "Pause now for remote", "Pause now to quit program", "Pause now for Taps", "Pause now for status check"],
    ["05 Quick change to cancel", "06 Quick change to overtime", "07 Quick change to shootout", "08 Quick change to regulation", "09 Quick change to demo", "10 Quick change to remote", "11 Quick change to quit", "12 Quick change to Taps", "13 Quick change to calibrate volume", "14 Quick change to status check"],
    ["Say the word 'off'", "Say the word 'on'"]
]

mode_names = ["WIN", "NO_GOAL", "GOAL"]

MODEL_PATH = MODEL_PATH = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/Goal_Model_2.model"

FALSE_NEGATIVE = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/TBD_FILES/probable_false_negative_"

TRIGGER_POSITIVE = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/TBD_FILES/just_before_the_trigger_"

LOG_FILE = "/Users/schoolwork/Documents/Goal_Horn_Project_Stuff/Goal_Horn_Program/Goal_Horn_Program_Subsets/TBD_FILES/new_reoccurring_samples_"

model = keras.models.load_model(MODEL_PATH, compile=True)

## CLASSES ######################################

class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self,size_max):
        self.max = size_max
        self.data = []
        self.diff = 0

    class __Full:
        """ class that implements a full buffer """
        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data[self.cur] = x
            self.cur = (self.cur+1) % self.max
            self.diff += 1
        def get(self):
            """ return list of elements in correct order """
            return self.data[self.cur:]+self.data[:self.cur]

        def get_diff(self):
            diff = self.diff
            self.diff = 0
            return diff

    def append(self,x):
        """append an element at the end of the buffer"""
        self.data.append(x)
        self.diff += 1
        if len(self.data) == self.max:
            self.cur = 0
            # Permanently change self's class from non-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data

    def get_diff(self):
            diff = self.diff
            self.diff = 0
            return diff

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
ringBuffer2 = RingBuffer(50) # Was 36 
ringBuffer2a = RingBuffer(50)
ringBuffer3 = RingBuffer(25) # Was 36

pauseTimer = ShutdownTimer()
logTimer = ShutdownTimer()
cooldownTimer = ShutdownTimer()
totalTimer = ShutdownTimer()

false_num = Counter(1) 
false_num2 = Counter(1)
trigger_num = Counter(1) 
trigger_num2 = Counter(1) 
log_num = Counter(1)
log_num2 = Counter(1)

logTime = 0
total_pause_time = 0

cooled = 0
loop_thru = False


## FUNCTION DEFINITIONS #######################################

def runApp(app_name):
    subprocess.getoutput("osascript -e 'tell application \"" + app_name + "\" run'")

def printStatus(overtime, shootout, demo, remote):
    if overtime: print("\nOvertime mode: ON")
    else: print("\nOvertime mode: off")

    if shootout: print("Shootout mode: ON")
    else: print("Shootout mode: off")

    if demo: print("Demo mode: ON")
    else: print("Demo mode: off")

    if remote: print("Remote control: ON")
    else: print("Remote control: off")

def convert_time(seconds, sep=None, shorten=False):
    hours = math.floor(seconds/3600)
    seconds -= hours * 3600

    mins = math.floor(seconds/60)
    seconds -= mins * 60
    
    seconds = math.floor(seconds)

    separator = [" hours, ", " minutes and ", " seconds"]
    i = 0

    if sep is not None:
        separator.append(sep)
        separator.append(sep)
        separator.append(None)
        i = 3

    time = str(hours) + separator[i]; i += 1
    
    if mins < 10:
        time += "0"
    
    time += str(mins) + separator[i]; i += 1
    if seconds < 10:
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

def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

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

        delay(1)

def getState():
    return subprocess.getoutput("osascript -e 'tell application \"iTunes\" to player state as string'") 

def getTrack():
    return subprocess.getoutput("osascript -e 'tell application \"iTunes\" to name of current track as string'")

def getVolume():
    return float(subprocess.getoutput("osascript -e 'tell application \"iTunes\" to sound volume as integer'"))

def setVolume(vol, direction=None):
    delta_V = 0
    current_V = getVolume()
    if direction is None or direction == False:
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

def getCurTime():
    curTime = subprocess.getoutput("osascript -e 'tell application \"iTunes\" to player position'")

    if curTime == "missing value":
        return 0
    else:
        return float(curTime)

def getSongLength():
    print("Song length == {}".format(subprocess.getoutput("osascript -e 'tell application \"iTunes\" to finish of current track as float'")))
    return float(subprocess.getoutput("osascript -e 'tell application \"iTunes\" to duration of current track as float'"))

def setCurTime(cursor):
    """
    maxTime = getSongLength()
    if cursor < 0:
        cursor = 0
    elif cursor > maxTime:
        cursor = getSongLength()
    """

    subprocess.getoutput("osascript -e 'tell application \"iTunes\" to set player position to " + str(cursor) + "'")

def play(track_name=None, playlist="All New York Islanders Goal Horns", curTime=None):
    if track_name is None:
        subprocess.getoutput("osascript -e 'tell application \"iTunes\" to play'")

        if curTime is not None:
            setCurTime(curTime)
    else:
        subprocess.getoutput("osascript -e 'tell application \"iTunes\" to play (first track of playlist \"" + playlist + "\" whose name is \"" + track_name + "\")'")

        cur = 0
        if curTime is not None:
            cur = curTime

        if math.floor(cur) != math.floor(getCurTime()):
            setCurTime(cur)


    if getState() != "playing":
        if track_name is None:
            print("\nUnable to begin playback!")
            flash_screen(3, 2)

        else:
            print("Track \"" + track_name + "\" not found in playlist \"" + playlist + "\".\nSearching in full library...")

            subprocess.getoutput("osascript -e 'tell application \"iTunes\" to play (first track of playlist \"Library\" whose name is \"" + track_name + "\")'")

            if getState == "playing":
                cur = 0
                if curTime is not None:
                    cur = curTime

                if math.floor(cur) != math.floor(getCurTime()):
                    setCurTime(cur)
            else:
                print("Track \"" + track_name + "\" does not exist. ")
                flash_screen(3, 2)


def stop():
    subprocess.getoutput("osascript -e 'tell application \"iTunes\" to stop'")

def pause():
    subprocess.getoutput("osascript -e 'tell application \"iTunes\" to pause'")

def restart_iTunes(track=None, playlist="All New York Islanders Goal Horns", cur=None):
    subprocess.getoutput("osascript -e 'tell application \"iTunes\" to quit'")

    delay(3)
    subprocess.getoutput("osascript -e 'tell application \"iTunes\" to run'")

    if track is not None:
        play(track_name=track, playlist=playlist, curTime=cur)

def speakStatus(overtime, shootout, demo, remote):
    speechTimer = ShutdownTimer()
    print("Speaking status...\n")
    printStatus(overtime, shootout, demo, remote)
    if overtime: 
        speechTimer.startTimer(2)
        play(STATUS_TRACKS[STATUS_NAME_INDEX][OT_INDEX], "Library")
        while getState() == "playing" and speechTimer.getTimeLeft() > 0: 1

    elif shootout: 
        speechTimer.startTimer(2)
        play(STATUS_TRACKS[STATUS_NAME_INDEX][SO_INDEX], "Library")
        while getState() == "playing" and speechTimer.getTimeLeft() > 0: 1
    
    else:
        speechTimer.startTimer(2)
        play(STATUS_TRACKS[STATUS_NAME_INDEX][REGULATION_INDEX], "Library")
        while getState() == "playing" and speechTimer.getTimeLeft() > 0: 1


    if demo and getState() == "playing": 
        speechTimer.startTimer(1)
        play(STATUS_TRACKS[STATUS_NAME_INDEX][DEMO_INDEX], "Library")
        while getState() == "playing" and speechTimer.getTimeLeft() > 0: 1
        speechTimer.startTimer(1)
        play(STATUS_TRACKS[STATUS_ON_OFF][int(demo)], "Library")
        while getState() == "playing" and speechTimer.getTimeLeft() > 0: 1

    if remote and getState() == "playing": 
        speechTimer.startTimer(1)
        play(STATUS_TRACKS[STATUS_NAME_INDEX][REMOTE_INDEX], "Library")
        while getState() == "playing" and speechTimer.getTimeLeft() > 0: 1
        speechTimer.startTimer(1)
        play(STATUS_TRACKS[STATUS_ON_OFF][int(remote)], "Library")
        while getState() == "playing" and speechTimer.getTimeLeft() > 0: 1

    if getState() != "playing":
        print("Status speech cancelled!")
        play(STATUS_TRACKS[STATUS_NAME_INDEX][CANCEL_INDEX], "Library")
        delay(1)
    
    stop()

def calibrateVolume():
    OG_Vol = getVolume()
    new_vol = OG_Vol
    done = False
    rate = 10
    
    loopTimer = ShutdownTimer()
    volumeTimer = ShutdownTimer()

    command = "\r"

    print("\nCalibrate iTunes volume? \nVolume will start at 50% and \ncontinue rising at a rate of 10% of maximum per second. \nTo calibrate at a different rate, press the tab key. \nTo cancel calibration and remain at the current volume, press 'c'. \nTo continue calibrating at the current rate, press enter. \nTo quit the entire program, press 'q'.")

    stop()

    while command != "c" and command != "\n" and command != "q":
        command = kb.getch()
        kb.off()

        if command == "\t":
            rate = int(input("Enter new rate of change of volume per second as a percentage of maximum: "))

            while rate < 1 or rate > 25:
                if rate < 1:
                    rate = int(input("That sounds painfully slow. Please pick something faster: "))
                elif rate > 25:
                    rate = int(input("That sounds dangerously fast. Please pick something slower: "))
            
            break
        elif command == "c":
            done = True
        elif command == "q":
            return True

          
    delta_vol = rate/5
    loopTimer.startTimer(4)
    volumeTimer.startTimer(0.2)
    volUp = True
    
    setVolume(50)

    play(GOAL_TRACK)
    setCurTime(1.1)

    breakout = 10

    print("\nCommencing calibration. Press the space bar to hold at the current volume, \npress it again to begin sweeping in the other direction, \npress 'c' to cancel calibration and restore the original settings, \npress 'q' to quit the entire program, \nor press enter to set the current volume as permanent")

    while done == False and breakout > 0:
        command = kb.getch()
        kb.off()

        if loopTimer.getTimeLeft() <= 0 and breakout == 10:
            play(GOAL_TRACK)
            subprocess.getoutput("osascript -e 'tell application \"iTunes\" to set player position to 1.1'")
            loopTimer.startTimer(4)
            if getVolume() == 0 and volUp == False:
                volUp = (not volumeTimer.getIsPaused())
                print("Minimum volume reached!")
                flash_screen(2, 1)
                delay(3)
                continue
            elif getVolume() == 100 and volUp == True:
                volUp = volumeTimer.getIsPaused()
                print("Maximum volume reached!")
                flash_screen(2, 1)
                delay(3)
                continue
        
        if command == " ":
            if volumeTimer.getIsPaused() == True:
                volumeTimer.resumeTimer()
                volUp = (not volUp)
                if volUp:
                    print("Increasing volume...")
                else:
                    print("Decreasing volume...")
            else:
                volumeTimer.pauseTimer()
                print("Paused!")
        
        elif command == "c":
            new_vol = OG_Vol
            print("Restoring volume to {}%...".format(new_vol))

            done = True
        elif command == "\n":
            new_vol = getVolume()
            print("Setting volume to {}%...".format(new_vol))
            
            done = True
        elif command == "q":
            return True
        elif command != "\r":
            print("Invalid input! ")
        elif getState() == "paused":
            while breakout > 0 and getState() == "paused":
                breakout -= 1
            if breakout <= 0:
                new_vol = getVolume()
                print("Setting volume to {}%...".format(new_vol))
        else:
            breakout = 10
                
        if volumeTimer.getTimeLeft() <= 0 and volumeTimer.getIsPaused() == False:
            direction = 1
            if volUp == False:
                direction = -1
            
            setVolume(direction * delta_vol, True)
            volumeTimer.startTimer(0.2)
            
    stop()
    setVolume(new_vol)
    print("New volume == {}".format(new_vol))
    flash_screen(3, 1)
    return False

def cycleThruStatusOptions():
    old_volume = getVolume()
    if old_volume < 60:
        old_volume = 60
    
    setVolume(90)
    songRepeatTimer = ShutdownTimer()
    songRepeatTimer.startTimer(STATUS_TRACK_LENGTH)
    play(STATUS_TRACKS[STATUS_CHANGE_INDEX][0], "Library")
    track = getTrack()
    delay(0.1)

    new_status = 0

    while getState() != "paused":
        while True:
            if getState() == "paused":
                if new_status == STATUS_CHECK_INDEX:
                    speakStatus(overtime, shootout, demo, remote); delay(1)
                    songRepeatTimer.startTimer(STATUS_TRACK_LENGTH)
                    play(STATUS_TRACKS[STATUS_CHANGE_INDEX][new_status], "Library", curTime=0)
                    delay(0.1)
                    continue
                else:
                    break

            elif track != getTrack() or getCurTime() < 0.1:
                new_status += 1
                new_status = new_status % 9
                songRepeatTimer.startTimer(STATUS_TRACK_LENGTH)
                play(STATUS_TRACKS[STATUS_CHANGE_INDEX][new_status], "Library")
                track = getTrack()
                delay(0.1)
            elif songRepeatTimer.getTimeLeft() <= 0:
                songRepeatTimer.startTimer(STATUS_TRACK_LENGTH)
                play(STATUS_TRACKS[STATUS_CHANGE_INDEX][new_status], "Library")
                delay(0.1)
    
    setVolume(old_volume)




## CALLBACK FUNCTION AND MAIN #############################################################

def callback(in_data, frame_count, time_info, flag):
        
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    audio_data = np.nan_to_num(audio_data)

    audio_data = librosa.resample(audio_data, 44100, 22050)

    for i in audio_data:
        ringBuffer.append(i)
            
    return (in_data, pyaudio.paContinue)


if __name__ == "__main__":
    totalTimer.startTimer(1)
    runApp("iTunes")
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
        setVolume(90)

    play(OT_GOAL_TRACK)

    yes = input("\nCalibrate iTunes volume? Current volume is {} \nEnter 'y' for yes,'q' to quit, enter to continue in default mode \nand any other key for no: ".format(getVolume()))

    if yes == "y":
        quit_ = calibrateVolume()
    elif yes == "q":
        quit_ = True
    else:
        stop()


    while len(ringBuffer.get()) < 22050 and quit_ == False: 1

    if quit_ == False:
        event_type = " "
        if yes != "":
            event_type = input("\nIs this a recap or a full game? \nEnter 'r' for recap, 'g' for full game, 'q' to quit, \nor simply press enter to continue in default mode: ")
        else:
            event_type = ""

        while event_type != "r" and event_type != "g" and event_type != "q" and event_type != "":
            event_type = input("\nInvalid input!\nIs this a recap or a full game? \nEnter 'r' for recap, 'g' for full game, 'q' to quit, or simply press enter to turn off the log timer: ")

        if event_type == "r":
            logTime = 360
        elif event_type == "g":
            logTime = 1500
        elif event_type == "q":
            quit_ = True
        elif event_type == "":
            logTime = 1500*360

    logTimer.startTimer(logTime)

    kbOverload = Counter(0)
    need2save = False
    alerted = True
    i2 = Counter(0)

    ringBuffer3.append(True)   
    for i in range(20):
        ringBuffer3.append(False)    

    if quit_ == False:
        print("Resetting playlist \"New York Islanders Soundboard\"...")
        
        stop()
        volume = getVolume()
        print("volume == {}".format(volume))
        setVolume(0)
        play(OT_GOAL_TRACK, curTime=107)
        trouble = 0

        while getState() == "playing" and getTrack() == OT_GOAL_TRACK and quit_ == False:
            
            if trouble >= 30 and getCurTime() <= 107:
                print("Something's wrong. Try restarting iTunes.")
                restart_iTunes(OT_GOAL_TRACK, cur=107)
                if getState() == "paused":
                    print("Didn't work. Goodbye!")
                    flash_screen(num_loops=2)
                    quit_ = True
                else:
                    trouble = 0
            elif getCurTime() <= 107:
                trouble += 1

        play(WIN_TRACK, curTime=78.4)
        trouble = 0

        while getState() == "playing" and getTrack() == WIN_TRACK and quit_ == False:
            if trouble >= 30 and getCurTime() <= 78.4:
                print("Something's wrong. Try restarting iTunes.")
                restart_iTunes(WIN_TRACK, cur=78.4)
                if getState() == "paused":
                    print("Didn't work. Goodbye!")
                    flash_screen(num_loops=2)
                    quit_ = True
                else:
                    trouble = 0
            elif getCurTime() <= 78.4:
                trouble += 1

        setVolume(volume)
        play(GOAL_TRACK)
        delay(0.5)
        setCurTime(40)
        trouble = 0

        while getState() == "playing" and getTrack() == GOAL_TRACK and quit_ == False:
            if trouble >= 30 and getCurTime() <= 40:
                print("Something's wrong. Try restarting iTunes.")
                restart_iTunes(GOAL_TRACK, cur=40)
                if getState() == "paused":
                    print("Didn't work. Goodbye!")
                    flash_screen(num_loops=2)
                    quit_ = True
                else:
                    trouble = 0
            elif getCurTime() <= 40:
                trouble += 1

        play(QUIET_TRACK)
        while getState() == "playing" and getTrack() == QUIET_TRACK and quit_ == False:
            trouble += 1
            if trouble >= 30:
                print("Something's wrong. Try restarting iTunes.")
                restart_iTunes(QUIET_TRACK)
                if getState() == "paused":
                    print("Didn't work. Goodbye!")
                    flash_screen(num_loops=2)
                    quit_ = True
                else:
                    trouble = 0
            


        stop()
    
        print("\nLet's watch some hockey!!!\n\nOvertime mode: off\nShootout mode: off\nDemo mode: off\nRemote control: off")
        #flash_screen(3, 2)

    kb = kbHitMod.KBHit()
        
    while stream.is_active():

        command = kb.getch()
        kb.off()
        
        if command == "\r":
            keypressed = False
        else: keypressed = True

        if quit_ == True:
            break
        elif (mode == NO_GOAL and pauseTimer.getIsRunning() == False) or loop_thru == True:
            signal = ringBuffer.get()
            signal = np.array(signal, np.float32)
            ringBuffer2.append(signal)
            ringBuffer2a.append(ringBuffer.get_diff())
            
            ringBuffer3.append(loop_thru)

            cooldown_nums = ringBuffer3.get()
            cooled = cooldown_nums.count(True)
            if cooled == 0 and alerted == False:
                print("\nAnd we're back!")
                alerted = True
                loop_thru = False
                cooldownTimer.startTimer(10)
            elif len(cooldown_nums) == 24:
                print("\nAnd we're live!")
                cooldownTimer.startTimer(10)
            elif cooled > 0 and alerted == False and loop_thru == False:
                continue

            if signal.shape[0] == 22050 and (mode == NO_GOAL or loop_thru == True):

                X = librosa.feature.mfcc(signal, sr=22050, n_mfcc=44, hop_length=512, n_fft=2048)
                X = X.T

                X_new = X[np.newaxis, ...]
                prediction = model(X_new) 
                
                if (np.argmax(prediction) != NO_GOAL and cooled == 0) or loop_thru == True:
                    i2.add()
                    if loop_thru == False:
                        mode = np.argmax(prediction)
                        loop_thru = True
                        #print("Goal!")
                        #continue
                    elif i2.get() <= 12: # was 15
                        continue
                    else:
                        i2.reset()
                        loop_thru = False
                        #cooled -= 100
                        
                        
                if mode == GOAL and cooled == 0:
                    print("prediction == {}".format(prediction))
                    #state = getState()
                    #track_name = getTrack()
                    #mode = GOAL
                    alerted = False
                    #cooled += 100

                    #if state == "paused" or ((track_name != GOAL_TRACK and track_name != WIN_TRACK and track_name != OT_GOAL_TRACK) or track_name == QUIET_TRACK):
                    if overtime:
                        pauseTimer.startTimer(108)
                        play(OT_GOAL_TRACK)
                        print("Goal! We win!")
                    elif shootout:
                        print("Shootout goal! We win?")
                        pauseTimer.startTimer(79.44)
                        play(WIN_TRACK)
                    else:
                        print("Goal!")
                        pauseTimer.startTimer(42.353)
                        play(GOAL_TRACK)

                        print("\n")

                        """
                        for i in range(3):
                            signal = ringBuffer.get()
                            signal = np.array(signal, np.float32)
                            ringBuffer2.append(signal)
                            ringBuffer3.append(True)
                            delay(0.1)
                        """
                        continue #uncomment this if you comment the continue when loop_thru is set to true
                            
                elif mode == WIN and cooled == 0:
                    print("prediction == {}".format(prediction))
                    #state = getState()
                    #track_name = getTrack()
                    #mode = WIN
                    alerted = False
                    #cooled += 100

                    #if state == "paused" or ((track_name != GOAL_TRACK and track_name != WIN_TRACK and track_name != OT_GOAL_TRACK) or track_name == QUIET_TRACK):
                    play(WIN_TRACK)
                    pauseTimer.startTimer(79.44)
                    """
                    for i in range(3):
                        signal = ringBuffer.get()
                        signal = np.array(signal, np.float32)
                        ringBuffer2.append(signal)
                        ringBuffer3.append(True)
                        delay(0.1)
                    """
            
                    print("We win!\n")
                    continue #uncomment this if you comment the continue when loop_thru is set to true
                
                elif mode != NO_GOAL and demo == False:
                    data = ringBuffer2.get()
                    timestamps = ringBuffer2a.get()
                    timestamps[-25] = 0
                    while os.path.exists(TRIGGER_POSITIVE + trigger_num.gets()+ "_" + trigger_num2.gets() + ".wav"):
                        trigger_num.add()
                        trigger_num2.reset()

                    with open(TRIGGER_POSITIVE + trigger_num.gets() + "_" + "timestamps.json", "w") as fp0:
                        json.dump(timestamps[-25:], fp0, indent=4)

                    print("Timestamps saved to \n{}\n".format(TRIGGER_POSITIVE + trigger_num.gets() + "_" + "timestamps.json"))

                    for d in data[-25:]:
                        scipy.io.wavfile.write(TRIGGER_POSITIVE + trigger_num.gets()+ "_" + trigger_num2.gets() + ".wav", 22050, d)
                    
                        d = librosa.feature.mfcc(y=d, sr=22050, n_mfcc=44, hop_length=512, n_fft=2048)
                        d = d.T
                        with open(TRIGGER_POSITIVE + trigger_num.gets() + "_" + trigger_num2.gets() + ".json", "w") as fp:
                            json.dump(d.tolist(), fp, indent=4)
                        
                        print("Data saved to \n{} \nand {}\n".format(TRIGGER_POSITIVE + trigger_num.gets() + "_" + trigger_num2.gets() + ".wav", TRIGGER_POSITIVE + trigger_num.gets() + "_" + trigger_num2.gets() + ".json"))
                        
                        trigger_num2.add()

                    trigger_num.add()
                    trigger_num2.reset()

                elif logTimer.getTimeLeft() <= 0 and demo == False and cooldownTimer.getTimeLeft() <= 0 and cooldownTimer.getIsRunning() == True and np.argmax(prediction) == mode == NO_GOAL:
                    logTimeCurrent = logTime + logTimer.getTimeLeft()
                    logTimer.startTimer(logTimeCurrent)
                    data = ringBuffer2.get()
                    while os.path.exists(LOG_FILE + log_num.gets()+ "_" + log_num2.gets() + ".wav"):
                        log_num.add()
                        log_num2.reset()

                    timestamps = ringBuffer2a.get()
                    timestamps[-10] = 0
                
                    with open(LOG_FILE + log_num.gets() + "_" + "timestamps.json", "w") as fp0:
                        json.dump(timestamps[-10:], fp0, indent=4)

                    print("Timestamps saved to \n{}\n".format(LOG_FILE + log_num.gets() + "_" + "timestamps.json"))

                    for d in data[-10:]:
                        scipy.io.wavfile.write(LOG_FILE + log_num.gets()+ "_" + log_num2.gets() + ".wav", 22050, d)
                    
                        d = librosa.feature.mfcc(y=d, sr=22050, n_mfcc=44, hop_length=512, n_fft=2048)
                        d = d.T
                        with open(LOG_FILE + log_num.gets() + "_" + log_num2.gets() + ".json", "w") as fp:
                            json.dump(d.tolist(), fp, indent=4)
                        
                        print("Data saved to \n{} \nand {}\n".format(LOG_FILE + log_num.gets() + "_" + log_num2.gets() + ".wav", LOG_FILE + log_num.gets() + "_" + log_num2.gets() + ".json"))
                        
                        log_num2.add()

                    log_num.add()
                    log_num2.reset()

        state = getState()
        
        if pauseTimer.getIsRunning():
            cooldownTimer.stopTimer()
            if getState() == "paused" or getTrack() == QUIET_TRACK:
                if pauseTimer.getTimeElapsed() <= 3:
                    print("Never mind!")
                
                print("Back to the game!")
                if demo:
                    print("Demo mode: ON")
                else:
                    print("Demo mode: off")

                pauseTimer.stopTimer()
                mode = NO_GOAL
                stop()

                #for i in range(10):
                #ringBuffer3.append(False)

            elif pauseTimer.getTimeLeft() <= 0:
                print("Back to the game!")
                
                if demo:
                    print("Demo mode: ON")
                else:
                    print("Demo mode: off")

                pauseTimer.stopTimer()
                mode = NO_GOAL
                stop()
            elif pauseTimer.getTimeLeft() < 4:
                if mode == WIN or overtime or shootout:
                    quit_ = True     
                    print("Game over! Goodbye!\n")  
                else:
                    ringBuffer3.append(False)
                    ringBuffer3.append(False)
                    ringBuffer3.append(False)
            #else:
                #for i in range(10):
                #ringBuffer3.append(True)
        elif state == "playing" and mode == NO_GOAL:
            print("Oh, did I miss something?")
            smoothTimer = ShutdownTimer()
            smoothTimer.startTimer(3)
            
            if remote == False:
                if overtime and getTrack() != OT_GOAL_TRACK:
                    play(OT_GOAL_TRACK)
                elif shootout and getTrack() != WIN_TRACK:
                    play(WIN_TRACK)
                    smoothTimer.startTimer(3)
                elif getTrack() != GOAL_TRACK:
                    play(GOAL_TRACK)

            while smoothTimer.getTimeLeft() >= 0:
                if getState() == "paused" and remote == False:
                    if getTrack() != WIN_TRACK:
                        play(WIN_TRACK)
                        smoothTimer.startTimer(3)
                    else:
                        mode = NO_GOAL
                        stop()
                        smoothTimer.stopTimer()
                        print("I guess not!")
                        break

            
            if getState() == "playing":
                print("I guess I did!")
                track = getTrack()
                if track == GOAL_TRACK or track == OT_GOAL_TRACK or shootout:
                    mode = GOAL
                    if shootout == True:
                        pauseTimer.startTimer(79.44 - smoothTimer.getTimeElapsed())
                    elif overtime == True:
                        pauseTimer.startTimer(108 - smoothTimer.getTimeElapsed())
                    else:    
                        pauseTimer.startTimer(42.253 - smoothTimer.getTimeElapsed())
                    
                elif track == WIN_TRACK:
                    mode = WIN
                    pauseTimer.startTimer(79.44 - smoothTimer.getTimeElapsed())

                if demo == False:    
                    data = ringBuffer2.get()
                    while os.path.exists(FALSE_NEGATIVE + false_num.gets() + "_" + false_num2.gets() + ".wav"):
                        false_num.add()
                        false_num2.reset()

                    timestamps = ringBuffer2a.get()
                    timestamps[0] = 0
                    
                    with open(FALSE_NEGATIVE + false_num.gets() + "_" + "timestamps.json", "w") as fp0:
                        json.dump(timestamps, fp0, indent=4)

                    print("Timestamps saved to \n{}\n".format(FALSE_NEGATIVE + false_num.gets() + "_" + "timestamps.json"))

                    print("len(data) == {}".format(len(data)))

                    for d in data:
                        scipy.io.wavfile.write(FALSE_NEGATIVE + false_num.gets() + "_" + false_num2.gets() + ".wav", 22050, d)
                        
                        d = librosa.feature.mfcc(y=d, sr=22050, n_mfcc=44, hop_length=512, n_fft=2048)
                        d = d.T
                        with open(FALSE_NEGATIVE + false_num.gets() + "_" + false_num2.gets() + ".json", "w") as fp:
                            json.dump(d.tolist(), fp, indent=4)
                        
                        print("Data saved to \n{} \nand {}\n".format(FALSE_NEGATIVE + false_num.gets() + "_" + false_num2.gets() + ".wav", FALSE_NEGATIVE + false_num.gets() + "_" + false_num2.gets() + ".json"))
                        
                        false_num2.add()
                    
                    false_num2.reset()
                    false_num.add()

                    print("Data saved for manual sorting! \nmode == {}".format(mode_names[mode]))     



        if keypressed == True:
            kbOverload.add()
            if kbOverload.get() >= 2:
                quit_ = True
                print("Error: Keyboard overloaded! Goodbye!")
            elif command == "o":
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
            elif command == "q":
                quit_ = True
                stop()
                print("Quitting... goodbye!\n")
            elif command == "d":
                demo = (not demo)
                if demo == True:
                    print("Demo mode: ON")
                else:
                    print("Demo mode: off")
            elif command == "c":
                quit_ = calibrateVolume()
                kbOverload.reset()
            elif command == "v":
                play(OT_GOAL_TRACK)
                command = input("\nCurrent volume is {}.\nEnter desired volume from 1 to 100, 'c' to cancel or \n'q' to quit the program: ".format(getVolume()))
    
                if is_integer(command) == True:
                    setVolume(int(command))
                    print("\nNew volume is {}.".format(getVolume()))
                    flash_screen(3, 1)
                elif command == "q":
                    quit_ = True
                elif command != "c": 
                    print("\nHuh? Never mind.")
                else:
                    print("\nNever mind.")

                if quit_ == False:
                    delay(3)

                stop()
                kbOverload.reset()
            elif command == "m":
                if logTime == 360:
                    logTime = 1500
                    logTimer.addTime(1140)
                    print("\nGame mode: FULL GAME")
                elif logTime == 360*1500:
                    logTime = 360
                    logTimer.addTime(-360*1499)
                    print("\nGame mode: RECAP")
                elif logTime == 360*1500:
                    logTime = 360*1500
                    logTimer.addTime(359*1500)
                    print("\nGame mode: NONE")


                print("logTimer reads {} seconds elapsed".format(logTimer.getTimeElapsed()))
                print("logTimer reads T-{}".format(logTimer.getTimeLeft()))
            elif command == "p":
                muteTime = time.time()
                state = getState()
                track = getTrack()
                position = pauseTimer.getTimeElapsed()

                if pauseTimer.getIsRunning():
                    pauseTimer.pauseTimer()
                if cooldownTimer.getIsRunning():
                    cooldownTimer.pauseTimer()
                logTimer.pauseTimer()

                if state == "playing":
                    position = getCurTime()
                    pause()


                command = input("Enter 'q' to quit, or any other key to resume: ")

                muteTimer = time.time() - muteTime
                total_pause_time += int(muteTimer)

                if command == "q":
                    quit_ = True
                    print("\nYou were paused for {}.\n".format(convert_time(muteTimer, shorten=True)))
                    break
                else:
                    print("\nYou were paused for {}.\nTotal pause time: {}\n".format(convert_time(muteTimer, shorten=True), convert_time(total_pause_time, shorten=True)))
                    if pauseTimer.getIsPaused():
                        pauseTimer.resumeTimer()
                    if cooldownTimer.getIsPaused():
                        cooldownTimer.resumeTimer()
                    logTimer.resumeTimer()

                if state == "playing" or track != getTrack():
                    play(track, curTime=position)
                elif state == "playing":
                    play()
                    if position != getCurTime() and position != 0:
                        play(curTime=position)
                else:
                    print("If you can't hear this, check your speakers.")
                    play(GOAL_TRACK)
                    delay(0.9)
                    play(QUIET_TRACK)
                    stop()

                ghost = True
                while getState() == "playing" and pauseTimer.getTimeLeft() > 0: 
                    if pauseTimer.getTimeLeft() > 4 or not (mode == WIN or ((shootout or overtime) and mode == GOAL)):
                        ghost = False
                        break

                printStatus(overtime, shootout, demo, remote)
                if ghost == True:
                    print("\nlogTimer reads T-{}".format(math.floor(logTimer.getTimeLeft())))

                    if logTime <= 300: print("game_type == RECAP")
                    else: print("game_type == FULL GAME")
                
            elif command == "l":
                print("Final orders:")
                final_orders = input()
                if final_orders == "play taps" or final_orders == "Play taps":
                    print("Playing \"Taps\"..."); play("Islanders Taps", "Library")

                    while getState() == "playing" and getTrack() == "Islanders Taps": 1

                    quit_ = True; print("(sigh)"); stop()
                    pauseTimer.startTimer(4)
                    break
                else:
                    print("Never mind! We haven't lost yet!")
                    continue
            elif command == "r":
                remote = (not remote)
                if remote == True:
                    print("\nRemote control: ON\n")
                else:
                    print("\nRemote control: off\n")
            elif command == " ":
                print("prediction == {}".format(prediction))
        else:
            kbOverload.reset(0)  
            
        
    stream.close()
    pa.terminate()

    while pauseTimer.getIsRunning() == True and pauseTimer.getTimeLeft() > 3 and getState() == "playing": 1

    if pauseTimer.getTimeLeft() <= 3 and pauseTimer.getIsRunning() == True and (getTrack() == WIN_TRACK or getTrack() == OT_GOAL_TRACK):
        
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



