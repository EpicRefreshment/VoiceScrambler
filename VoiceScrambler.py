#################################################################################################################################
# Group 4 Voice Scrambler
# Author: Jonathan Wolford
# Date Created: 03/11/2024
# Data Last Modified: 03/18/2024
# PUT DESCRIPTION HERE
## Permutation Change Mode
### Parameter: -mode, -m
### Description: Change permutation exactly at specified rate = 0,
###              Only change permutation during pauses in speech = 1,
###              Change permutation at specified rate adjusting for pauses in speech if possible = 2
### Default: 0
## Input Speech File
### Parameter: -file, -f
### Description: Name of speech audio file (.wav) to scramble and descramble
### Default: speech_test.wav
## Permutation Rate Change
### Parameter: -rate, -r
### Description: Rate of permutation change in fractional steps
###              Ex. 1/2 is 1 change every 2 seconds (max rate: 2/1)
### Default: 0/0
## Number of Bands
### Parameter: -band, -b
### Description: Splits the speech signal's spectrum by specified number of bands
### Default: 0
## Debug
### Parameter: -debug, -d
### Description: Off - 0, On - 1 (Outputs debug print statements to console)
### Default: 0
#################################################################################################################################

import time
import signal
import sys
import functools
import struct
import enum
import argparse

from scipy import signal as sg
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from ScramblerHelper import *

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m','--mode', type=int, default=0)
    parser.add_argument('-f','--file', type=str, default='speech_test.wav')
    parser.add_argument('-r','--rate', type=str, default='0/0')
    parser.add_argument('-b','--band', type=int, default=1)
    parser.add_argument('-d','--debug', type=int, default=0)

    args = parser.parse_args()
    return args

def debug_print(msg, debug):
    if debug:
        print(msg)

def read_wav_file(file):
    # Read wav file
    Fs, x = wavfile.read(file)
    Ts = 1 / Fs
    duration = len(x) / Fs
    t = np.linspace(0,duration,len(x))

    return [Fs, Ts, duration, x, t]

def write_wav_file(signal, file, Fs, invert_flag):
    if (invert_flag):
        # strip '.wav' from input file, add 'scrambled.wav'
        wavfile_out_name = file[:-4] + 'scrambled.wav'
    else:
        # strip '.wav' from input file, add 'descrambled.wav'
        wavfile_out_name = file[:-4] + 'descrambled.wav'
        
    # normalize signal for output
    m = np.max(signal)
    signalf32 = (signal/m).astype(np.float32)
    
    wavfile.write(wavfile_out_name, Fs, signalf32)

def determine_rate(rate_fractional, duration):
    # This returns the rate in amount of seconds expected between change
    num_changes = int(rate_fractional.split('/')[0])
    seconds = int(rate_fractional.split('/')[1])

    # sanity check for zeros
    if (num_changes == 0 or seconds == 0):
        return 0

    # check boundaries
    rate = seconds / num_changes
    if (rate < 0.5):
        print("Invalid rate. Rate must not exceed 2 permutation changes per second.")
        return -1
    elif (rate >= duration):
        print("Invalid rate. Rate must either be 0 or change at least once within duration of speech.")
        return -1
    else:
        return rate

def determine_num_permutations(rate, duration):
    # This determines the number of permutation changes expected in the signal
    # based on the given rate and duration of the signal
    if (rate == 0):
        return 1
    else:
        num_perm = duration / rate
        num_perm = int(num_perm) # we want an integer, not a float

        return num_perm

def check_mode(mode):
    if (mode < 0 or mode > 2):
        print("Invalide mode! Valid Options = 0, 1, 2")
        return -1
    return 0
    
        
def check_bands(bands):
    # check for edge conditions related to number of bands to split the spectrum
    # Any other boundaries to check? I assume there's a maximum we want to have based
    # on input
    if (bands < 0):
        return -1
    else:
        return 0    

def scrambler(mode, filename, speech_parameters, rate, bands, debug):
    debug_print("Application starting...\n", debug)

    Fs = speech_parameters[0]
    Ts = speech_parameters[1]
    duration = speech_parameters[2]
    speech_original = speech_parameters[3]
    time_vector = speech_parameters[4]

    # For debug sanity check
    if (mode == 0):
        mode_string = "Direct"
    elif (mode == 1):
        mode_string = "Gap Detection"
    elif (mode == 2):
        mode_string = "Combo"
    debug_str = "Speech Paramaters:\nSample Rate = {} Hz\nSample Period = {:4f} seconds\nSpeech duration = {} seconds\n".format(Fs, Ts, duration)
    debug_print(debug_str, debug)
    debug_str = "Voice Scrambler Paramaters:\nPermutation Change Rate = {:4f} seconds\n# of Bands to Split = {} bands\nmode = {}\n".format(rate, bands, mode_string)
    debug_print(debug_str, debug)

    # Remove any DC offset in the signal
    speech_dc_removed = remove_dc(speech_original)
    debug_str = "DC offset removed from signal.\n"
    debug_print(debug_str, debug)

    # Split signal in time to vary permutations
    # TO DO! use num_perms to split signal in time and generate the necessary amount of carriers
    # Will eventually work with each defined mode.
    # Make sure to keep track of where the signal was split for descrambling
    num_perms = determine_num_permutations(rate, duration)
    if (mode == 0):
        # Just change it at the defined rate exactly
        pass
    elif (mode == 1):
        # Just split based on gap detection. Ignore user provided rate
        pass
    elif (mode == 2):
        # Split signal based on user defined rate, but use gap detection within
        # the bounds of given permutation change rate
        # Ex. For a give rate of 2/1 (2 changes per second), attempt to find 2 gaps per second
        # in the signal and make the split there. If not, either try the next best spot according
        # to gap detection algorithm or simply just choose a spot to change.
        
        # This could potentially be updated to interpret "nominal, permissible rate"
        # as change if there's enough gaps in time window, but do not change permutation if not.
        pass

    debug_str = "Signal split in time based on given rate.\n"
    debug_print(debug_str, debug)
    
    # Bandsplit speech
    # NOTE: this will change (likely a for loop) based on immediately preceding code
    if (bands > 1):
        speech_band_split = bandsplit(speech_dc_removed, bands, Fs)
    else:
        speech_band_split = [speech_dc_removed] # no split, no change

    debug_str = "Signal split into requested number of bands.\n"
    debug_print(debug_str, debug)

    # SCRAMBLE!
    # generate carrier frequencies and associated low pass filters
    carriers = generate_carriers(bands, time_vector)
    lpfs = generate_lpfilters(bands, Fs)

    debug_str = "Carriers and associated low pass filters generated.\n"
    debug_print(debug_str, debug)

    speech_inverted_bands = []
    for i in range(0, len(speech_band_split)):
        speech_inverted_bands.append(invert(speech_band_split[i], carriers[i], lpfs[i]))
    
    if (bands > 1):
        # add bands back together
        pass
    else:
        speech_scrambled = speech_inverted_bands[0]

    debug_str = "Signal scrambled via voice inversion.\n"
    debug_print(debug_str, debug)

    write_wav_file(speech_scrambled, filename, Fs, True)

    debug_str = "Scrambled signal output to wav file.\n"
    debug_print(debug_str, debug)

    # DESCRAMBLE!
    # Do everything again

    # Remove any DC offset in the signal
    speech_dc_removed = remove_dc(speech_scrambled)

    ### Split signal in time at same points as before regardless of mode ###

    # Bandsplit scrambled speech in same manner as before
    # NOTE: this will change (likely a for loop) based on immediately preceding code
    if (bands > 1):
        speech_band_split = bandsplit(speech_dc_removed, bands, Fs)
    else:
        speech_band_split = [speech_dc_removed] # no split, no change
        
    speech_deinverted_bands = []
    for i in range(0, len(speech_band_split)):
        speech_deinverted_bands.append(invert(speech_band_split[i], carriers[i], lpfs[i]))

    if (bands > 1):
        # add bands back together
        pass
    else:
        speech_descrambled = speech_deinverted_bands[0]

    debug_str = "Signal descrambled via voice inversion.\n"
    debug_print(debug_str, debug)

    write_wav_file(speech_descrambled, filename, Fs, False)

    debug_str = "Descrambled signal output to wav file.\n"
    debug_print(debug_str, debug)

def main():
    # Arguments are parsed and handled in main then handed off to scrambler
    args = parse_arguments()

    speech_parameters = read_wav_file(args.file)
    rate = determine_rate(args.rate, speech_parameters[2])

    band_check = check_bands(args.band)
    mode_check = check_mode(args.mode)

    if (rate < 0 or band_check < 0 or mode_check < 0):
        print("Argument error detected. Exiting...")
        return 0

    debug = bool(args.debug)

    scrambler(args.mode, args.file, speech_parameters, rate, args.band, debug)

if __name__ == "__main__":
    main()
