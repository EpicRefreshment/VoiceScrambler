from scipy import signal as sg
import numpy as np
import matplotlib.pyplot as plt

standard_carriers = [2632, 2718, 2868, 3023, 3196, 3339, 3495, 3729]

def getfilts(n, fs):
    
    filts = []
    s = 1   
    step = int((fs / 2) / n)
    
    for b in range(n):
        up = ((b+1) * step)
        f = sg.cheby1(5, 4, (s, up - 1), 'bandpass', output='sos', fs=fs)
        filts.append(f)
        s = up
        
    return filts

def specshow(D, hl, sr):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)


    lb.display.specshow(lb.amplitude_to_db(D, ref=np.max(D)),y_axis='log', sr=sr, hop_length=hl,
                             x_axis='time', ax=ax)

    ax.set(title='Log-frequency power spectrogram')
    ax.label_outer()

def bandspec(xs, hl):
    
    n = len(xs)
    step = int(hl/n)
    
    f = [(i*step, ((i+1) * step) - 1) for i in range(n)]
    bands = [spectrum(x, hl)[f[i][0]:f[i][1]] for i, x in enumerate(xs)]
    
    return bands
    
def spectrum(x, hl):
    
    return np.abs(lb.stft(x, hop_length=hl))

def remove_dc(signal):
    signal = signal - np.mean(signal)

    return signal
    
def lpfilter(cutoff, fs):
    f = sg.cheby1(5, 4, cutoff, 'lowpass', output='sos', fs=fs)
    
    return f
    
def invert(signal, carrier, filt):

    prefilter = sg.sosfilt(filt, signal)
    inv_signal = signal * carrier
    postfilter = sg.sosfilt(filt, inv_signal)
    
    return postfilter

def carrier(f, t):
    
    return np.sin(f * (2 * np.pi) * t)

def generate_carriers(num_bands, t):
    carriers = []
    for i in range(0, num_bands):
        n = i % len(standard_carriers)
        carriers.append(carrier(standard_carriers[n], t))

    return carriers

def generate_lpfilters(num_bands, fs):
    lpfilters = []
    for i in range(0, num_bands):
        n = i % len(standard_carriers)
        lpfilters.append(lpfilter(standard_carriers[n], fs))

    return lpfilters

def digfreq(f, fs):
    return (2 * np.pi * f) / fs

def bandsplit(signal, num_bands, fs):
    
    filts = getfilts(num_bands, fs)    
    bands = [sg.sosfilt(f, signal) for f in filts]    
    bands = bandspec(bands, 1024)
    
    return bands

def shuffle(bands, book):
    
    return [bands[i] for i in book]

def bandunsplit(bands):
    
    return np.vstack(bands)
