<<<<<<< HEAD:cnn_classifier_example/chirp.py
import numpy as np 
import numpy.matlib as mat
import math
from numpy.core.function_base import linspace

def linearChirp(init_freq, fin_freq, T, x):
    f0 = init_freq
    f1 = fin_freq
    c = (f1-f0)/T
    y = np.sin(2*np.pi*(c/2 * x**2 + f0*x))
    ft = c*x+f0
    return y,ft


def geometricChirp(init_freq,fin_freq,T,x):
    f0 = init_freq
    f1 = fin_freq
    k = (f1/f0)**(1/T)
    ft_geo = f0*k**x
    frac = (k**x - 1)/np.log(k)
    y_geo = np.sin(2*np.pi*f0*(frac))
    return y_geo, ft_geo

def hyperbolicChirp(init_freq, fin_freq, T,x):
    f0 = init_freq
    f1 = fin_freq
    ft_hyper = (f0*f1*T)/((f0-f1)*x+f1*T)
    log_term = np.log(1-((f1-f0)/f1*T)*x)
    coeff = 2*np.pi*(-f0*f1*T)/(f1-f0)
    y_hyper = np.sin(coeff*log_term)
    return y_hyper, ft_hyper


def sinusoidalMod(carrier_freq, mod_freq,x):
    y = np.sin(carrier_freq*x)*np.sin(mod_freq*x)
    ft = np.sin(carrier_freq*x)
    return y, ft

def BPSK(d,L):
    N = len(d)
    a = np.zeros(len(d))
    
    for i in range(len(a)):
        a[i] = 2*d[i]-1
    
    out = mat.repmat(a,1,L)
    out = np.transpose(out[:])
    t = linspace(0,N*L-1)
    return out,t

def dft(ys):
    PI2 = math.pi * 2
    i = complex(0, 1)
    N = len(ys)
    ts = np.arange(N) / N
    freqs = np.arange(N)
    args = np.outer(ts, freqs)
    M = np.exp(i * PI2 * args)
    amps = M.conj().transpose().dot(ys)
    return amps


def idft(amps,N):
    ys = dft(amps) / N
    return ys

def BPSK(data,bitPW,freq,samples_per_bit = 1000):
    # bitPW: Length of pulse that denotes 1 bit in ms
    N = len(data)
    t = linspace(0,N+1,N*samples_per_bit)
    out = np.zeros(len(t))
  
    for i in range(len(data)):
        j = 0
        while j < bitPW:
            #if data is a 1 then we output a 0 phase
            if data[i] == 1:
                # fill output with 0 phase signal 
                out[j] = np.sin(2*np.pi*freq*t[j]+0)
            #if data is not 1 then it is a 0
            else: 
                #fill output with 180 phase signal
                out[j] = np.sin(2*np.pi*freq*t[j]+180)
            j = j + 1
    

    return out

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
=======
import numpy as np 
import numpy.matlib as mat
import math
from numpy.core.function_base import linspace

def linearChirp(init_freq, fin_freq, T, x):
    """
    Creates Linear Chrip signal 
    
    Parameters
    ----------
    init_freq : int
        starting frequency 
        
    fin_freq : int
        ending frequency
        
    T : int 
        time it takes to go from init_freq to fin_freq 
       
    x : numpy array 
        linearly spaced numpy array time vector 
        
    Returns
    -------
    numpy array of signal, numpy array of frequency
    """
    
    f0 = init_freq
    f1 = fin_freq
    c = (f1-f0)/T
    y = np.sin(2*np.pi*(c/2 * x**2 + f0*x))
    ft = c*x+f0
    return y,ft


def geometricChirp(init_freq,fin_freq,T,x):
    """
    Creates Geometric Chrip signal 
    
    Parameters
    ----------
    init_freq : int
        starting frequency 
        
    fin_freq : int
        ending frequency
        
    T : int 
        time it takes to go from init_freq to fin_freq 
       
    x : numpy array 
        linearly spaced numpy array time vector 
        
    Returns
    -------
    numpy array of signal, numpy array of frequency
    """
        
    f0 = init_freq
    f1 = fin_freq
    k = (f1/f0)**(1/T)
    ft_geo = f0*k**x
    frac = (k**x - 1)/np.log(k)
    y_geo = np.sin(2*np.pi*f0*(frac))
    return y_geo, ft_geo


def sinusoidalMod(carrier_freq, mod_freq,x):
    """
    Creates Sinusoidal Modulated signal 
    
    Parameters
    ----------
    carrier_freq : int
        carrier frequency 
        
    mod_freq : int
        modulated frequency
     
    x : numpy array 
        linearly spaced numpy array time vector 
        
    Returns
    -------
    numpy array of signal, numpy array of frequency
    """
    y = np.sin(carrier_freq*x)*np.sin(mod_freq*x)
    ft = np.sin(carrier_freq*x)
    return y, ft

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
        
        
        
if __name__ == "linearChirp":
    linearChirp()

if __name__ == "geometricChirp":
    geometricChirp()
    
if __name__ == "sinusoidalMod":
    sinusoidalMod()
if __name__ == "printProgressBar":
    printProgressBar()
    
   
>>>>>>> main:src/chirp.py
