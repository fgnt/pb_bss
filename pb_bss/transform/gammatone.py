import numpy as np
import math as math
from scipy.signal import lfilter


def gammatone_filterbank(signal: np.ndarray, sample_rate: int = 16000, n: int = 23, low_freq: int = 125, high_freq: int = 0):
    """
    Implements a gammetone filterbank with arbitrary amount of gammatone filters and frequency range
    :param signal: input signal
    :param sample_rate: sample rate of the input signal
    :param n: number of gammatone filters
    :param low_freq: lowest center frequency (of the first gammatone filter)
    :param high_freq: highest center frequency (for the n+1 gammatone filter, which is not constructed here!)
           default: sample_rate/2
    :return: list with n-entries each contains the signal filtered by the corresponding gammatone filter
    """
    if(high_freq == 0):
        high_freq = sample_rate/2

    cfs = calculate_cfs(low_freq, high_freq, n)
    A0,A11,A12,A13,A14,A2,B0,B1,B2,gain = _calculate_coefficients(cfs,sample_rate,n);

    ret = []
    for i in range(n):
        y1 = lfilter([A0 / gain[i], A11[i] / gain[i], A2 / gain[i]], [B0, B1[i], B2[i]], signal)
        y2 = lfilter([A0, A12[i], A2], [B0, B1[i], B2[i]], y1)
        y3 = lfilter([A0, A13[i], A2], [B0, B1[i], B2[i]], y2)
        y4 = lfilter([A0, A14[i], A2], [B0, B1[i], B2[i]], y3)
        ret.append(y4)
    return ret


def calculate_cfs(low_f: float, high_f: float, n: int) -> np.ndarray:
    """
    Calculate n center frequencies that are linear distributed on the ERBS scale between low_f and high_f.
    Important: high_f is not returned just the frequency below it
    :param low_f: low frequency in Hz
    :param high_f: high frequency in Hz
    :param n: Number of required cfs
    :return: numpy ndarray with the calculated cfs
    """
    ret = np.ndarray((n,))
    
    low_f = Hz_2_ERBS(low_f)
    high_f = Hz_2_ERBS(high_f)
    step_size = (high_f-low_f)/n

    for i in range(n):
        ret[i] = ERBS_2_Hz(low_f+i*step_size)
    return ret


def Hz_2_ERBS(f: float) -> float:
    return 21.4*math.log(0.00437*f+1, 10)


def ERBS_2_Hz(f: float) -> float:
    return (10**(f/21.4)-1)/0.00437


def _calculate_coefficients(cfs: np.ndarray, sample_rate: int, n: int):
    """Calculation of the coefficients that are needed for the gammatone filter.
    Based on the paper: Apple TR #35 (https://engineering.purdue.edu/~malcolm/apple/tr35/PattersonsEar.pdf)

    :param cfs: center frequencies of the filters
    :param sample_rate:  sample rate of the signal
    :param n: number of gammatone filters
    :return: Tupel of the coefficients for all n gammatone filters, each element of the returned tupel is a list with n elements
    """
    EarQ = 9.26449
    minBW = 24.7
    
    T = 1/sample_rate
    ERB = cfs/EarQ+minBW
    B = 1.019*2*math.pi*ERB
    
    cos_1 = T*np.divide(np.cos(2*cfs*math.pi*T), np.exp(B*T))
    sin_1 = T*np.divide(np.sin(2*cfs*math.pi*T), np.exp(B*T))
    
    A0 = T
    A2 = 0
    B0 = 1
    B1 = -2*np.divide(np.cos(2*cfs*math.pi*T), np.exp(B*T))
    B2 = np.exp(-2*B*T)
    
    A11 = -(cos_1 + (3+2**1.5)**0.5*sin_1)
    A12 = -(cos_1 - (3+2**1.5)**0.5*sin_1)
    A13 = -(cos_1 + (3-2**1.5)**0.5*sin_1)
    A14 = -(cos_1 - (3-2**1.5)**0.5*sin_1)
    
    cos_2 = np.cos(2*cfs*math.pi*T)
    sin_2 = np.sin(2*cfs*math.pi*T)
    
    c_1 = -2*np.exp(4j*cfs*math.pi*T)*T
    c_2 = 2*np.exp(-1*B*T+2j*cfs*math.pi*T)*T
    
    dividend = (c_1+c_2*(cos_2-(3-2**1.5)**0.5*sin_2))*(c_1+c_2*(cos_2+(3-2**1.5)**0.5*sin_2))*(c_1+c_2*(cos_2-(3+2**1.5)**0.5*sin_2))*(c_1+c_2*(cos_2+(3+2**1.5)**0.5*sin_2))
    divisor = np.power((-2/np.exp(2*B*T) - 2*np.exp(4j*cfs*math.pi*T)+np.divide(2*(1 + np.exp(4j*cfs*math.pi*T)),np.exp(B*T))),4)
    
    gain = np.abs(np.divide(dividend,divisor))
    
    return A0,A11,A12,A13,A14,A2,B0,B1,B2,gain
