"""Gaddy's EMG front-end functions, copied VERBATIM (by program, asserted
byte-identical) from silent_speech/read_emg.py (github.com/dgaddy/silent_speech):
  notch: lines 31-33
  notch_harmonics: lines 35-38
  remove_drift: lines 27-29
  subsample: lines 40-44
  apply_to_all: lines 46-50
so they can be imported without that module's absl/textgrids dependencies.
Do not edit the function bodies."""
import numpy as np
import scipy.signal


def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)


def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1,8):
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal


def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal))/old_freq
    sample_times = np.arange(0, times[-1], 1/new_freq)
    result = np.interp(sample_times, times, signal)
    return result


def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:,i], *args, **kwargs))
    return np.stack(results, 1)
