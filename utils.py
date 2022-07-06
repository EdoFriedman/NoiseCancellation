import numpy as np
from scipy import fft
import soundfile
import glob
from matplotlib import pyplot as plt
import pathlib


def create_freq_graph(sound_files_root="LibriSpeech/dev-clean", file_ext="flac"):  # Data from http://www.openslr.org/12
    """Create a graph showing the frequency distribution of human speech
    We can then use this data to remove any sound with a frequency outside the range we find"""
    frequencies = np.zeros(8000)
    counter = 0
    for path in glob.iglob(f"{sound_files_root}/**/*.{file_ext}", recursive=True):
        audio, sample_rate = soundfile.read(path)
        time_range_length = sample_rate // 20
        if counter >= 500:
            break
        counter += 1
        print(f"Files sampled: {counter}")
        resolved_path = pathlib.Path(path).resolve()
        for fft_result, frame_start, frame_end in fft_generator(audio, sample_rate):
            fft_result = np.abs(fft_result)
            # fft_result *= 1000
            for freq, amplitude in enumerate(fft_result):
                if amplitude > 0:
                    freq_hz = freq * sample_rate // time_range_length
                    # Uncomment to print the names of files where there are high frequencies,
                    # can help us test edge cases later
                    # if freq_hz > 900 and amplitude > 1:
                    #     print(
                    #         f"frequency: {freq_hz}\t"
                    #         f"file path: \"{resolved_path}\"\t"
                    #         f"time with high frequency sound: {frame_start / sample_rate}")
                    if freq_hz < len(frequencies):
                        frequencies[freq_hz] += amplitude
    plt.plot(frequencies)
    plt.show()


def create_freq_data(sound_files_root="LibriSpeech/dev-clean"):  # Data from http://www.openslr.org/12
    """Create a graph showing the frequency distribution of human speech
    We can then use this data to remove any sound with a frequency outside the range we find"""
    frequencies = np.zeros(8000)
    counter = 0
    for path in glob.iglob(sound_files_root + "/**/*.flac", recursive=True):
        audio, sample_rate = soundfile.read(path)
        time_range_length = sample_rate // 20
        if counter >= 500:
            break
        counter += 1
        print(f"Files sampled: {counter}")
        resolved_path = pathlib.Path(path).resolve()
        for fft_result, frame_start, frame_end in fft_generator(audio, sample_rate):
            fft_result = np.abs(fft_result)
            # fft_result *= 1000
            for freq, amplitude in enumerate(fft_result):
                if amplitude > 0:
                    freq_hz = freq * sample_rate // time_range_length
                    # Uncomment to print the names of files where there are high frequencies,
                    # can help us test edge cases later
                    # if freq_hz > 900 and amplitude > 1:
                    #     print(
                    #         f"frequency: {freq_hz}\t"
                    #         f"file path: \"{resolved_path}\"\t"
                    #         f"time with high frequency sound: {frame_start / sample_rate}")
                    if freq_hz < len(frequencies):
                        frequencies[freq_hz] += amplitude
    frequencies /= np.max(frequencies)
    np.save("speech_frequencies.npy", frequencies)


def fft_generator(data, sample_rate, frames_per_second=20):
    """Returns a generator that slices the audio clip to frames and yields the fft result of the frame"""
    data = data / 2 ** 15
    time_range_length = sample_rate // frames_per_second  # time in samples
    x = range(0, len(data) + 1, time_range_length)

    for i in range(len(x) - 1):
        fft_result = fft.rfft(data[x[i]:x[i + 1]])
        yield fft_result, x[i], x[i + 1]


def is_overtone(freq, frequencies, tol=None):
    """
    :param freq: frequency to check
    :param frequencies: frequencies freq might be an overtone of
    :param tol: tolerance
    :return: True if and only if freq is an overtone of any frequency in important_frequencies
    """
    tol_arr = frequencies * tol
    return (np.logical_and(freq % frequencies <= tol_arr, freq // frequencies >= 2)).any()


# For guesser

def calculate_reduced_signal_norm(data, old_data):
    reduced_signal_arr = data - old_data
    reduced_signal_arr **= 2
    return reduced_signal_arr.sum() ** 0.5


def create_tone_list():
    current_tone = 27.5
    tones = []
    while current_tone < 7903:
        tones.append(current_tone)
        current_tone *= 2 ** (1 / 48)
    return tones


if __name__ == '__main__':
    create_freq_data()
    # create_freq_graph("MS-SNSD/CleanSpeech_training","wav")
    # create_freq_graph()
