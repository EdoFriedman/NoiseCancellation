import os
import pathlib
import utils
from scipy import fft
from scipy.io import wavfile
import numpy as np

SPEECH_FREQUENCIES = np.sqrt(np.load("speech_frequencies.npy"))

last_heavy_coeffs = None

WINDOWS_PER_SECOND = 20


def rate_importance(freq, freq_amp, important_frequencies, amp_variance, removed_frequencies):
    """
    Rates the importance of the frequency in the audio clip.
    Returns a float in (∞,1]
    """
    common_speech_frequency_score = SPEECH_FREQUENCIES[round(freq)] if freq < len(SPEECH_FREQUENCIES) else 0
    importance_influencers = [freq_amp ** 2 / amp_variance, common_speech_frequency_score]
    weights = [2, 2]
    if utils.is_overtone(freq, important_frequencies, 1.e-2):#np.count_nonzero(freq % important_frequencies[:important_frequencies_count]) == important_frequencies_count:
        # importance_influencers.append(1)
        # weights.append(2)
        return 1
    elif utils.is_overtone(freq, removed_frequencies, 0):
        importance_influencers.append(0)
        weights.append(1)
    # if last_heavy_coeffs is not None and (
    #         freq in last_heavy_coeffs or freq - WINDOWS_PER_SECOND in last_heavy_coeffs or freq + WINDOWS_PER_SECOND in last_heavy_coeffs):
    #     importance_influencers.append(1)
    #     weights.append(0.5)
    res = np.average(importance_influencers, weights=weights)

    # if 7000 < freq < 7400:
    #     print(res > 0.2, freq_amp, importance_influencers, weights)
    return res


def remove_background_noise_from_window(freq_domain_audio):
    global last_heavy_coeffs
    important_frequencies = np.full(len(freq_domain_audio) // 2,
                                    len(freq_domain_audio) * WINDOWS_PER_SECOND)
    important_freq_count = 0
    removed_frequencies = np.full(len(freq_domain_audio), len(freq_domain_audio) * WINDOWS_PER_SECOND)
    removed_freq_count = 0
    heavy_coeffs = set()
    freq_domain_audio_absolute = np.abs(freq_domain_audio)
    amp_variance = np.sum(freq_domain_audio_absolute ** 2)
    for freq in range(len(freq_domain_audio)):
        freq_in_hz = freq * WINDOWS_PER_SECOND
        amplitude = freq_domain_audio_absolute[freq]
        importance_rating = rate_importance(freq_in_hz, amplitude, important_frequencies[:important_freq_count],
                                            amp_variance, removed_frequencies)
        if importance_rating < 0.5:
            freq_domain_audio[freq] = 0
            if amplitude > 0.01 * amp_variance:
                removed_frequencies[removed_freq_count] = freq_in_hz
                removed_freq_count += 1
        else:
            heavy_coeffs.add(freq_in_hz)
            if freq_in_hz >= 220 and important_freq_count < len(important_frequencies):
                important_frequencies[important_freq_count] = freq_in_hz
                important_freq_count += 1
    last_heavy_coeffs = heavy_coeffs


def write_to_wav(data, sample_rate, file_path):
    wavfile.write(file_path, sample_rate, data)


def remove_noise(file_path):
    sample_rate, data = wavfile.read(file_path)
    if len(data.shape) == 2:
        data = data[:, 0]
    new_data = np.zeros(data.shape)
    generator = utils.fft_generator(data, sample_rate, WINDOWS_PER_SECOND)
    for freq_domain_audio, window_start, window_end in generator:
        remove_background_noise_from_window(freq_domain_audio)
        irfft_result = fft.irfft(freq_domain_audio)
        new_data[window_start:window_end] = irfft_result
    new_data *= 2 ** 15
    return sample_rate, new_data.astype(np.int16)


def main():
    # audio_file = "Audio Clip With Background Noise.wav"
    # audio_file = "MS-SNSD/NoisySpeech_training/noisy62_SNRdb_40.0_clnsp62.wav"
    audio_files = ["MS-SNSD/NoisySpeech_training/noisy61_SNRdb_20.0_clnsp61.wav",
                   "MS-SNSD/NoisySpeech_training/noisy62_SNRdb_40.0_clnsp62.wav",
                   "MS-SNSD/NoisySpeech_training/noisy63_SNRdb_40.0_clnsp63.wav"]
    # audio_file = "MS-SNSD/CleanSpeech_training/clnsp62.wav"
    start, end = 0, 10
    # out_audio_file = "Audio Clip (hopefully) Without Background Noise.wav"
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    elif not os.path.isdir(output_dir):
        print(f"{output_dir} already exists and is not a directory")
        return
    for file in audio_files:
        file_path = pathlib.Path(file)
        wavfile.write(f"{output_dir}/{file_path.name}", *remove_noise(file))
        print(f'Saved file to "{output_dir}/{file_path.name}"')


if __name__ == '__main__':
    main()
