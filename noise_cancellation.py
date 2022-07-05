import utils
from scipy import fft
from scipy.io import wavfile
import numpy as np

SPEECH_FREQUENCIES = np.load("speech_frequencies.npy")

last_heavy_coeffs = None

FRAMES_PER_SECOND = 20


def dist_from_avg_calc(num, avg, variance):
    """Don't ask how did I come up with this function"""
    if variance == 0:
        return 1 if num == avg else 0
    res = -((num - avg) / (5 * variance)) ** 2 + 1
    # if res < 0:
    #     return 0
    return res


def rate_importance(freq, freq_amp, important_frequencies, important_frequencies_count, amp_variance, amp_average):
    """Rates the importance of the frequency in the audio clip.
    Returns a float in (∞,1]"""
    common_speech_frequency_score = SPEECH_FREQUENCIES[round(freq)] if freq < len(SPEECH_FREQUENCIES) else 0
    importance_influencers = [dist_from_avg_calc(freq_amp, amp_average, amp_variance), common_speech_frequency_score]
    # print(amp_variance / abs(freq_amp - amp_average))
    weights = [2.5, 2.5]
    if np.count_nonzero(freq % important_frequencies[:important_frequencies_count]) == important_frequencies_count:
        #  check if the sound is an overtone of another important sound
        importance_influencers.append(1)
        weights.append(2)
    if last_heavy_coeffs is not None and freq in last_heavy_coeffs:
        importance_influencers.append(1)
        weights.append(0.5)
    return np.average(importance_influencers, weights=weights)


def remove_background_noise_from_frame_using_freq(freq_domain_audio):
    global last_heavy_coeffs
    important_frequencies = np.full(len(freq_domain_audio) // 2,
                                    len(freq_domain_audio) * FRAMES_PER_SECOND)
    important_freq_count = 0
    heavy_coeffs = set()
    freq_domain_audio_absolute = np.abs(freq_domain_audio)
    freq_domain_audio_partial = freq_domain_audio_absolute[freq_domain_audio_absolute > 7]
    amp_average = np.mean(freq_domain_audio_partial)
    amp_variance = utils.calculate_variance(freq_domain_audio_partial, amp_average)
    if np.isnan(amp_average):
        amp_average = amp_variance = np.inf
    for freq in range(len(freq_domain_audio)):
        freq_in_hz = freq * FRAMES_PER_SECOND
        amplitude = freq_domain_audio_absolute[freq]
        if rate_importance(freq_in_hz, amplitude, important_frequencies, important_freq_count,
                           amp_variance, amp_average) < 0:
            if amplitude > 10:
                print(freq_in_hz, amplitude, amp_average, amp_variance, freq_domain_audio_partial.shape,
                      rate_importance(freq_in_hz, amplitude, important_frequencies,
                                      important_freq_count,
                                      amp_variance, amp_average))
                print(freq_domain_audio_partial)
            freq_domain_audio[freq] = 0
        else:
            heavy_coeffs.add(freq_in_hz)
            if freq_in_hz >= 220 and important_freq_count < len(important_frequencies):
                important_frequencies[important_freq_count] = freq_in_hz
                important_freq_count += 1
    last_heavy_coeffs = heavy_coeffs


def write_to_wav(data, sample_rate, file_path):
    wavfile.write(file_path, sample_rate, data)


def main():
    # audio_file = "Audio Clip With Background Noise.wav"
    # audio_file = "MS-SNSD/NoisySpeech_training/noisy62_SNRdb_40.0_clnsp62.wav"
    audio_file = "MS-SNSD/CleanSpeech_training/clnsp62.wav"
    start, end = 0, 10
    out_audio_file = "Audio Clip (hopefully) Without Background Noise.wav"
    sample_rate, data = wavfile.read(audio_file)
    print(f"sample rate: {sample_rate}")
    if len(data.shape) == 2:
        data = data[sample_rate * start:sample_rate * end, 0]
    else:
        data = data[sample_rate * start:sample_rate * end]
    new_data = np.zeros(data.shape)
    generator = utils.fft_generator(data, sample_rate, FRAMES_PER_SECOND)
    for freq_domain_audio, frame_start, frame_end in generator:
        remove_background_noise_from_frame_using_freq(freq_domain_audio)
        irfft_result = fft.irfft(freq_domain_audio)
        new_data[frame_start:frame_end] = irfft_result
    new_data *= 2 ** 15
    wavfile.write(out_audio_file, sample_rate, new_data.astype(np.int16))
    print(f'Saved file to "{out_audio_file}"')


if __name__ == '__main__':
    main()
