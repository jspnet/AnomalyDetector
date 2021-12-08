import argparse
import librosa
import numpy as np
import csv
import librosa.display
import glob
import os
import sys
from anomalydetector.hpfilter import HpFilter


class Preprocessor:
    def __init__(self, audio_dir, feat_dir, sr=16000, n_fft=1024, hop_length=512, n_mels=64, cutoff=None, power=2.0):
        self._audio_path = os.path.abspath(audio_dir + "/*.wav")
        self._audio_hpf_dir = os.path.abspath(audio_dir + "/hpf/")
        self._audio_hpf_path = os.path.abspath(audio_dir + "/hpf/*.wav")
        self._feat_dir = os.path.abspath(feat_dir)
        self._sr = sr
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._n_mels = n_mels
        self._power = power
        self._filter = None
        if cutoff is not None:
            self._filter = HpFilter(cutoff, sr=sr)

        if not os.path.isdir(self._feat_dir):
            os.makedirs(self._feat_dir)

    def __call__(self):
        train_dataset = glob.glob(os.path.abspath(self._audio_path))
        for i, file_path in enumerate(train_dataset):
            print(file_path + "  -> mel-spectrogram")
            log_mel = self._get_melspectrogram(file_path)

            out_file_num = log_mel.shape[1] / 10000
            if out_file_num > 1.:
                log_mel_ary = np.array_split(log_mel.T, out_file_num)
                for j, data in enumerate(log_mel_ary):
                    file_name = os.path.abspath(self._feat_dir + "/feat_%02d_%04d.csv" % (i, j))
                    with open(file_name, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(data)
            else:
                file_name = os.path.abspath(self._feat_dir + "/feat_%02d_0001.csv" % i)
                with open(file_name, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(log_mel.T.tolist())

    def _get_melspectrogram(self, audio_file):
        if self._filter is None:
            y, sr = librosa.load(audio_file, sr=self._sr)
        else:
            y, sr = self._filter.hpf(audio_file)

        # fftの幅で割り切れない半端を捨てる
        y = y[:int((len(y)-len(y) % self._n_fft))]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self._n_mels,
                                             n_fft=self._n_fft, hop_length=self._hop_length)
        return 20.0 / self._power * np.log10(mel + sys.float_info.epsilon)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--audio_dir", default="./in/")
    parser.add_argument("-o", "--feat_dir", default="./in/feature/")
    parser.add_argument("-r", "--sampling_rate", default=16000)
    parser.add_argument("-f", "--n_fft", default=1024)
    parser.add_argument("-p", "--hop_length", default=512)
    parser.add_argument("-m", "--n_mels", default=64)
    parser.add_argument("-c", "--cutoff", default=1000)
    args = parser.parse_args()
    proc = Preprocessor(args.audio_dir, args.feat_dir, args.sampling_rate,
                        args.n_fft, args.hop_length, args.n_mels, args.cutoff)
    proc()


if __name__ == '__main__':
    main()
