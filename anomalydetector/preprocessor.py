import argparse
import librosa
import numpy as np
import csv
import librosa.display
import glob
import os
import sys
import scipy.signal as sp
import soundfile as sf


class Preprocessor:
    def __init__(self, audio_dir, feat_dir, sr=16000, n_fft=1024, hop_length=512, n_mels=64, power=2.0):
        self._audio_path = os.path.abspath(audio_dir + "/*.wav")
        self._feat_dir = os.path.abspath(feat_dir)
        self._sr = sr
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._n_mels = n_mels
        self._power = power

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
        y, sr = librosa.load(audio_file, sr=self._sr)
        # fftの幅で割り切れない半端を捨てる
        y = y[:int((len(y)-len(y) % self._n_fft))]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self._n_mels,
                                             n_fft=self._n_fft, hop_length=self._hop_length)
        return 20.0 / self._power * np.log10(mel + sys.float_info.epsilon)

    def _hpf(self, audio_file, cutoff=1000, tap=125):
        # cutoff :  カットオフ周波数(Hz)
        # tap :  タップ数（本講義では固定でも良い※時間があればいじっても良い！）

        # wavファイルのロード
        x, _ = librosa.load(audio_file, sr=self._sr)
        # フィルタ係数を算出する
        cutoff = cutoff / (self._sr/2)
        coef_hpf = sp.firwin(tap, cutoff, pass_zero=False)

        # 音データにフィルタをかける
        x_t = sp.lfilter(coef_hpf, 1, x)
        # フィルタ処理後の音データを保存
        sf.write("sample_hpf.wav", x_t, self._sr, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--audio_dir", default="./in/")
    parser.add_argument("-o", "--feat_dir", default="./in/feature/")
    parser.add_argument("-r", "--sampling_rate", default=16000)
    parser.add_argument("-f", "--n_fft", default=1024)
    parser.add_argument("-p", "--hop_length", default=512)
    parser.add_argument("-m", "--n_mels", default=64)
    args = parser.parse_args()
    proc = Preprocessor(args.audio_dir, args.feat_dir, args.sampling_rate, args.n_fft, args.hop_length, args.n_mels)
    proc()


if __name__ == '__main__':
    main()
