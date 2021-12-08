import argparse
import librosa
import librosa.display
import os
import scipy.signal as sp
import soundfile as sf


class Filter:
    def __init__(self, cutoff=1000, tap=125, sr=16000, out_dir=None):
        # cutoff :  カットオフ周波数(Hz)
        # tap :  タップ数（本講義では固定でも良い※時間があればいじっても良い！）
        self._cutoff = cutoff
        self._tap = tap
        self._sr = sr

        self._out_dir = out_dir
        if out_dir is not None:
            self._out_dir = os.path.abspath(out_dir)
            if not os.path.isdir(self._out_dir):
                os.makedirs(self._out_dir)

    def hpf(self, audio_file):
        # wavファイルのロード
        x, sr = librosa.load(audio_file, sr=self._sr)
        # フィルタ係数を算出する
        cutoff = self._cutoff / (self._sr/2)
        coef_hpf = sp.firwin(self._tap, cutoff, pass_zero=False)

        # 音データにフィルタをかける
        x_t = sp.lfilter(coef_hpf, 1, x)
        if self._out_dir is not None:
            # フィルタ処理後の音データを保存
            out_path = os.path.split(os.path.basename(audio_file))
            out_file_path = os.path.abspath(self._out_dir + "/hpf_" + out_path[1])
            sf.write(out_file_path, x_t, self._sr, subtype="PCM_16")
        return x_t, sr


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_file", type=str, required=True)
    parser.add_argument("-o", "--out_dir", default=None)
    parser.add_argument("-c", "--cutoff", default=1000)
    parser.add_argument("-t", "--tap", default=125)
    parser.add_argument("-r", "--sampling_rate", default=16000)
    args = parser.parse_args()
    filt = Filter(args.cutoff, args.tap, args.sampling_rate, args.out_dir)
    filt.hpf(args.input_file)


if __name__ == '__main__':
    main()
