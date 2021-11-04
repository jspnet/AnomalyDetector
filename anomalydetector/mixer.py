# -*- coding: utf-8 -*-
import os
import glob
import argparse
import librosa
import soundfile as sf


class Mixer:
    def __init__(self, input_file, anm="./anm/", out="./out/", sr=48000, out_sec=10, start_point=60):
        self._in_file = input_file
        self._anm_dir = anm
        self._out_dir = out
        self._sample_rate = sr
        self._start_point = sr * start_point
        self._out_sample_num = sr * out_sec

    def __call__(self):
        # 入力WAV読み込み
        wav, sr = librosa.load(self._in_file, sr=self._sample_rate)

        if not os.path.isdir(self._out_dir):
            os.makedirs(self._out_dir)

        # 正常評価音出力
        end_point = self._start_point + self._out_sample_num
        out_wav_np = wav[self._start_point:end_point]

        filename = self._out_dir + "eval_nor.wav"
        sf.write(filename, out_wav_np, sr, subtype="PCM_16")
        print("NORMAL eval sound output: ", filename)

        # 疑似異常音走査
        for i, anm_file in enumerate(glob.glob(self._anm_dir + "/*.wav")):
            anm_wav, _ = librosa.load(anm_file, sr=self._sample_rate)
            add_wav = self.add_anm(out_wav_np, anm_wav)

            filename = self._out_dir + "eval_anm_" + str(i) + ".wav"
            sf.write(filename, add_wav, sr, subtype="PCM_16")
            print("ANOMALY eval sound output: ", filename)

    def add_anm(self, nor_wav, anm_wav):
        nor_half = int(self._out_sample_num / 2)
        anm_half = int(anm_wav.shape[0] / 2)
        # 正常評価音の中間地点に疑似異常音を追加する
        add_wav = nor_wav.copy()
        add_wav[(nor_half-anm_half):(nor_half+anm_half)] += anm_wav
        return add_wav


def main():
    arg_parser = argparse.ArgumentParser(description='AnomalyMixer')
    arg_parser.add_argument('-i', '--input_file', type=str, required=True)
    arg_parser.add_argument('-a', '--anm_dir', type=str, default="./anm/")
    arg_parser.add_argument('-o', '--out_dir', type=str, default="./out/")
    arg_parser.add_argument('-r', '--sample_rate', type=int, default=48000)
    arg_parser.parse_args()
    args = arg_parser.parse_args()

    mixer = Mixer(args.input_file, anm=args.anm_dir, out=args.out_dir, sr=args.sample_rate)
    mixer()


if __name__ == "__main__":
    main()
