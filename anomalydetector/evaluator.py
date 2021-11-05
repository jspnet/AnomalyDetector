import argparse
import librosa
import numpy as np
import librosa.display
import sys
import glob
import os
import csv
from tqdm import tqdm
from anomalydetector.model import AdModel


class Evaluator:
    def __init__(self, checkpoint, audio_file_path, output_path, sr, frame_num, n_mels, n_fft, hop_length, power=2.0):
        self._checkpoint = checkpoint
        self._audio_file_path = audio_file_path
        self._output_path = output_path
        self._sr = sr
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._n_mels = n_mels
        self._power = power
        self._frame_num = frame_num
        self._input_dim = ((self._frame_num * 2) + 1) * self._n_mels

        if not os.path.isdir(self._output_path):
            os.makedirs(self._output_path)

    def __call__(self):
        # モデル初期化
        checkpoint_dir = self._checkpoint
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"'{checkpoint_dir}' does not exist")
        checkpoints = [name for name in os.listdir(checkpoint_dir) if "ckpt" in name]
        if not checkpoints:
            raise ValueError(f"No checkpoint exists")
        checkpoints.sort()
        checkpoint_name = checkpoints[-1].split(".")[0]

        model = AdModel.get_model(self._input_dim)
        model.load_weights(f"{checkpoint_dir}/{checkpoint_name}.ckpt")

        for audio_file in glob.glob(self._audio_file_path + "/*.wav"):
            self._evaluate(model, audio_file)

    def _evaluate(self, model, file_path):
        _, file_name = os.path.split(file_path)
        base, _ = os.path.splitext(file_name)
        output_csv = self._output_path + "/" + base + ".csv"

        print(file_path + "  -> mel-spectrogram")
        log_mel = self._get_melspectrogram(file_path).T

        detect_frame_num = ((self._frame_num * 2) + 1)
        feat_len = log_mel.shape[0] - detect_frame_num
        feat_ary = np.empty((feat_len, self._input_dim))

        for i in tqdm(range(feat_len)):
            # 観測点+前後フレームを足したフレーム分だけ抜き出してデータへ追加する
            feat_ary[i] = log_mel[i:i + detect_frame_num].reshape([1, self._input_dim])

        # 推論フェーズ
        detection_result = []
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            for i, input_data in enumerate(feat_ary):
                input_data = np.expand_dims(input_data, axis=0)
                output_data = model.predict(input_data)

                # 入力と出力の平均二乗誤差を求める
                mse = ((input_data - output_data) ** 2).mean(axis=1)
                detection_result.append(mse[0])
                writer.writerow([mse[0]])

        print("output: " + output_csv)
        return detection_result

    def _get_melspectrogram(self, audio_file_path):
        y, sr = librosa.load(audio_file_path, sr=self._sr)
        # fftの幅で割り切れない半端を捨てる
        y = y[:int((len(y) - len(y) % self._n_fft))]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self._n_mels,
                                             n_fft=self._n_fft, hop_length=self._hop_length)
        return 20.0 / self._power * np.log10(mel + sys.float_info.epsilon)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", default="./out/ckpt/", help="モデルの保存先ディレクトリ")
    parser.add_argument("--audio_file_path", default="./in/sample/", help="異常検知対象ファイル")
    parser.add_argument("--output_path", default="./out/result/", help="異常検知対象ファイル")
    parser.add_argument("--sr", default=16000, help="サンプリングレート")

    # モデルパラメータ
    parser.add_argument("--n_mels", default=64, help="入力データの次元数")
    parser.add_argument("--frame_num", default=5, help="前後に連結するフレーム数")

    # データセットパラメータ
    parser.add_argument("-f", "--n_fft", default=1024)
    parser.add_argument("-p", "--hop_length", default=512)

    args = parser.parse_args()
    eval = Evaluator(args.checkpoint, args.audio_file_path, args.output_path,
                     args.sr, args.frame_num, args.n_mels, args.n_fft, args.hop_length)
    result = eval()


if __name__ == '__main__':
    main()
