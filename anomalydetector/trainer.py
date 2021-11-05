import argparse
import os
import glob
import numpy as np
from tqdm import tqdm
from anomalydetector.model import AdModel


class Trainer:
    def __init__(self, checkpoint, feat_path, sr=16000, n_mels=64, frame_num=5, batch_size=100, epoch=10):
        self._checkpoint = checkpoint
        self._feat_path = os.path.abspath(feat_path + "/*.csv")
        self._detect_frame_num = ((frame_num * 2) + 1)
        self._input_dim = n_mels * self._detect_frame_num
        self._sr = sr
        self._batch_size = batch_size
        self._frame_num = frame_num
        self._epoch = epoch

        if not os.path.isdir(checkpoint):
            os.makedirs(checkpoint)

    def get_data_set(self, train_cnt_rate=0.9):
        train_dataset = glob.glob(os.path.abspath(self._feat_path))
        all_feat_ary = np.empty((0, self._input_dim))
        for file_path in train_dataset:
            print(file_path + " reading...")
            csv_ary = np.genfromtxt(file_path, delimiter=',')
            feat_len = csv_ary.shape[0] - self._detect_frame_num
            feat_ary = np.empty((feat_len, self._input_dim))
            for i in tqdm(range(feat_len)):
                # 観測点+前後フレームを足したフレーム分だけ抜き出してデータへ追加する
                feat_ary[i] = csv_ary[i:i+self._detect_frame_num].reshape([1, self._input_dim])

            # 全体へ追加
            all_feat_ary = np.vstack([all_feat_ary, feat_ary])

        return all_feat_ary

    def __call__(self):
        # dataset
        train_dataset = self.get_data_set(self._feat_path)

        model = AdModel.get_model(self._input_dim)
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        # train
        for epoch in range(self._epoch):
            print(f"Epoch: {epoch + 1}")
            _ = model.fit(
                train_dataset, train_dataset, batch_size=self._batch_size,
                verbose=1
            )

        model.save_weights(f"{self._checkpoint}/{self._epoch:05d}.ckpt")
        model.save(f"{self._checkpoint}/model")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", default="./out/ckpt/", help="モデルの保存先ディレクトリ")
    parser.add_argument("--feat_path", default="./in/feature/", help="データセットに使うファイルがあるディレクトリ")

    parser.add_argument("--sr", default=16000, help="サンプリングレート")
    parser.add_argument("--n_mels", default=64, help="入力データの次元数")
    parser.add_argument("--frame_num", default=5, help="前後に連結するフレーム数")
    parser.add_argument("--batch_size", default=100, help="バッチサイズ")

    parser.add_argument("--epoch", default=50, help="エポック数")
    parser.add_argument("--id", default="0", help="使用するgpuのid　※nvidia-smiコマンドで見れるGPU番号に対応")

    args = parser.parse_args()
    trainer = Trainer(args.checkpoint, args.feat_path, args.sr, args.n_mels, args.frame_num, args.batch_size, args.epoch)
    trainer()


if __name__ == '__main__':
    main()
