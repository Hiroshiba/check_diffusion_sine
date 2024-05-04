# check_diffusion_sine

flow matching を使ってサイン波生成してみるサンプル。

## 準備

実装時は pytorch は 2.0.1 を使った。

```py
pip install -r requirements.txt
```

## 学習

```py
WANDB_MODE=offline python train.py config.yaml ./output_dir/
```

`WANDB_MODE=offline`をつけないと WandB サービスへのログインが求められる。

## 可視化

`visualize.ipynb`参照

## いろいろメモ

- 300 エポック（50000 イテレーション）くらいのモデルだと、ステップ 30 くらいかければかなりきれいなサイン波が作れる
- ステップ数を大幅に増やすと逆効果になる
- より小さいモデルで学習させたとき、エポック数も増やしていくと逆に波が汚くなっていった
- なぜか test loss が train loss ほど下がらない
  - もちろんネットワークの `.eval()` を抜くと train loss とがっつり相関する
  - layer normalization か dropout の影響な気がするけど、まだちゃんと調べられてない
