# check_diffusion_sine

flow matching を使ってサイン波生成してみるサンプル。  
３割くらい上手く行ってる気がするけどなんか大部分が直感に合わない。

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

- ５エポックくらいでサイン波っぽいのが出てくる
- 推論時のステップ数を増やしまくると逆に汚くなる
- 学習時間を伸ばしても逆に汚くなる
- なぜか test loss が下がらない
  - ネットワークの `.eval()` を抜くと train loss と相関する
  - layer normalization か dropout の影響な気がするけど、まだちゃんと調べられてない
