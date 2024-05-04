# check_diffusion_sine

flow matching を使ってサイン波生成してみるサンプル。  
３割くらい上手く行ってる気がするけどなんか大部分が直感に合わない。

## 準備

実装時は pytorch は 2.0 系を使った。ライブラリの依存関係が壊れており、 numpy や librosa を後から別バージョンにしないとたぶん動かない。

```py
pip install -r requirements.txt
pip install "numpy<1.20"
pip install "librosa==0.10.2"
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
