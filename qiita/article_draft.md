---
title: "5言語でハイドロフォンデコンボリューションを実装して数値的一貫性を検証した話"
tags: ["Python", "Rust", "C++", "CSharp", "信号処理"]
---

# 5言語でハイドロフォンデコンボリューションを実装して数値的一貫性を検証した話

## はじめに

水中音響計測で使用されるハイドロフォン（水中マイク）の測定データから、真の音圧信号を復元するためのデコンボリューション処理を、Python・Octave・C++・C#・Rustの5言語で実装し、言語間の数値的一貫性を検証しました。

本プロジェクトは、Weber & Wilkens (2023) のチュートリアル[^1]をベースに、多言語展開とクロス検証を行ったものです。

## デコンボリューションとは

ハイドロフォンで測定された信号 $y(t)$ は、真の音圧信号 $x(t)$ とハイドロフォンの周波数応答 $H(f)$ の畳み込みです：

$$Y(f) = X(f) \cdot H(f)$$

デコンボリューションは、周波数領域で除算することで $X(f)$ を復元します：

$$X(f) = \frac{Y(f)}{H(f)}$$

## 実装のポイント

### FFT正規化の統一

言語間の一貫性を確保するため、全実装で同じFFT正規化規約を採用しました：

- **順方向FFT**: スケーリングなし
- **逆方向FFT**: 1/N スケーリング

| 言語 | FFTライブラリ | 備考 |
|------|-------------|------|
| Python | NumPy (PocketFFT) | 標準的な規約 |
| Octave | 組み込み fft/ifft | 標準的な規約 |
| C++ | Eigen FFT (kissfft) | complex-to-complex使用 |
| C# | MathNet.Numerics | NoScaling + 手動1/N |
| Rust | rustfft | 手動1/N |

### 正則化

ゼロ除算を防ぐため、全言語で `epsilon = 1e-12` を周波数応答に加算しています。

### モンテカルロ不確かさ伝播

GUM準拠のモンテカルロ法で不確かさを伝播します。信号と周波数応答の両方にガウスノイズ摂動を加え、1000回のシミュレーションから平均と標準偏差を計算します。

## クロス検証結果

5言語 × 全10ペアの比較で、全て相対誤差 **1e-15以下** を達成しました：

```
SUMMARY: 10/10 comparisons passed
Maximum relative error: < 1.2e-15
```

これは要求仕様（1e-14未満）を大幅に上回る精度です。

## Dev Container環境

VSCode Dev Containersを使用して、全言語のビルド環境を一つのコンテナにパッケージ化しました。`F1 → "Reopen in Container"` だけで開発環境が立ち上がります。

## BDD/スペック駆動開発

Gherkin形式のFeatureファイルでテスト仕様を定義し、behaveで自動テストを実行します：

```gherkin
Feature: Hydrophone Deconvolution
  Scenario: Known signal recovery
    Given a known input signal
    And a known frequency response
    When I convolve and then deconvolve
    Then the recovered signal should match the original within 1e-10
```

## まとめ

- 5言語で同一アルゴリズムを実装し、数値的等価性を確認
- FFT正規化規約の統一が言語間一貫性の鍵
- C++ではEigen FFTのreal-to-complex変換に注意が必要
- C#とRustでは手動でIFFTの1/Nスケーリングが必要
- Dev Container + BDDで再現可能な開発環境を構築

## 参考文献

[^1]: Weber, M., & Wilkens, V. (2023). Tutorial-Deconvolution (v1.4.1). Zenodo. https://doi.org/10.5281/zenodo.10079801

## ライセンス

本プロジェクトはCC BY 4.0ライセンスの下で公開されています。
