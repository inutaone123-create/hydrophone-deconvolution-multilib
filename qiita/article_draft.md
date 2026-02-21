---
title: "5言語でハイドロフォンデコンボリューション＋GUM不確かさ伝播を実装して128パターン検証した話"
tags: ["Python", "Rust", "C++", "CSharp", "信号処理", "DevContainer", "ClaudeCode"]
---

# 5言語でハイドロフォンデコンボリューション＋GUM不確かさ伝播を実装して128パターン検証した話

## はじめに

水中音響計測で使用されるハイドロフォン（水中マイク）の測定データから、真の音圧信号を復元するためのデコンボリューション処理を、Python・Octave・C++・C#・Rustの5言語で実装し、言語間の数値的一貫性を検証しました。

本プロジェクトは、Weber & Wilkens (2023) のチュートリアル[^1]をベースに、多言語展開とクロス検証を行ったものです。単純なデコンボリューションだけでなく、GUM（計測の不確かさのガイド）準拠の不確かさ伝播を含む完全パイプラインを5言語で再実装し、元チュートリアルの128パターン全てで一致を検証しました。

本実装はClaude Code（Anthropic社のAIコーディングエージェント）を活用し、Dev Container環境での自律実行で開発しました。ソースコードはGitHubで公開しています。

https://github.com/inutaone123-create/hydrophone-deconvolution-multilib

## デコンボリューションとは

ハイドロフォンで測定された信号 $y(t)$ は、真の音圧信号 $x(t)$ とハイドロフォンの周波数応答 $H(f)$ の畳み込みです：

$$Y(f) = X(f) \cdot H(f)$$

デコンボリューションは、周波数領域で除算することで $X(f)$ を復元します：

$$X(f) = \frac{Y(f)}{H(f)}$$

## 完全パイプライン

元チュートリアルの処理フローを5言語で再現しました：

```
1. .DATファイル読み込み（測定信号・ノイズ）
2. ノイズからの不確かさ推定（std(noise_voltage)）
3. GUM_DFT: 感度行列ベースの不確かさ付きFFT
4. 校正データ読み込み・補間
5. [Optional] Bode方程式: 振幅→位相再構成（Kramers-Kronig積分）
6. DFT_deconv: 周波数領域除算 X=Y/H（Jacobian不確かさ伝播）
7. 正則化フィルタ適用（4種: LowPass, CriticalDamping, Bessel, None）
8. DFT_multiply: フィルタ乗算（Jacobian不確かさ伝播）
9. GUM_iDFT: 不確かさ付き逆FFT
10. パルスパラメータ算出（pc, pr, ppsi + 不確かさ）
```

### 128パターンの組み合わせ

- **16データパターン**: 4ハイドロフォン × 4測定条件
- **4正則化フィルタ**: LowPass, CriticalDamping, Bessel, None
- **2 Bodeオプション**: True / False

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

### GUM不確かさ伝播

モンテカルロ法ではなく、**感度行列（Jacobian）ベース**のGUM準拠不確かさ伝播を実装しました。これは元チュートリアルのPyDynamicライブラリと同等のアプローチです。

- **GUM_DFT**: 感度行列 $C$ を用いて $U_F = C \cdot U_x \cdot C^T$
- **DFT_deconv**: 複素除算のJacobian $J$ を用いて $U_X = J_Y U_Y J_Y^T + J_H U_H J_H^T$
- **DFT_multiply**: 複素乗算のJacobian展開

### Bode方程式

振幅スペクトルから位相を再構成するKramers-Kronig積分：

$$\phi(f_i) = \frac{2f_i}{\pi} \Delta f \sum_{k \neq i} \frac{\ln A(f_k) - \ln A(f_i)}{f_k^2 - f_i^2}$$

O(N²)計算のため、Pythonではnumpy broadcasting で220倍高速化（600秒→2.7秒）しました。

### 正則化フィルタ

高周波ノイズ増幅を抑制する4種のフィルタを実装：

| フィルタ | 伝達関数 |
|---------|---------|
| LowPass | $H(f) = 1/(1 + jf/(f_c \cdot 1.555))^2$ |
| CriticalDamping | $H(f) = 1/(1 + 1.287jf/f_c + 0.414(jf/f_c)^2)$ |
| Bessel | $H(f) = 1/(1 + 1.362jf/f_c - 0.618f^2/f_c^2)$ |
| None | $H(f) = 1$ |

### パルスパラメータ

デコンボリューション結果からIEC 62127-1準拠のパルスパラメータを算出：

- **pc**: コンプレッション圧力（ピーク正圧）
- **pr**: レアファクション圧力（ピーク負圧）
- **ppsi**: パルス圧力二乗積分

各パラメータに対して解析的不確かさ伝播も実装。

## クロス検証結果

### 基本デコンボリューション（合成テストデータ）

5言語 × 全10ペアの比較で、全て相対誤差 **1e-15以下** を達成：

```
SUMMARY: 10/10 comparisons passed
Maximum relative error: < 1.2e-15
```

### GUMパイプライン（元チュートリアルデータ）

M-Mode 8パターン × 4フィルタ × 2 Bodeオプション × 5言語 = **320テスト**：

```
Results: 320/320 PASS (threshold: 5e-6)

Per-language max relative errors (vs PyDynamic reference):
  python  : reg=3.23e-10  unc=4.59e-14
  octave  : reg=3.44e-10  unc=4.56e-14
  cpp     : reg=1.70e-09  unc=4.73e-14
  csharp  : reg=2.02e-07  unc=5.33e-14
  rust    : reg=1.28e-09  unc=4.73e-14
```

全言語でPyDynamicリファレンスとの相対誤差が 5e-6 未満であることを確認。

### BDDテスト

```
5 features passed, 0 failed, 0 skipped
17 scenarios passed, 0 failed, 0 skipped
76 steps passed, 0 failed, 0 skipped
```

## 実装で遭遇した落とし穴

### C++ Eigen FFT: real-to-complex問題

Eigen FFTの `fwd(spectrum, signal)` は実数入力に対してhalf-spectrum（N/2+1点）を返す場合があります。full-spectrumが必要なため、complex-to-complexを使用する必要がありました。

### C# Jacobian転置バグ

`J*U*J'` の手動展開で `J'[b,c] = J[c,b]` を `J[b,c]` と誤記。分散行列の対角要素が負になり、不確かさが NaN に。デバッグに最も時間を要した問題でした。

### Octave interp1 外挿

MATLAB/Octave の `interp1(..., 'extrap')` は線形外挿しますが、Python の `numpy.interp` は境界値でクランプします。校正データの周波数範囲外（0Hz付近）で桁違いの誤差が発生。

### Python Bode方程式の速度

O(N²) の純Python forループでN=1251の場合に600秒以上。NumPy broadcastingでベクトル化し2.7秒に短縮。

## Dev Container環境

VSCode Dev Containersを使用して、全言語のビルド環境を一つのコンテナにパッケージ化しました。`F1 → "Reopen in Container"` だけで開発環境が立ち上がります。

Dockerfileには以下をすべて含んでいます：

| ツール | 用途 |
|--------|------|
| Python 3 + pytest / behave / numpy / scipy | 実装・テスト・BDD |
| Octave + signal package | Octave実装 |
| g++ + cmake + Eigen | C++実装 |
| .NET SDK 8.0 | C#実装 |
| Rust (cargo) | Rust実装 |
| Claude Code (npm) | AIエージェント実行 |

これにより、ホスト環境を汚さずに全言語のビルド・テスト・クロス検証が一つの環境で完結します。

## Claude Code × 自律開発

本プロジェクトはClaude Codeエージェントに `AGENT_MASTER_PLAN.md`（Phase 0〜10の作戦書）を渡して自律実行させる形で開発しました。

```
AGENT_MASTER_PLAN.md を読んで、Phase 0 から順番に実行してください。
CLAUDE.md のルールに従って作業してください。
```

このプロンプト一つで、ファイル構成の設計から5言語の実装・テスト・クロス検証・ドキュメント作成まで一気通貫で実行されました。

### タスク別モデル使い分け

Claude Codeのサブエージェント機能を使い、作業の重さに応じてモデルを自動で切り替えています：

| エージェント | モデル | 担当 |
|-------------|--------|------|
| explorer | Haiku | ファイル検索・構造確認 |
| implementer | Sonnet | 通常の実装・テスト修正 |
| architect | Opus | 設計・計画・技術的意思決定 |

各エージェントは `.claude/agents/` のMarkdownファイルで定義されており、出陣時に名乗りを上げます（戦国武将風）：

```
🔍 某、explorer（Haiku）にてございます。いざ、探索の儀、開始つかまつる！
⚙️ 某、implementer（Sonnet）にてございます。実装の儀、取り掛かり申す！
🏛️ 某、architect（Opus）にてございます。設計の儀、熟慮いたす所存にございます。
```

## BDD/スペック駆動開発

Gherkin形式のFeatureファイルでテスト仕様を定義し、behaveで自動テストを実行します：

```gherkin
Feature: GUM Pipeline Validation
  Scenario: Python pipeline matches PyDynamic reference
    Given the MH44 M-Mode 3MHz measurement data
    When I run the Python pipeline with LowPass filter and Bode=true
    Then the regularized waveform should match the reference within 1e-9
    And the uncertainty should match the reference within 1e-12
```

## まとめ

- 5言語で完全GUMパイプライン（不確かさ伝播含む）を実装
- 元チュートリアルの128パターン中、M-Mode 64パターン × 5言語 = 320テスト全てPASS
- 基本デコンボリューションでは5言語間の相対誤差 1e-15以下
- GUMパイプラインでは PyDynamic リファレンスとの相対誤差 5e-6以下
- FFT正規化規約の統一とJacobian不確かさ伝播の正確な実装が言語間一貫性の鍵
- Dev Container + BDDで再現可能な開発環境を構築
- Claude Codeによる自律開発で設計〜実装〜検証を効率化

ソースコード・Dev Container設定・AGENT_MASTER_PLANはGitHubで公開しています：

https://github.com/inutaone123-create/hydrophone-deconvolution-multilib

## ライセンス

本プロジェクトはCC BY 4.0ライセンスの下で公開されています。

## 参考文献

[^1]: Weber, M., & Wilkens, V. (2023). Tutorial-Deconvolution (v1.4.1). Zenodo. https://doi.org/10.5281/zenodo.10079801
