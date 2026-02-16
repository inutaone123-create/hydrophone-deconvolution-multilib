# AGENT MASTER PLAN（最終版）
# Claude Code エージェント実行用マスタープラン

このドキュメントは、Dev Container環境でClaude Codeエージェントがハイドロフォンデコンボリューションの多言語実装プロジェクトを自律的に実行するための完全な設計書です。

---

## 🎯 プロジェクト目標

### 最終成果物
1. **5言語での完全実装**（Python, Octave, C++, C#, Rust）
2. **PocketFFT統一**による数値的一貫性
3. **BDD/スペック駆動開発**環境
4. **完全なクロス検証**（< 1e-14相対誤差）
5. **Dev Container環境**（「実践Claude Code入門」スタイル）
6. **GitHub管理準備**（CI/CD含む）
7. **Qiita記事ドラフト**

### 元のチュートリアル
- **著者**: Martin Weber, Volker Wilkens
- **DOI**: 10.5281/zenodo.10079801
- **ライセンス**: CC BY 4.0
- **URL**: https://github.com/Ma-Weber/Tutorial-Deconvolution

### このプロジェクトのライセンス
- **CC BY 4.0**（元と同じ）
- **帰属表示必須**（全ファイルに含める）

---

## 🐳 Dev Container環境

### 実行環境
- **場所**: Dev Container内
- **Claude Code**: コンテナ内にインストール
- **作業ディレクトリ**: /workspace

### 前提条件
- VSCode with Dev Containers拡張機能
- Docker Desktop（WSL2バックエンド）
- 初期設定ファイル（Dockerfile等）は既に配置済み

---

## 📁 プロジェクト構造

```
hydrophone-deconvolution-multilib/
├── README.md                          # 帰属表示含む
├── LICENSE                            # CC BY 4.0
├── Dockerfile                         # Claude Code含む開発環境
├── docker-compose.yml                 # サービス定義
├── .devcontainer/
│   └── devcontainer.json              # Dev Container設定
├── .vscode/
│   ├── settings.json
│   └── tasks.json
├── docs/
│   ├── API_REFERENCE.md
│   ├── IMPLEMENTATION_NOTES.md
│   └── CROSS_VALIDATION_REPORT.md
├── features/                          # BDD仕様
│   ├── deconvolution.feature
│   ├── uncertainty.feature
│   ├── cross_validation.feature
│   └── steps/
│       └── *.py
├── python/
│   ├── deconvolution/
│   │   ├── __init__.py
│   │   ├── core.py
│   │   └── uncertainty.py
│   ├── tests/
│   └── setup.py
├── octave/
│   ├── +deconvolution/
│   │   ├── deconvolve_without_uncertainty.m
│   │   └── deconvolve_with_uncertainty.m
│   └── tests/
├── cpp/
│   ├── CMakeLists.txt
│   ├── include/
│   ├── src/
│   └── tests/
├── csharp/
│   ├── Deconvolution.sln
│   └── Deconvolution/
├── rust/
│   ├── Cargo.toml
│   ├── src/
│   └── tests/
├── test-data/
│   └── generate_test_data.py
├── validation/
│   └── compare_results.py
├── .github/
│   └── workflows/
└── qiita/
```

---

## 🔢 実行フェーズ（Phase 0-10）

### Phase 0: 確認と準備
- 現在地確認（/workspace）
- 既存ファイル確認
- Git設定

### Phase 1: ライセンスとドキュメント基盤
- LICENSE作成
- README.md作成（帰属表示）
- 各言語用LICENSE_HEADERテンプレート

### Phase 2: プロジェクト設定
- .gitignore
- .vscode/settings.json, tasks.json

### Phase 3: BDD仕様定義
- features/*.feature作成

### Phase 4: Python実装（リファレンス）
- PocketFFT使用
- テスト作成

### Phase 5: Octave実装
- signal パッケージ使用

### Phase 6: C++実装
- Eigen FFT使用

### Phase 7: C#実装
- Math.NET使用

### Phase 8: Rust実装
- rustfft使用

### Phase 9: クロス検証
- 全言語結果比較（< 1e-14）

### Phase 10: 最終化
- GitHub Actions設定
- ドキュメント完成
- Qiita記事ドラフト

---

## 📜 ライセンス遵守要件

### 全ソースファイルに必須のヘッダー

#### Python
```python
"""
Hydrophone Deconvolution - Multi-language Implementation

Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
DOI: 10.5281/zenodo.10079801
Original License: CC BY 4.0

This implementation: 2024
License: CC BY 4.0
https://creativecommons.org/licenses/by/4.0/
"""
```

#### Octave/MATLAB
```matlab
% Hydrophone Deconvolution - Multi-language Implementation
%
% Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
% DOI: 10.5281/zenodo.10079801
% Original License: CC BY 4.0
%
% This implementation: 2024
% License: CC BY 4.0
```

#### C++
```cpp
/*
 * Hydrophone Deconvolution - Multi-language Implementation
 *
 * Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
 * DOI: 10.5281/zenodo.10079801
 * Original License: CC BY 4.0
 *
 * This implementation: 2024
 * License: CC BY 4.0
 */
```

#### C#
```csharp
/*
 * Hydrophone Deconvolution - Multi-language Implementation
 *
 * Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
 * DOI: 10.5281/zenodo.10079801
 * Original License: CC BY 4.0
 *
 * This implementation: 2024
 * License: CC BY 4.0
 */
```

#### Rust
```rust
//! Hydrophone Deconvolution - Multi-language Implementation
//!
//! Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
//! DOI: 10.5281/zenodo.10079801
//! Original License: CC BY 4.0
//!
//! This implementation: 2024
//! License: CC BY 4.0
```

---

## 🎯 共通API仕様

### 関数1: 不確かさなしデコンボリューション

全言語で同じ動作：
- 入力: measured_signal, frequency_response, sampling_rate
- 出力: deconvolved_signal
- アルゴリズム: FFT → 周波数領域除算 → IFFT

### 関数2: 不確かさありデコンボリューション

全言語で同じ動作：
- 入力: measured_signal, signal_uncertainty, frequency_response, response_uncertainty, sampling_rate, num_monte_carlo
- 出力: (deconvolved_signal, uncertainty)
- アルゴリズム: モンテカルロ法（1000サンプル）

---

## ✅ 各Phaseの成功基準

### Phase 0-2
- [ ] Git設定完了
- [ ] LICENSE, README.md作成
- [ ] .vscode設定完了

### Phase 3
- [ ] 3つのFeatureファイル作成
- [ ] Gherkin構文検証

### Phase 4
- [ ] Python実装完了
- [ ] pytest全パス
- [ ] PocketFFT使用確認

### Phase 5-8（各言語）
- [ ] 実装完了
- [ ] テスト全パス
- [ ] ライセンスヘッダー含む

### Phase 9
- [ ] 全言語で同一結果（< 1e-14）
- [ ] 10ペアの比較全てパス

### Phase 10
- [ ] GitHub Actions設定
- [ ] ドキュメント完成
- [ ] Qiita記事ドラフト

---

## 🚨 重要な制約条件

### 1. PocketFFT統一
- **Python**: numpy.fft（自動的にPocketFFT）
- **Octave**: fft/ifft（互換性あり）
- **C++**: Eigen FFT（互換性あり）
- **C#**: Math.NET（互換性あり）
- **Rust**: rustfft + 手動正規化（1/N）

### 2. FFT正規化
- **Forward FFT**: 正規化なし
- **Inverse FFT**: 1/N で正規化

### 3. 数値精度
- 倍精度浮動小数点（64-bit）
- 言語間の相対誤差 < 1e-14

### 4. ライセンス表示
- 全ソースファイルにヘッダー必須
- README.mdに詳細な帰属表示

---

## 📊 期待される最終状態

### テスト成功
```bash
# BDDテスト
behave features/
# → All scenarios passed

# クロス検証
python3 validation/compare_results.py
# → 10/10 comparisons PASSED
# → Max relative error: < 1e-14
```

### ファイル確認
```bash
ls -la
# LICENSE, README.md, Dockerfile, docker-compose.yml
# python/, octave/, cpp/, csharp/, rust/
# features/, test-data/, validation/
# .github/, docs/, qiita/
```

---

## 🎓 Claude Codeエージェントへの指示

**あなた（Claude Code）は、このMASTER_PLANに従って、Dev Container内でPhase 0からPhase 10まで自律的に実行してください。**

### 実行環境
- 作業ディレクトリ: /workspace
- 実行場所: Dev Container内
- 使用可能コマンド: python3, octave, g++, dotnet, rustc, cmake, git, など

### 各Phaseで
1. ✅ 目標を理解
2. ✅ 必要なファイルを作成
3. ✅ テストを実行
4. ✅ 成功基準を満たす
5. ✅ 次のPhaseへ

### エラー発生時
1. エラーを分析
2. 修正方法を判断
3. 自動的に修正
4. 再テスト

### 完了条件
- 全Phaseの成功基準を満たす
- 全テストがパス
- クロス検証が成功
- ライセンス表示が完璧

**次のドキュメント（AGENT_EXECUTION_GUIDE.md）で、各Phaseの詳細な実行手順を確認してください。**

---

## 📝 補足情報

### 開発優先順位
1. Python（リファレンス実装）
2. Octave（MATLAB互換）
3. C++（高速実装）
4. C#（.NET環境）
5. Rust（安全性重視）

### 参考リンク
- 元チュートリアル: https://github.com/Ma-Weber/Tutorial-Deconvolution
- PyDynamic: https://github.com/PTB-M4D/PyDynamic
- PocketFFT: https://gitlab.mpcdf.mpg.de/mtr/pocketfft
- CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

---

**準備完了！次のドキュメント（AGENT_EXECUTION_GUIDE.md）で実行開始！** 🚀
