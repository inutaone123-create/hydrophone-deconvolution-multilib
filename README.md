# 🏯 水中音波デコンボリューション 多言語実装御陣

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dev Container](https://img.shields.io/badge/Dev%20Container-Ready-blue.svg)](https://code.visualstudio.com/docs/remote/containers)

不確かさ伝播を伴うハイドロフォン計測デコンボリューションの多言語実装にございます。
Weber & Wilkens（2023）の御教示を基に、5つの流派で忠実に再現いたした次第。

## この御陣の強み

- **五流派による実装**：Python・Octave・C++・C#・Rust
- **PocketFFT統一**：全流派にわたり、FFTの作法を統一
- **不確かさ伝播**：GUM準拠モンテカルロ法にて執り行う
- **BDD仕様駆動**：Gherkin記法による仕様書を旗印に開発
- **クロス検証済み**：言語間の相対誤差 < 1e-14、十戦全勝
- **Dev Container完備**：御城（開発環境）を即座に築城可能

## 出陣の儀（クイックスタート）

### Dev Container にて出陣（推奨）

```bash
# VSCode にて御城を開く
code .

# F1 → "Dev Containers: Reopen in Container"

# 御城内にて Claude Code を召喚
claude
```

### 各流派のテストを執り行う

```bash
# BDD仕様試験
behave features/

# 流派別試験
python -m pytest python/tests/ -v
octave --no-gui octave/tests/test_deconvolution.m
cpp/build/test_deconvolution
dotnet test csharp/Tests/
cargo test --manifest-path rust/Cargo.toml

# 全流派クロス検証
python validation/compare_results.py
```

## 召喚される諸将

作業の重さに応じて、Claude が自動的に適切な将を召喚いたします：

| 将 | モデル | 得意な戦 |
|----|--------|---------|
| 🔍 explorer（斥候） | Haiku | ファイル探索・構造把握・軽い調査 |
| ⚙️ implementer（実装） | Sonnet | 通常の実装・テスト修正・バグ修正 |
| 🏛️ architect（軍師） | Opus | 設計・計画・複雑な技術的意思決定 |

各将は出陣時に名乗りを上げ申す。`.claude/agent.log` に召喚の記録が残り候。

## 御城の見取り図

```
.
├── python/        # Python流（リファレンス実装）
├── octave/        # Octave流
├── cpp/           # C++流（Eigen使用）
├── csharp/        # C#流（MathNet使用）
├── rust/          # Rust流（rustfft使用）
├── features/      # BDD仕様書（旗印）
├── validation/    # クロス検証の場
├── docs/          # 軍記物（ドキュメント）
└── .claude/
    ├── agents/    # 諸将の辞令書
    └── settings.json
```

## 帰属表示・出典

### 元となる御教示

本実装は以下のハイドロフォンデコンボリューション御教示を基にしております：

**著者**:
- Martin Weber (University of Helsinki)
  - ORCID: [0000-0001-5919-5808](https://orcid.org/0000-0001-5919-5808)
- Volker Wilkens (Physikalisch-Technische Bundesanstalt)
  - ORCID: [0000-0002-7815-1330](https://orcid.org/0000-0002-7815-1330)

**出典**:
Weber, M., & Wilkens, V. (2023). Tutorial-Deconvolution (Version v1.4.1) [Software]. Zenodo.
https://doi.org/10.5281/zenodo.10079801

**原典ライセンス**: CC BY 4.0

### 本実装について

多言語実装・拡張：Inuta、2024年

**原典からの変更点**：
1. 再利用可能なライブラリ関数へのリファクタリング
2. 同一の数値的挙動を持つ5言語実装
3. 言語間一貫性のためのPocketFFT整合
4. BDDテストフレームワークの整備
5. クロス検証スイートの作成
6. Dev Container環境の整備

## 御法度（ライセンス）

CC BY 4.0 — 詳細は [LICENSE](LICENSE) を参照されたし。

## 御引用の儀

本ライブラリを研究にて使用の際は、以下を引用くださるよう御願い申し上げ候：

**原典**:
```
Weber, M., & Wilkens, V. (2023). Tutorial-Deconvolution (v1.4.1).
Zenodo. https://doi.org/10.5281/zenodo.10079801
```

**本実装**:
```
Inuta. (2024). Hydrophone Deconvolution Multi-Language Library.
GitHub. https://github.com/inutaone123-create/hydrophone-deconvolution-multilib
```
