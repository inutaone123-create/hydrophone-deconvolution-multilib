# Project Rules

## Git設定
- user.name: Inuta
- user.email: inuta.one.123@gmail.com

## 作業ルール
- 完了報告はファイル（例: `docs/COMPLETION_REPORT.md`）に保存すること。チャット内だけでなくファイルとして残す
- 全ソースファイルにCC BY 4.0ライセンスヘッダーを含める（下記テンプレート参照）
- 日本語でコミュニケーション
- 一区切りついたらコミット＆プッシュまで行う

## ライセンスヘッダー テンプレート

新規ファイル追加時は言語に応じた形式でヘッダーを付ける：

**Python（`"""`）**
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

**C++ / C#（`/* */`）**
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

**Rust（`//!`）**
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

**Octave（`%`）**
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

## ブランチ戦略

2層構成を採用する（develop 層は設けない）：

```
main
  └── feature/xxx or fix/xxx  （作業ブランチ）
        ↓ 実装・テスト・クロス検証を完了
main ← マージ → push
```

1. main から作業ブランチを作成: `git checkout -b feature/xxx` or `git checkout -b fix/xxx`
2. 作業ブランチ上で実装・テスト・クロス検証まで完了させる
3. main にマージ: `git checkout main && git merge feature/xxx`
4. push 後、不要になったブランチを削除: `git branch -d feature/xxx`

**命名例**: `feature/add-gum-pipeline`, `feature/add-pulse-params`, `fix/cpp-fft-spectrum`, `fix/csharp-path-issue`
形式: `feature/<動詞>-<内容>` / `fix/<言語or機能>-<問題>`

## コミット＆プッシュ手順
1. `git status` で変更内容を確認
2. `git diff` でステージ済み・未ステージの差分を確認
3. `git log --oneline -5` で直近のコミットメッセージのスタイルを確認
4. 対象ファイルを `git add <files>` でステージ（`git add .` は避ける）
5. コミットメッセージを作成してコミット（Co-Authored-By 付き）
6. `git push` でリモートにプッシュ
7. リモート: https://github.com/inutaone123-create/hydrophone-deconvolution-multilib

## 仕様変更・修正ワークフロー

仕様変更や修正が発生した際は、以下の手順を繰り返し実行する：

1. **変更内容の理解** — 変更依頼を確認し、影響範囲を特定
2. **影響分析** — 変更が及ぶ言語・ファイルを洗い出す（1言語のみ／全5言語か明確にする）
3. **実装** — 全対象言語に対して修正を適用（ライセンスヘッダー維持）
4. **言語別テスト** — 各言語のテストを実行して全パスを確認
   ```
   Python:  python -m pytest python/tests/ -v
   Octave:  octave --no-gui octave/tests/test_deconvolution.m
   C++:     cpp/build/test_deconvolution
   C#:      dotnet test csharp/Tests/
   Rust:    cargo test --manifest-path rust/Cargo.toml
   ```
5. **クロス検証** — テストデータ再生成 → 全言語エクスポート再実行 → 検証スクリプト実行
   ```
   データ再生成: python python/export_result.py
   検証実行:     python validation/compare_results.py
   合格基準:     全ペア 相対誤差 < 5e-6
   ```
   ❌ 失敗した場合: 失敗したペアの言語を特定 → ステップ2（影響分析）に戻る
6. **BDDテスト** — `behave features/` を実行（全シナリオ PASS を確認）
7. **ドキュメント更新** — 仕様変更・新機能追加時は docs/, COMPLETION_REPORT, Qiita 記事を更新
8. **コミット＆プッシュ** — 作業ブランチでコミット → main にマージ → push（ブランチ戦略・コミット＆プッシュ手順に従う）

## 環境ノート
- `/workspace` は `git config --global --add safe.directory /workspace` が必要
- Dev Container内で作業中
- Dockerfile, docker-compose.yml, .devcontainer/ は変更しない

## 技術的知見

### Python
- FFT: Forward=スケーリングなし、Inverse=1/N（PocketFFT互換）

### Octave
- FFT: Pythonと同じスケーリングで動作

### C++
- **Eigen FFT**: real-to-complex は half-spectrum になる → **complex-to-complex を使う**こと
- ファイルパス: `/workspace` からの絶対パス推奨

### C#
- **MathNet**: `FourierOptions.NoScaling` + 手動 1/N スケーリング
- **`dotnet run`** は bin/ から実行される → ファイルパスは絶対パスにする

### Rust
- **rustfft**: 両方向 unnormalized → 手動 1/N スケーリング
