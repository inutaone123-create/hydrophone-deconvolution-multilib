# Project Rules

## Git設定
- user.name: Inuta
- user.email: inuta.one.123@gmail.com

## 作業ルール
- 完了報告はファイル（例: `docs/COMPLETION_REPORT.md`）に保存すること。チャット内だけでなくファイルとして残す
- 全ソースファイルにCC BY 4.0ライセンスヘッダーを含める
- 日本語でコミュニケーション
- 一区切りついたらコミット＆プッシュまで行う

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
2. **影響分析** — 変更が及ぶ言語・ファイルを洗い出す
3. **実装** — 全対象言語に対して修正を適用（ライセンスヘッダー維持）
4. **言語別テスト** — 各言語のテストを実行して全パスを確認
5. **クロス検証** — テストデータ再生成 → 全言語のエクスポート再実行 → `compare_results.py` で10/10パス確認
6. **BDDテスト** — `behave features/` を実行
7. **ドキュメント更新** — 必要に応じて docs/, COMPLETION_REPORT, Qiita 記事を更新
8. **コミット＆プッシュ** — 作業ブランチでコミット → main にマージ → push（ブランチ戦略・コミット＆プッシュ手順に従う）

## 環境ノート
- `/workspace` は `git config --global --add safe.directory /workspace` が必要
- Dev Container内で作業中
- Dockerfile, docker-compose.yml, .devcontainer/ は変更しない

## 技術的知見
- C++ Eigen FFT: real-to-complex は half-spectrum になる → complex-to-complex を使う
- C# MathNet: `FourierOptions.NoScaling` + 手動 1/N スケーリング
- Rust rustfft: 両方向 unnormalized → 手動 1/N スケーリング
- C# `dotnet run` はbin/から実行される → ファイルパスは絶対パスにする
