# Project Rules

## Git設定
- user.name: Inuta
- user.email: inuta.one.123@gmail.com

## 作業ルール
- 完了報告はファイル（例: `docs/COMPLETION_REPORT.md`）に保存すること。チャット内だけでなくファイルとして残す
- 全ソースファイルにCC BY 4.0ライセンスヘッダーを含める
- 日本語でコミュニケーション

## 環境ノート
- `/workspace` は `git config --global --add safe.directory /workspace` が必要
- Dev Container内で作業中
- Dockerfile, docker-compose.yml, .devcontainer/ は変更しない

## 技術的知見
- C++ Eigen FFT: real-to-complex は half-spectrum になる → complex-to-complex を使う
- C# MathNet: `FourierOptions.NoScaling` + 手動 1/N スケーリング
- Rust rustfft: 両方向 unnormalized → 手動 1/N スケーリング
- C# `dotnet run` はbin/から実行される → ファイルパスは絶対パスにする
