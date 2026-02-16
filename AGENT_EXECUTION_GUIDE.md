# AGENT EXECUTION GUIDE（最終完全版）
# Dev Container環境でのPhase別実行手順

このドキュメントには、**初期設定ファイル**と**Phase別実行手順**の両方が含まれています。

---

## 🎬 使用方法

### 手順1: 初期設定ファイルを作成

以下の**3つのファイル**を手動で作成してください。
（このドキュメントの「初期設定ファイル」セクションを参照）

1. `Dockerfile`
2. `docker-compose.yml`
3. `.devcontainer/devcontainer.json`

### 手順2: Dev Containerで開く

```bash
code .
# F1 → "Dev Containers: Reopen in Container"
```

### 手順3: Dev Container内でClaude Code実行

```bash
# VSCode内蔵ターミナル（既にコンテナ内）
claude code
```

**プロンプト**:
```
Dev Container内で作業中です。

AGENT_MASTER_PLAN.md、AGENT_EXECUTION_GUIDE.md、AGENT_TEMPLATES.md を読んで、
Phase 0 から Phase 10 まで実行してください。

Dockerfile、docker-compose.yml、.devcontainer/ は既に存在するので、
それらは変更せず、他のファイルを作成・編集してください。
```

---

## 📁 初期設定ファイル（手動作成が必要）

### ファイル1: Dockerfile

**パス**: `Dockerfile`

```dockerfile
FROM ubuntu:22.04

# タイムゾーン設定
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# 基本パッケージ
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    octave octave-signal \
    build-essential cmake gdb \
    rustc cargo \
    wget apt-transport-https \
    git curl vim nano \
    zsh sudo \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Node.js インストール（Claude Code用）
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Claude Code インストール
RUN npm install -g @anthropic-ai/claude-code

# .NET SDK 8.0
RUN wget https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y dotnet-sdk-8.0 && \
    rm -rf /var/lib/apt/lists/*

# Python パッケージ
RUN pip3 install --break-system-packages \
    numpy>=1.21.0 \
    scipy>=1.7.0 \
    matplotlib>=3.4.0 \
    pytest>=7.0.0 \
    pytest-cov>=3.0.0 \
    behave>=1.2.6 \
    black>=23.0.0 \
    ipython

# Octave パッケージ
RUN octave --eval "pkg install -forge signal"

# C++ Eigen ライブラリ
RUN apt-get update && apt-get install -y \
    libeigen3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["sleep", "infinity"]
```

### ファイル2: docker-compose.yml

**パス**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  dev:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - .:/workspace:cached
      - /var/run/docker.sock:/var/run/docker.sock
    working_dir: /workspace
    stdin_open: true
    tty: true
    environment:
      - PYTHONUNBUFFERED=1
    command: sleep infinity

  test:
    build: .
    volumes:
      - .:/workspace
    working_dir: /workspace
    command: behave features/ --format pretty

  python-test:
    build: .
    volumes:
      - ./python:/app
      - ./test-data:/test-data
    working_dir: /app
    command: bash -c "pip install --break-system-packages -e . && pytest tests/ -v"

  octave-test:
    build: .
    volumes:
      - ./octave:/app
      - ./test-data:/test-data
    working_dir: /app
    command: octave tests/test_deconvolution.m

  cpp-test:
    build: .
    volumes:
      - ./cpp:/app
    working_dir: /app
    command: bash -c "mkdir -p build && cd build && cmake .. && make && ./test_deconvolution"

  csharp-test:
    build: .
    volumes:
      - ./csharp:/app
    working_dir: /app
    command: bash -c "dotnet build && dotnet test"

  rust-test:
    build: .
    volumes:
      - ./rust:/app
    working_dir: /app
    command: cargo test --release

  validation:
    build: .
    volumes:
      - .:/workspace
    working_dir: /workspace
    command: python3 validation/compare_results.py
```

### ファイル3: .devcontainer/devcontainer.json

**パス**: `.devcontainer/devcontainer.json`

```json
{
  "name": "Hydrophone Deconvolution Multi-Lang",
  "dockerComposeFile": "../docker-compose.yml",
  "service": "dev",
  "workspaceFolder": "/workspace",
  
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-python.black-formatter",
        "ms-vscode.cpptools",
        "ms-vscode.cmake-tools",
        "ms-dotnettools.csharp",
        "rust-lang.rust-analyzer"
      ],
      "settings": {
        "terminal.integrated.defaultProfile.linux": "bash",
        "python.testing.pytestEnabled": true,
        "python.testing.unittestEnabled": false,
        "[python]": {
          "editor.formatOnSave": true,
          "editor.defaultFormatter": "ms-python.black-formatter"
        },
        "C_Cpp.default.compilerPath": "/usr/bin/g++",
        "rust-analyzer.checkOnSave.command": "clippy"
      }
    }
  },
  
  "forwardPorts": [],
  
  "postCreateCommand": "echo 'Dev Container ready! Run: claude code'",
  
  "remoteUser": "root"
}
```

---

## 🚀 Phase別実行手順

### Phase 0: 確認と準備

```bash
# 現在地確認
pwd  # /workspace

# Git初期化（まだなら）
git init
git config user.name "Inuta"
git config user.email "inuta.one.123@gmail.com"

# ディレクトリ構造作成
mkdir -p docs features/steps
mkdir -p python/deconvolution python/tests
mkdir -p octave/+deconvolution octave/tests
mkdir -p cpp/include cpp/src cpp/tests
mkdir -p csharp/Deconvolution csharp/Tests
mkdir -p rust/src rust/tests
mkdir -p test-data validation qiita/figures
mkdir -p .github/workflows .vscode

# 環境確認
python3 --version   # Python 3.10+
octave --version    # GNU Octave 6.4+
g++ --version       # g++ 11+
dotnet --version    # 8.0+
rustc --version     # 1.6+
claude code --version  # Claude Code確認
```

**成功基準**:
- [ ] 全ディレクトリ作成完了
- [ ] 全コマンド動作確認
- [ ] Claude Code利用可能

---

### Phase 1: ライセンスとドキュメント

#### 1.1: LICENSE

**ファイル**: `LICENSE`

```
Creative Commons Attribution 4.0 International (CC BY 4.0)

Full license: https://creativecommons.org/licenses/by/4.0/legalcode

You are free to:
- Share: copy and redistribute the material
- Adapt: remix, transform, and build upon the material
for any purpose, even commercially.

Under the following terms:
- Attribution: You must give appropriate credit, provide a link to the 
  license, and indicate if changes were made.

---

ATTRIBUTION

This work is based on:
"Tutorial-Deconvolution" by Martin Weber and Volker Wilkens (2023)
DOI: 10.5281/zenodo.10079801
Original License: CC BY 4.0

Original Authors:
- Martin Weber (University of Helsinki, ORCID: 0000-0001-5919-5808)
- Volker Wilkens (Physikalisch-Technische Bundesanstalt, ORCID: 0000-0002-7815-1330)

Multi-language Implementation:
- [To be filled by user], 2024

Changes from Original:
1. Refactored Python tutorial into reusable library functions
2. Implemented in Octave, C++, C#, and Rust
3. Added PocketFFT alignment across all languages
4. Developed BDD/spec-driven testing framework
5. Created comprehensive cross-validation suite
```

#### 1.2: README.md

**ファイル**: `README.md`

```markdown
# Hydrophone Deconvolution Multi-Language Library

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dev Container](https://img.shields.io/badge/Dev%20Container-Ready-blue.svg)](https://code.visualstudio.com/docs/remote/containers)

Multi-language implementation of hydrophone measurement deconvolution with uncertainty propagation.

## 🌟 Features

- ✅ **5 Language Implementations**: Python, Octave, C++, C#, Rust
- ✅ **PocketFFT Compatible**: Consistent FFT across all languages
- ✅ **Uncertainty Propagation**: GUM-compliant Monte Carlo method
- ✅ **Spec-Driven Development**: BDD with Gherkin features
- ✅ **Cross-Validated**: < 1e-14 relative error between languages
- ✅ **Dev Container Ready**: Instant development environment

## 🚀 Quick Start

### Using Dev Container (Recommended)

```bash
# Open in VSCode
code .

# Reopen in Container
# F1 → "Dev Containers: Reopen in Container"

# Inside container
claude code  # Start development
```

### Running Tests

```bash
# BDD tests
behave features/

# Language-specific tests
cd python && pytest tests/
cd octave && octave tests/test_deconvolution.m
cd cpp && mkdir build && cd build && cmake .. && make && ./test_deconvolution
cd csharp && dotnet test
cd rust && cargo test

# Cross-validation
python3 validation/compare_results.py
```

## 🙏 Credits and Attribution

### Original Tutorial

This project is based on the hydrophone deconvolution tutorial:

**Authors**:
- Martin Weber (University of Helsinki)
  - ORCID: [0000-0001-5919-5808](https://orcid.org/0000-0001-5919-5808)
- Volker Wilkens (Physikalisch-Technische Bundesanstalt)
  - ORCID: [0000-0002-7815-1330](https://orcid.org/0000-0002-7815-1330)

**Publication**:
Weber, M., & Wilkens, V. (2023). Tutorial-Deconvolution (Version v1.4.1) [Software]. Zenodo.
https://doi.org/10.5281/zenodo.10079801

**Original License**: CC BY 4.0

### This Implementation

Multi-language implementation and extensions, 2024.

**Changes from Original**:
1. Refactored tutorial code into reusable library functions
2. Implemented in 5 languages with identical numerical behavior
3. Added PocketFFT alignment for cross-language consistency
4. Developed comprehensive BDD testing framework
5. Created cross-validation suite
6. Added Dev Container environment

## 📄 License

CC BY 4.0 - See [LICENSE](LICENSE) for full text.

## 📖 Citation

If you use this library in research, please cite:

**Original tutorial**:
```
Weber, M., & Wilkens, V. (2023). Tutorial-Deconvolution (v1.4.1).
Zenodo. https://doi.org/10.5281/zenodo.10079801
```

**This implementation**:
```
[Author]. (2024). Hydrophone Deconvolution Multi-Language Library.
GitHub. https://github.com/[username]/hydrophone-deconvolution-multilib
```
```

**成功基準**:
- [ ] LICENSE作成
- [ ] README.md作成（帰属表示完備）

---

### Phase 2: プロジェクト設定

#### 2.1: .gitignore

**ファイル**: `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*.so
venv/
env/
*.egg-info/
.pytest_cache/

# Octave
*.asv
*.m~
octave-workspace

# C++
*.o
*.out
build/
cmake-build-*/

# C#
bin/
obj/
*.user
*.suo

# Rust
target/
Cargo.lock

# IDEs
.vscode/.ropeproject
.idea/
*.swp

# OS
.DS_Store

# Test outputs
test-results/
validation/results/
*.log
```

#### 2.2: .vscode/settings.json

**ファイル**: `.vscode/settings.json`

```json
{
  "terminal.integrated.defaultProfile.linux": "bash",
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.linting.enabled": true,
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-python.black-formatter"
  },
  "C_Cpp.default.compilerPath": "/usr/bin/g++",
  "rust-analyzer.checkOnSave.command": "clippy",
  "files.watcherExclude": {
    "**/target/**": true,
    "**/build/**": true,
    "**/.venv/**": true
  }
}
```

#### 2.3: .vscode/tasks.json

**ファイル**: `.vscode/tasks.json`

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "BDD: Run All Features",
      "type": "shell",
      "command": "behave",
      "args": ["features/", "--format", "pretty"],
      "group": {"kind": "test", "isDefault": true}
    },
    {
      "label": "Python: Run Tests",
      "type": "shell",
      "command": "bash",
      "args": ["-c", "cd python && pytest tests/ -v"]
    },
    {
      "label": "Validation: Cross-Check All Languages",
      "type": "shell",
      "command": "python3",
      "args": ["validation/compare_results.py"]
    }
  ]
}
```

**成功基準**:
- [ ] .gitignore作成
- [ ] .vscode設定作成

---

### Phase 3-8: 実装フェーズ

**詳細なコードは AGENT_TEMPLATES.md を参照**

各言語で以下を実装：
1. ライセンスヘッダー含むソースコード
2. テストコード
3. ビルド設定（必要な場合）

---

### Phase 9: クロス検証

#### validation/compare_results.py

```python
"""
Cross-language validation script.

Hydrophone Deconvolution - Multi-language Implementation
Based on Tutorial-Deconvolution by Weber & Wilkens (2023)
DOI: 10.5281/zenodo.10079801
License: CC BY 4.0
"""

import numpy as np
from pathlib import Path

def main():
    languages = ['python', 'octave', 'cpp', 'csharp', 'rust']
    results = {}
    
    # Load results
    results_dir = Path('validation/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for lang in languages:
        filepath = results_dir / f'{lang}_result.csv'
        if filepath.exists():
            results[lang] = np.loadtxt(filepath)
            print(f"✓ Loaded {lang}: {len(results[lang])} samples")
        else:
            print(f"⚠ Missing {lang} results")
    
    # Compare all pairs
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60 + "\n")
    
    passed = 0
    total = 0
    
    for i, lang1 in enumerate(languages):
        if lang1 not in results:
            continue
        for lang2 in languages[i+1:]:
            if lang2 not in results:
                continue
            
            total += 1
            max_diff = np.max(np.abs(results[lang1] - results[lang2]))
            rel_diff = max_diff / np.max(np.abs(results[lang1]))
            
            print(f"{lang1.upper()} vs {lang2.upper()}:")
            print(f"  Max abs diff: {max_diff:.2e}")
            print(f"  Relative diff: {rel_diff:.2e}")
            
            if rel_diff < 1e-14:
                passed += 1
                print("  ✓ PASS\n")
            else:
                print("  ✗ FAIL\n")
    
    print("="*60)
    print(f"SUMMARY: {passed}/{total} comparisons passed")
    print("="*60)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main())
```

**実行**:
```bash
python3 validation/compare_results.py
```

**成功基準**:
- [ ] 10/10 comparisons passed
- [ ] Max relative error < 1e-14

---

### Phase 10: 最終化

#### .github/workflows/tests.yml

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build container
      run: docker-compose build
    
    - name: Run BDD tests
      run: docker-compose run test
    
    - name: Run validation
      run: docker-compose run validation
```

**成功基準**:
- [ ] GitHub Actions設定完了
- [ ] 全ドキュメント完成

---

## ✅ 全Phase完了後の確認

```bash
# 1. 環境確認
python3 --version && octave --version && g++ --version && dotnet --version && rustc --version

# 2. 全テスト
behave features/

# 3. クロス検証
python3 validation/compare_results.py

# 4. ライセンス確認
grep -r "Weber & Wilkens" --include="*.py" --include="*.m" --include="*.cpp" --include="*.cs" --include="*.rs" | wc -l

# 5. Git状態
git status
```

---

## 🎉 完了！

全Phaseが成功したら、プロジェクトは完成です！

次のステップ：
1. GitHubリポジトリ作成
2. プッシュ
3. Qiita記事執筆

**お疲れ様でした！** 🚀
