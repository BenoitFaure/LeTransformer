# LeTransformer

**LeTransformer** is an implementation of the Transformer architecture using PyTorch.

The project serves as a personal refresher on the architecture from "Attention is All You Need". Significant optimizations can be made, and advancements in the field can be implemented. The scripts which come with the project serve as a proof to myself that the model boots up. Perhaps I should make some tests.

All core logic lives in the ```./src/``` directory.

## 🛠️ Development Environment

The project comes with a **Dev Container** configuration to help smooth the experience. To run this container you have a few options.

#### [1] VSCode

1. Install
    - Docker
    - VSCode
    - VSCode DevContainer Extension
2. Open the cloned repo and you'll get a prompt to open in one
    - Or ```Ctr+Shift+P >>> Dev Containes: Reopen in Container```

#### [2] GitHub CodeSpaces

On the GitHub Repo's page go: ```Code >> Code Spaces >> Open Code Spaces```

#### [3] DevContainer CLI

Install DevContainer:
``` bash
npm install -g @devcontainer/cli
```
Start the container:
``` bash
devcontainer up
```
Open a shell:
``` bash
devcontainer exec bash
```

## 📦Installation

The environment uses [uv](https://github.com/astral-sh/uv) for Python package and dependency management.

#### User Install

Just sync the lock file.
``` bash
uv sync
```

#### Developer Install

Install in editable mode.
``` bash
uv pip install -e
```

## 🚀CLI
You can run the following two commands. The objective is to show it works, if you want to train a transformer. Just pull from HuggingFace.

For training, recommended in order to have a model
``` bash
uv run train
```

For inference
``` bash
uv run inference
```