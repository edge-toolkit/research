# Research

Reproductions of research papers and demos on [edge-toolkit](https://github.com/edge-toolkit/core). Each one takes
a published result or demo, reruns it with edge-toolkit's hub and, where applicable, its runners in place of the
original infrastructure, and records what changed.

## Reproductions

- [Split learning for health: Distributed deep learning without sharing raw patient data][vepakomma2018]
  (Vepakomma, Gupta, Swedish and Raskar, MIT Media Lab, 2018):
  [split-learning-mit](reproductions/split-learning-mit)
- [On-Device Training Under 256KB Memory][lin2022]
  (Lin et al., 2022): [tiny-training-mit](reproductions/tiny-training-mit)

[vepakomma2018]: https://arxiv.org/pdf/1812.00564
[lin2022]: https://arxiv.org/pdf/2206.15472

Each reproduction is self-contained under `reproductions/<name>/`:

- `README.md`: what is being reproduced, how to run it, and how it differs from the original.
- `mise.toml`: the tools and tasks to set it up and run it.
- `modules/`: the edge-toolkit modules written for it.

A reproduction may also contain:

- `upstream/`: original code as a Git submodule pinned to the reproduced version.
- `scenario.yaml` and `deployment/`: an et-cli cluster input and its generated deployment.

Browser-hosted modules that do not need a headless runner may use hand-written mise tasks and generate large build
inputs locally.

## Getting started

Install [mise](https://mise.jdx.dev/), then clone with the submodules:

```bash
git clone --recurse-submodules https://github.com/edge-toolkit/research.git
```

From there, follow the README of the reproduction you want to run.

## Checks

The repository-wide checks run from the root:

```bash
mise install
```

```bash
mise run check
```

That runs dprint (Markdown, JSON, YAML), editorconfig-checker, gitleaks, ruff (Python lint and format), taplo (TOML)
and typos. `mise run fmt` applies the formatters. Upstream submodules, virtualenvs and `node_modules` are excluded
throughout, and the generated `deployment/` directories are left to et-cli rather than the formatters.
