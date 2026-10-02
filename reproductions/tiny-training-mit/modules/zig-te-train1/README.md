# zig-te-train1

Zig WASM module that serves TinyEngine-generated sparse-update training and inference through edge-toolkit. It builds
three backbones: MCUNet, MobileNetV2 and ProxylessNAS.

## Quick start (fresh checkout)

```bash
mise run setup
```

This checks out pinned upstream revisions, prepares the Python environment, regenerates the model inputs and C code,
and builds the WASM artifacts. Linux x86-64 uses locked Pixi dependencies; Apple Silicon macOS builds TVM under
`.tvm-macos/`. Other hosts require a compatible interpreter supplied through `PYTHON`. The task is safe to rerun,
although it repeats regeneration.

Prerequisite: `mise`. From the reproduction root, `mise install` installs Zig, Pixi, Node and the released server.

## Directories

- `src/` — Zig entrypoint + C bridge code that exposes the TinyEngine path
- `shim/` — CMSIS/ARM-DSP stubs so TinyEngine kernels compile cleanly for wasm
- `patches/` — Runtime wrappers around upstream TinyEngine and Tiny Training (see `patches/README.md`)
- `tools/` — Model extraction, generated-C injection, environment setup and smoke-test helpers
- `codegen-{mcunet,mbv2,proxyless}/` — Regenerable C source for each backbone (output of `mise run codegen`)
- `triplets/{mcunet,mbv2,proxyless}/` — Regenerable IR triplets per backbone (`graph.json` + `params.pkl` + `scale.json`, output of `mise run triplets:regen`). Gitignored.
- `pkg/` — Web package artifacts: a camera-first root interface, `pkg/demo/` (reproducible demo HTML/JS), `pkg/*.js` and gitignored `pkg/*.wasm`
- `vendor/upstream/` — Pinned, shallow upstream checkouts (gitignored, populated by `mise run setup`)
- `vendor/{tinyengine,cmsis}/` — Runtime kernel subset for the Zig build (output of `mise run pull-vendor`)

## Pipeline at a glance

```
upstream tiny-training            ──▶ Stage 1 ──▶ triplets/MODEL/
  (pretrained .pkl checkpoints       (mise run triplets:regen)
   in vendor/upstream/.../assets/    needs locked Pixi env
   mcu_models/)                      (apache-tvm + torch)

triplets/MODEL/                   ──▶ Stage 2 ──▶ codegen-MODEL/
                                     (mise run codegen)
                                     uses locked Pixi env

codegen-MODEL/ + src/             ──▶ zig build ──▶ pkg/*.wasm
                                     (mise run build)
```

## Common tasks

- `mise run setup`: regenerate and build everything from a fresh checkout.
- `mise run setup:upstream`: create the pinned TinyEngine and Tiny Training checkouts.
- `mise run pull-vendor`: refresh the runtime kernel subset.
- `mise run env:create`: install the platform Python environment.
- `mise run triplets:regen [MODEL]`: regenerate Stage 1 triplets; `MODEL` defaults to `all`.
- `mise run codegen [MODEL]`: regenerate Stage 2 C code from the triplets.
- `mise run build`: run `zig build -Doptimize=ReleaseSmall`.
- `mise run setup:sample-pack`: prepare the optional COCO val2014 sample pack.
- `mise run smoke`: verify feature variation, binary margins and sparse updates in the MCUNet WASM.

## Demo sample pack

The browser demo can train from generated synthetic samples, uploaded images, or
a prepared local sample pack. From the reproduction root, prepare the data used
by the demo's **Load sample pack** button with:

```bash
cd modules/zig-te-train1
mise run setup:sample-pack
```

This creates the gitignored `pkg/demo/sample-pack/` directory. If it is absent, **Load sample pack** reports that the
pack is unavailable.

`setup:sample-pack` downloads COCO 2014 validation annotations and only the
selected images from `images.cocodataset.org`, samples 40 person + 40 scene
training images and 10 person + 10 scene validation images, cover-crops them to
128x128, and writes the local pack.

Useful variants:

- Use a different sampling seed:

  ```bash
  mise run setup:sample-pack 123
  ```

- Discard the cache and redownload annotations/images:

  ```bash
  mise run setup:sample-pack --force-download
  ```

The full `mise run setup` pipeline does not build the sample pack by default.
To include it in a fresh setup run, use:

```bash
WITH_SAMPLE_PACK=1 mise run setup
```

## Python environment

`env:create` is platform-aware:

- **linux-64**: installs the locked Pixi env from `pixi.toml` / `pixi.lock`.
- **osx-arm64**: uses `tools/tvm-macos-env/pixi.toml` for build tools, builds
  TVM 0.11.1 from source into the gitignored `.tvm-macos/` directory, and
  creates `.tvm-macos/venv`.

The macOS path intentionally does not use Homebrew. It still requires Apple's
compiler toolchain from Xcode Command Line Tools:

```bash
xcode-select --install
```

The macOS TVM source build enables LLVM because `triplets:regen` needs
`target.build.llvm` during Relay/autodiff. The Pixi build-tool env pins
`llvmdev` to 14.x to avoid TVM 0.11.1 compile failures against newer LLVM APIs.
If a previous no-LLVM `.tvm-macos/` build exists, `mise run env:create` rebuilds
it automatically when the recorded build config does not match.

The regeneration stages then use:

- **Stage 1** (`triplets:regen`): python 3.9 + apache-tvm + torch CPU.
- **Stage 2** (`codegen`): runs the TinyEngine codegen wrapper from the same platform env.

Set `PYTHON` to use a specific interpreter directly. On Linux, if Pixi is not
on PATH, the tasks can still fall back to conda via `CONDA_ENV`.
