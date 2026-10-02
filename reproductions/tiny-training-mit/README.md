# Tiny Training on edge-toolkit

A source reproduction of TinyEngine sparse-update on-device training in a browser. The module builds MCUNet,
MobileNetV2 and ProxylessNAS training graphs as WebAssembly and runs them in a Web Worker connected to edge-toolkit.

## Scope

This is an engineering reproduction of the paper's code-generation and sparse-update execution path in a browser.
It is not a replication of the paper's MCU accuracy, latency, energy, or memory tables: the runtime target is
WebAssembly rather than the STM32 hardware used for the published measurements, and the interactive demo adds a
two-class prototype head for person-versus-scene adaptation. The generated TinyEngine training path remains
available and is exercised by the smoke verifier.

The experiment-specific Zig, C, JavaScript, patches and build pipeline live here; the published `et-ws-server`
provides the external infrastructure.

## Layout

- `modules/zig-te-train1/src/`: Zig entry point and C bridge.
- `modules/zig-te-train1/shim/`: ARM and CMSIS compatibility headers for WebAssembly.
- `modules/zig-te-train1/patches/`: wrappers around upstream code-generation paths.
- `modules/zig-te-train1/tools/`: regeneration, model extraction and smoke-test helpers.
- `modules/zig-te-train1/pkg/`: lightweight browser loader, worker and demo source. Built `.wasm` files are ignored.
- `mise.toml`: reproduction-level setup, build and server tasks.

## Generated files

Large and regenerable outputs are intentionally not committed:

- local TVM and Pixi environments;
- full upstream TinyEngine and Tiny Training checkouts;
- reduced TinyEngine and CMSIS vendor trees;
- graph triplets and generated model C sources;
- Zig caches and compiled WebAssembly;
- downloaded demo datasets.

The module's `.gitignore` records these boundaries. A fresh setup recreates the outputs from pinned upstream
revisions, a locked Linux Python environment, and a version-constrained macOS environment. The top-level Zig and
server tools intentionally remain floating during active development; pin them before archiving a final experiment
release.

## Set up and build

Install the toolchain and released edge-toolkit server. If no compatible server binary is available, mise builds it
with the project-local stable Rust toolchain:

```bash
mise install
```

Generate all model inputs and build the WebAssembly package:

```bash
mise run setup
```

This checks out the upstream projects, prepares TVM, regenerates the model inputs and C code, and builds the WASM
package. Linux x86-64 uses the locked Pixi environment. Apple Silicon macOS builds TVM locally, so its first setup is
substantially slower. Other hosts require a compatible Python environment supplied through `PYTHON`.

After setup, a normal source-only rebuild is:

```bash
mise run build
```

The resulting package contains:

```text
modules/zig-te-train1/pkg/
├── et_ws_zig_te_train1.js
├── et_ws_zig_te_train1_worker.js
├── et_ws_zig_te_train1-mcunet.wasm
├── et_ws_zig_te_train1-mbv2.wasm
├── et_ws_zig_te_train1-proxyless.wasm
└── package.json
```

Run the headless verification with:

```bash
mise run smoke
```

## Run it

Start the released edge-toolkit server:

```bash
mise run server
```

Open the URL printed by `et-ws-server`. The root page provides camera/file training; its **Demo** link opens the
reproducible person-versus-scene experiment under `modules/zig-te-train1/pkg/demo/`.

The browser loads `zig-te-train1` directly through hand-written mise tasks, without an `et-cli` scenario or headless
pyo3/WASI runner.

## Optional local sample pack

The demo works with synthetic or user-uploaded images without downloading a dataset. To generate its optional
person-versus-scene sample pack:

```bash
cd modules/zig-te-train1
mise run setup:sample-pack
```

That creates the ignored `pkg/demo/sample-pack/` directory served alongside the demo.
