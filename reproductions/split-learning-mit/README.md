# Split learning on edge-toolkit

A reproduction of split learning, where a model is cut at one layer: a client holding the data trains the layers
below the cut, a server trains the layers above it, and only the cut-layer activations and their gradients cross
the network. It builds on the MIT
[split-learning-demo](https://github.com/jayvdb/split-learning-demo/tree/frontend-train-tensorflow), in which a
browser trains the client half of an MNIST CNN with TensorFlow.js and a Python server trains the other half with
PyTorch.

Here the Python server is replaced by edge-toolkit. The browser connects to `et-ws-server`, which relays its
frames to a split-learning agent running in `et-ws-pyo3-runner`. The frames the agent sends back are relayed to the
browser. The wire format is the upstream one, so the upstream web app runs unmodified.

| Path                            | What it is                                                                    |
| ------------------------------- | ----------------------------------------------------------------------------- |
| `upstream/`                     | The upstream demo, as a git submodule on `frontend-train-tensorflow`          |
| `modules/split-learning-server` | The server half as a pyo3 module: batch serving, training step and reset only |
| `scenario.yaml`                 | The et-cli input describing the cluster                                       |
| `deployment/`                   | What et-cli generates from `scenario.yaml`; do not edit by hand               |
| `requirements.txt`              | The Python packages the server half imports                                   |

## Prerequisites

- [mise](https://mise.jdx.dev/)
- Docker, for OpenObserve
- A Rust toolchain: mise builds et-cli, the hub and the pyo3 runner from crates.io

## Run it

From this directory:

```bash
mise install
```

```bash
mise run setup
```

`setup` checks out the upstream demo, creates `.venv` with PyTorch and friends, downloads MNIST, installs the web
app, and builds `et-ws-server` and `et-ws-pyo3-runner`. The first run takes a while.

Then start three things, each in its own terminal. First OpenObserve, the collector the hub sends its telemetry to:

```bash
mise run o2
```

Once it is up, the hub and the split-learning server agent:

```bash
mise run server
```

And the web app:

```bash
mise run web
```

Open <http://localhost:5173>, set the parameter server field to `ws://localhost:8080/ws` (the upstream default
points at its own Python server on port 8000), and start training. When the server agent stops, it stores the
trained server half as `server_mnist.onnx` in the hub's storage, under the `split-learning-server` agent.

## Regenerate the deployment

After changing `scenario.yaml` or the module:

```bash
mise run generate
```

This runs `et-cli`, which `mise install` builds from crates.io.

Instead of the `o2` and `server` steps, `mise run generated-scenario` from `deployment/` starts all three at once.
OpenObserve's UI is at <http://localhost:5080/>.

## Differences from upstream

- The server half handles only what a browser needs to train and use the model: `request_batch`,
  `activations_and_labels`, `reset_server` and `activations`. Saving the browser's client half is left out.
- Upstream saves the server half when the browser disconnects. The hub doesn't tell an agent when another agent
  disconnects, so here it is saved when the agent shuts down.
- The browser's URL has to be entered by hand. Upstream's headless Playwright run (`pnpm run train`) has no way to
  set it, so it can't be pointed at the hub yet.
