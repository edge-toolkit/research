# Harbour-FL engine module

This pyo3 workflow module implements the three-client multimodal vessel-classification demonstration. Each client
trains the same presence-aware fusion head over its local cached audio/image embeddings. The coordinator performs
sample-weighted FedAvg, evaluates each global model, and exchanges model pointers through Edge Toolkit storage and
the WebSocket hub. Raw samples remain local to the clients.

`HARBOUR_FL_ROLE` fixes a process as `client` or `coordinator` when it starts. Client processes additionally require
`HARBOUR_FL_CLIENT_ID`. The project scenario starts three clients and one coordinator from this single reusable
module package.

The coordinator emits version 1 `mlflow-event` messages. The generic MLflow bridge consumes those events without
making model training depend on MLflow availability. See the project-level README for setup and run instructions.

## Internal components

`pkg/et_ws_pyo3_harbour_fl_demo.py` is a thin runner entry point. Implementation lives in the installed
`src/harbour_fl/` package; these are Python internals, not additional selectable ws-modules.

| Component                     | Responsibility                                                            |
| ----------------------------- | ------------------------------------------------------------------------- |
| `runtime.py`, `protocol.py`   | Per-process state, runner hooks, hub protocol, and browser status.        |
| `client.py`, `coordinator.py` | Local fitting, round coordination, update validation, and resume.         |
| `data.py`, `learning.py`      | Prepared embeddings, batching, head training, and FedAvg.                 |
| `vision/`                     | Standalone camera-detector inspection utilities.                          |
| `checkpoints.py`              | Model serialization; Core provides durable pointers and integrity checks. |
| `instrumentation.py`          | Core lifecycle/checkpoint SDK integration and structured diagnostic logs. |
| `tracking.py`                 | Loss metrics, confusion matrices, and events for Core's MLflow bridge.    |

The application package has no benchmark-controller dependency. `setup:python` provisions it for local runners;
the application Dockerfile copies it into `/opt/et-python/harbour_fl` for containers.
