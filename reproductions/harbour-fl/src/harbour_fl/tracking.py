"""Application metrics and artifacts sent to the generic Core MLflow bridge."""

from __future__ import annotations

import html
import json
import os
from typing import Any

from .settings import MLFLOW_PRODUCER, MLFLOW_SCHEMA_VERSION, MLFLOW_TYPE, SCHEMA_VERSION


class TrackingMixin:
    def _confusion_matrix_artifacts(
        self, job_id: str, round_number: int, labels: list[str], matrix: list[list[int]]
    ) -> list[dict[str, str]]:
        """Store machine-readable and browser-viewable confusion-matrix artifacts."""
        stem = f"{job_id}-confusion-round-{round_number:03d}"
        payload = {"schema_version": SCHEMA_VERSION, "round": round_number, "labels": labels, "matrix": matrix}
        json_name = f"{stem}.json"
        html_name = f"{stem}.html"
        self.state["storage"].put(json_name, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        maximum = max((value for row in matrix for value in row), default=1) or 1
        header = "".join(f"<th scope='col'>{html.escape(label)}</th>" for label in labels)
        rows = []
        for label, values in zip(labels, matrix, strict=True):
            cells = []
            for value in values:
                intensity = value / maximum
                background = f"rgba(24, 119, 242, {0.08 + 0.84 * intensity:.3f})"
                foreground = "#fff" if intensity >= 0.55 else "#172033"
                cells.append(f"<td style='background:{background};color:{foreground}'>{value}</td>")
            rows.append(f"<tr><th scope='row'>{html.escape(label)}</th>{''.join(cells)}</tr>")
        document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Harbour FL confusion matrix — round {round_number}</title>
<style>
body{{font:14px system-ui,sans-serif;color:#172033;margin:24px;background:#f7f9fc}}
.card{{width:max-content;max-width:100%;padding:20px;background:white;border:1px solid #dfe5ef;
border-radius:12px;box-shadow:0 4px 16px #17203312}}
table{{border-collapse:separate;border-spacing:4px;text-align:center}}th{{font-weight:600}}
td{{min-width:56px;height:48px;border-radius:6px;font-weight:700}}
.axis{{color:#526079}}caption{{font-size:18px;font-weight:700;text-align:left;margin-bottom:14px}}
</style></head><body><main class="card">
<table><caption>Global test confusion matrix — round {round_number}</caption>
<thead><tr><th class="axis">Actual ↓ / Predicted →</th>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table>
</main></body></html>"""
        self.state["storage"].put(html_name, document.encode())
        artifact_dir = f"evaluation/round-{round_number:03d}"
        return [
            {
                "bucket": self.state["agent_id"],
                "filename": html_name,
                "artifact_path": f"{artifact_dir}/confusion-matrix.html",
            },
            {
                "bucket": self.state["agent_id"],
                "filename": json_name,
                "artifact_path": f"{artifact_dir}/confusion-matrix.json",
            },
        ]

    def _mlflow_round_metrics(self, metrics: dict[str, float], updates: list[dict[str, Any]]) -> dict[str, float]:
        """Select loss-centric MLflow charts while retaining richer UI/log metrics elsewhere."""
        values = {
            "global_test_loss": float(metrics["global_test_loss"]),
            "participating_clients": float(metrics["participating_clients"]),
            **{key: float(metrics[key]) for key in ("round_work_seconds",) if key in metrics},
        }
        for update in updates:
            client_id = str(update["client_id"])
            client_metrics = update["metrics"]
            values[f"client.{client_id}.training_loss"] = float(client_metrics["train_loss"])
            values[f"client.{client_id}.validation_loss"] = float(client_metrics["validation_loss"])
        return values

    def _emit_mlflow(self, event: str, event_id: str, run_key: str, **fields) -> None:
        value = {
            "type": MLFLOW_TYPE,
            "schema_version": MLFLOW_SCHEMA_VERSION,
            "producer": MLFLOW_PRODUCER,
            "event_id": event_id,
            "event": event,
            "run_key": run_key,
            "experiment": os.environ.get("HARBOUR_FL_MLFLOW_EXPERIMENT", "vessel-multimodal-fl"),
            "run_name": os.environ.get("HARBOUR_FL_MLFLOW_RUN_NAME", run_key),
            **fields,
        }
        self._send_json(value)
