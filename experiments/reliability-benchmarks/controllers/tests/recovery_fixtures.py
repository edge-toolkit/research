"""Core supervisor report fixtures with explicit incident decisions and evidence."""


def core_report(mode="observe-only", status="recovered"):
    recovered = status == "recovered"
    detected = status not in {"suspected", "transient"}
    return {
        "schema_version": 1,
        "job_id": "job",
        "mode": mode,
        "clock_basis": "observer-monotonic-seconds",
        "incidents": [
            {
                "workload": "worker",
                "onset": 1,
                "detected": 3 if detected else None,
                "recovered": 9 if recovered else None,
                "baseline": 1,
                "status": status,
                "actions": [{"kind": "restart-reserved"}, {"kind": "restart-requested"}]
                if mode == "controller-assisted"
                else [],
            }
        ],
        "unknown_observations": 2,
        "recovery_seconds": [8] if recovered else [],
        "recovered_incidents": int(recovered),
        "detected_incidents": int(detected),
        "failure_detection_seconds": [2] if detected else [],
        "mean_failure_detection_seconds": 2 if detected else None,
        "mttr_seconds": 8 if recovered else None,
        "restart_budget": {"schema_version": 1, "job_id": "job", "restarts": {}},
        "checkpoint_measurements": [
            {
                "workload": "worker",
                "kind": kind,
                "instance": "new",
                "observed_at": 8,
                "progress": 1,
                "bytes": 100,
                "duration_millis": 125,
            }
            for kind in ("checkpoint-restored", "resume-completed")
        ]
        if recovered
        else [],
    }
