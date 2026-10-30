# Designing comparable experiments

Use a documented reference workload, repeat each fault condition, and compare it with a matching healthy run. Keep the
application's client count fixed when it represents the intended deployment; vary the conditions you want to study.

The `temporary_scheduling_outage` worker fault holds an unsatisfiable node selector for 1–300 seconds and then removes
it. It preserves one desired replica so Core can observe unavailability under its single-replica supervision model.
The hold interval describes the placement constraint; Pod termination/startup affect the actual service interruption.
The injector preserves other selectors and retries its placement cleanup during final experiment restoration.
Intentional `temporary_scale_down` remains available for application rejoining tests, but zero desired replicas are
outside Core's recovery-observation contract and do not provide Core detection/MTTR measurements for that interval.

## Reference conditions

An adapter's recovery configuration can supply `default_resilience` when a persisted manual session requires older
settings. Its `resilience` object remains the deployment's declared benchmark configuration, including dependencies.
The controller restores `default_resilience` during cleanup without substituting it for the deployment permission
allowlist. Without this optional field, cleanup restores the declared `resilience` settings. Existing saved state
must match the full restored configuration; this option does not migrate or discard state or restart budgets.

Scenarios requiring useful progress after a temporary fault can set `require_post_fault_progress: true` and
`post_fault_progress_timeout_seconds` (1–7200 seconds). It defaults to the configured recovery deadline,
or 240 seconds when no deadline is supplied. The wait starts after the fault action returns. Pressure actions wait for
release before returning. Missing progress fails the experiment within this window and still runs cleanup. Component
readiness alone does not satisfy the requirement.

Network pressure results distinguish `congestion-observed` from `saturation-verified`. Training traffic shares the
controlled link with the probes. Low probe goodput can demonstrate congestion without verifying a saturation
curve. Reports retain both classifications and their supporting measurements; neither is a physical link-capacity
measurement. Pressure classification does not replace the separate requirement for post-fault application progress.

The network injector verifies receiver reachability from the affected client before applying its egress limit.
Refused/reset UDP requests are retained as probe transport errors. These produce `probe-path-error` and do not
establish congestion or saturation. Receiver status is collected before cleanup when available; the helper lifetime
includes setup margin and scales with the profile duration.

Scenarios can supply a `reference` object with an identifier and operator-declared conditions:

```json
{
  "reference": {
    "id": "classification-reference-v1",
    "dataset_version": "prepared-v1",
    "model_version": "classifier-v1",
    "hardware_profile": "documented-edge-hardware-v1",
    "network_profile": "local-cluster",
    "cache_condition": "images-and-data-already-present"
  },
  "seed": 17
}
```

An adapter may implement `reference_conditions(scenario)` to add dataset manifest hashes, model/source versions,
hyperparameters, participation rules and checkpoint policy. The controller also records deployed image tags, resource
requests/limits, node placement, image identities and node capacity/system details. Environment values and credentials
are excluded from this snapshot. Record processor/GPU model details in the declared hardware profile when the
cluster metadata does not provide them.

Conditions are stored under `reference.conditions`, with a stable `reference.fingerprint`. Unavailable observations
are listed under `reference.missing`. Declare workload conditions and ensure the operator can read Deployment, Pod
and node metadata before collecting comparative results. Operator-declared values describe assumptions;
the controller does not provision that hardware or independently validate dataset contents against manifest hashes.

## Fault timing

The existing `after_work_unit` setting injects after the completed-unit count exceeds the specified index. For example,
`after_work_unit: 0` waits for one completed unit. A `trigger` can also specify elapsed time or application events:

```json
{
  "fault": {
    "type": "delete_pod",
    "target": "worker-2",
    "trigger": {
      "event_kind": "training-started",
      "workload": "worker-2",
      "progress": 2,
      "delay_seconds": 0,
      "jitter_seconds": 1
    }
  }
}
```

The trigger waits for a matching `ET_EVENT` with progress at least the specified value. `after_seconds` sets a minimum
elapsed time after setup and reference capture. `delay_seconds` and seeded `jitter_seconds` delay injection after
the controller first observes the trigger. A `choices` list inside `trigger` selects one trigger definition using the
experiment seed. Limits are 16 choices, 300 seconds of delay and 60 seconds of jitter.

Reports retain the selected definition, event identity and timing, fault timestamps and event-to-injection delay
when the application's wall-clock event time is available. Polling and
command execution introduce delay: an event identifies an observed stage, not a guarantee that injection occurs before
that stage ends. Short transfer or write windows may require application instrumentation that deliberately holds a
boundary. Record any such delay as part of the reference conditions. An unobserved trigger produces a failed experiment.

## Repetitions and paired baselines

Application entry points using `et_experiments.benchmarking.cli` support:

```bash
python benchmarking/benchmark.py scenario-name --repetitions 10 --paired-baseline --seed 17
```

This runs a healthy baseline before each fault run, increments the timing seed for each pair, and restores configuration
between runs. The training/data seed remains a separate application setting. A baseline keeps the same workload
and recovery mode while clearing fault-specific success gates and SLO checks. Its `preparation_fault` lets the adapter
prepare equivalent memory limits or helper containers using `faults.prepare_fault()` without injecting the fault.

Each `series-*.json` contains baseline reports, fault reports and a repetition summary. Failed runs and their evidence
remain included; cancellation or failed cleanup stops the series. Series require cleanup between runs, so
`--keep-configuration` applies to individual runs only.

A failed healthy baseline fails the paired-series outcome while preserving the fault run's individual result.

`baseline_comparison` records differences and ratios for measured SLIs when reference fingerprints match, observations
are complete and the baseline passed. Mismatched or incomplete references produce `incompatible`, retaining the original
measurements for inspection. A zero baseline value has a difference but no ratio.

The UI already supports sequential repetitions through an application's catalog. An adapter can forward a catalog's
`paired_baseline` option to the shared CLI; use the request's repetition number to derive distinct timing seeds. Summary
results group experiments by application, scenario, reference fingerprint, fault configuration, selected trigger,
recovery mode and provider. Inspect those groups when comparing different stages or operating conditions.

Summaries include incident counts, unsuccessful and undetected outcomes, recovery/detection distributions, sample
standard deviation and a Wilson 95% interval for the incident recovery-success proportion. It assumes independent,
comparable incidents. Tail quantiles are shown from 20 samples for p95 and 100 for p99; these reporting thresholds
do not establish precise tail estimates. Report the sample count and retain individual results.

## Service continuity and participant recovery

The `useful_recovery` report distinguishes two observations:

- **Service continuity:** the primary application reports committed `progress` between confirmed fault apply/remove
  commands, or during the fault-action interval when explicit apply/remove boundaries are unavailable.
- **Participant recovery:** a target worker produces a fresh `contribution-ready` after fault removal, and the primary
  reports `contribution-accepted` for the same progress and process instance.

For participant evidence, the worker event uses its workload identity and instance. The accepted event includes
`participant_workload` (or `participant`) and `participant_instance`. Adapters translate domain identities into workload
names. Set `require_participant_recovery` to make fresh accepted participation a run-success condition.

These are independent outcomes: remaining workers may continue while one worker is unavailable. A `not-observed` result
means the required evidence was absent; it does not establish continuous service unavailability. Instantaneous commands
provide short continuity windows. Timed outages provide more useful observations of degraded service.

`processing_time_seconds` measures the interval from the primary's initial `operation-started` at progress zero to its
first committed `progress` reaching the work target. `processing_units_per_minute` uses this interval, including any
interruption and recovery. Both remain unmeasured when the boundaries are missing. Setup-inclusive wall time and
resource observations remain available separately. Compare processing time, unit latency, throughput and model quality
with the paired baseline.

## Checkpoint experiments

Vary checkpoint cadence, checkpoint age at interruption and actual workload/model size in separate documented profiles.
Inject around committed checkpoints and pre-commit boundaries, then verify that unfinished work continues. The
controller retains pre-fault diagnostic events and final observations so replacement preserves collected evidence.

`useful_recovery.checkpoint` pairs restoration and resume events from the same process. It records saved/restored
progress, checkpoint age and bytes, unsaved completed units, and time from resume to the next committed progress.
These describe progress metadata; the application verifies payload integrity and reconstructs its model.
Checkpoint-write, load and resume durations remain in the recovery metrics.
Repeated operation attempts are counted separately from unsaved completed work. Partial computation inside an
unfinished work unit remains outside those counts.

Set `require_unfinished_resume` to require fresh committed progress beyond the restored state. The checkpoint schedule
should preserve a final checkpoint, and recovery should validate its saved state against the workload.

## Longer and combined experiments

Increase the work target and timeout together for longer runs. A repetition series tests independent incidents;
a longer single run tests sustained processing, checkpoint activity and continued participation over time. Record actual
duration: a fixed round count does not guarantee a particular elapsed soak period.

`combined` supports two simultaneous, bounded faults on different allowlisted workers. Supported child types are
`temporary_scale_down`, `network_latency`, `network_bandwidth`, `network_loss` and `degraded_network`, with durations of
1 to 300 seconds. The adapter calls `faults.prepare_fault()` with the combined definition before starting workers.

```json
{
  "fault": {
    "type": "combined",
    "after_work_unit": 0,
    "faults": [
      { "type": "temporary_scale_down", "target": "worker-3", "duration_seconds": 45 },
      { "type": "network_latency", "target": "worker-2", "duration_seconds": 45, "latency_ms": 200 }
    ]
  }
}
```

Child results retain their own timestamps and cleanup outcomes. A child failure fails the combined action, while the
other bounded action completes its cleanup. Required Core recovery needs recovered incidents for both targets; temporary
scale-to-zero can remain unmeasured by Core's single-replica backend. Participant-return gates assess the application
outcome separately. Run individual faults first, then evaluate selected combinations with a matched baseline.

### Optional functional service checks

Applications can provide a `service_probe_monitor(backend, job_id, scenario)` adapter hook for independently observed
functional checks. Opt-in `service_probes` settings specify an interval, a response deadline and the number of workers
required for a usable service. The engine starts the observer after application rollout and finishes pending checks
before collecting logs and cleaning up. Detailed outcomes are saved separately and linked from `service_probes` in the
job report and controller result.

The shared monitor estimates availability using fixed intervals and the configured worker quorum. Observer failures and
unissued checks remain unknown. Where unknown intervals prevent a point estimate, the report retains lower/upper bounds
and observation coverage. Application failures, invalid results and response timeouts are unsuccessful checks.
Per-worker outcomes and successful response latencies are retained. This measurement is separate from training
completion, Core recovery evidence and resource readiness. Each adapter must document the function exercised and the
eligible observation window; a lightweight task does not establish availability of the complete application pipeline.
