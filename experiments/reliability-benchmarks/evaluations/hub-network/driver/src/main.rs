//! Core-compatible registered-agent heartbeat workload with explicit per-vessel TCP source ports.
//! The heartbeat loop follows Core's hub.rs and ws-runner-common heartbeat_interval: discard the immediate tick,
//! then send empty WebSocket pings every five seconds, read incoming frames, and delay missed ticks.

use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};
use tokio::net::{TcpSocket, TcpStream};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinSet;
use tokio_tungstenite::{WebSocketStream, client_async, tungstenite::Message};

type Socket = WebSocketStream<TcpStream>;
type HeartbeatError = (&'static str, String, Option<Duration>);

#[derive(Default)]
struct Stats {
    cutoff_gate: Mutex<()>,
    attempted: AtomicUsize,
    finished: AtomicUsize,
    registered: AtomicUsize,
    connected: AtomicUsize,
    sent_pings: AtomicUsize,
    pongs: AtomicUsize,
    hold_pings: AtomicUsize,
    hold_pongs: AtomicUsize,
    failures: AtomicUsize,
    holding: AtomicBool,
    registration_censored: AtomicBool,
    cancelled_registrations: AtomicUsize,
    timing_registered: AtomicUsize,
    timing_sent: AtomicUsize,
    timing_matched: AtomicUsize,
    timing_failures: AtomicUsize,
    errors: Mutex<Vec<Value>>,
    error_counts: Mutex<HashMap<String, usize>>,
    timing_ms: Mutex<Vec<f64>>,
    heartbeat_lag_ms: Mutex<Vec<f64>>,
    send_ms: Mutex<Vec<f64>>,
    runtime_lag_ms: Mutex<Vec<f64>>,
    alive_ms: Mutex<Vec<f64>>,
    control_status: Mutex<Value>,
    registration_attempts: AtomicUsize,
    registration_successes: AtomicUsize,
    retry_attempts: AtomicUsize,
    reconnections: AtomicUsize,
    recovery_started: AtomicUsize,
    recovery_completed: AtomicUsize,
    reconnect_ms: Mutex<Vec<f64>>,
    useful_recovery_ms: Mutex<Vec<f64>>,
    presence: Mutex<Presence>,
    hold_deadline: Mutex<Option<tokio::time::Instant>>,
    hold_ready: Notify,
    measurement_started: AtomicBool,
    phase: AtomicUsize,
    timings: Mutex<HashMap<String, Vec<f64>>>,
    last_vessel_pong: Mutex<HashMap<usize, Instant>>,
    admission: Mutex<Option<Admission>>,
    diagnostic_enabled: AtomicBool,
}

struct Presence {
    current: usize,
    last: Instant,
    active: bool,
    seconds: f64,
    minimum: usize,
    diagnostics: Option<DiagnosticBook>,
}
impl Default for Presence {
    fn default() -> Self {
        Self {
            current: 0,
            last: Instant::now(),
            active: false,
            seconds: 0.0,
            minimum: 0,
            diagnostics: None,
        }
    }
}

struct VesselState {
    connected: bool,
    stage: &'static str,
    stage_started: Instant,
    attempts: usize,
    source: Option<String>,
    last_disconnect: Option<(Instant, String, String)>,
    last_error: Option<(Instant, String, String)>,
    last_reply: Option<Instant>,
}
impl VesselState {
    fn snapshot(&self, index: usize, now: Instant) -> Value {
        let event = |value: &Option<(Instant, String, String)>| {
            value.as_ref().map(|(at, kind, detail)|
            json!({"age_seconds":now.saturating_duration_since(*at).as_secs_f64(),"operation":kind,"detail":detail}))
        };
        json!({"agent":index,"vessel_id":format!("vessel-{index:05}"),"connected":self.connected,
            "stage":self.stage,"stage_seconds":now.saturating_duration_since(self.stage_started).as_secs_f64(),
            "attempt_count":self.attempts,"retry_count":self.attempts.saturating_sub(1),"source":self.source,
            "last_disconnect":event(&self.last_disconnect),"last_error":event(&self.last_error),
            "last_reply_age_seconds":self.last_reply.map(|at| now.saturating_duration_since(at).as_secs_f64())})
    }
}
#[derive(Default)]
struct FollowedVessel {
    registered: Option<Instant>,
    replied: Option<Instant>,
}
struct DiagnosticBook {
    started: Instant,
    states: Vec<VesselState>,
    cutoff: Option<Instant>,
    cohort: HashMap<usize, FollowedVessel>,
    followup_end: Option<Instant>,
}
impl DiagnosticBook {
    fn new(count: usize, started: Instant) -> Self {
        Self {
            started,
            cutoff: None,
            cohort: HashMap::new(),
            followup_end: None,
            states: (0..count)
                .map(|_| VesselState {
                    connected: false,
                    stage: "not-started",
                    stage_started: started,
                    attempts: 0,
                    source: None,
                    last_disconnect: None,
                    last_error: None,
                    last_reply: None,
                })
                .collect(),
        }
    }
    fn accepts(&self, now: Instant) -> bool {
        self.followup_end.is_none_or(|end| now <= end)
    }
    fn cutoff(&mut self, now: Instant) -> Value {
        self.cutoff = Some(now);
        self.cohort = self
            .states
            .iter()
            .enumerate()
            .filter(|(_, state)| !state.connected)
            .map(|(index, _)| (index, FollowedVessel::default()))
            .collect();
        let missing: Vec<Value> = self
            .states
            .iter()
            .enumerate()
            .filter(|(_, state)| !state.connected)
            .map(|(index, state)| state.snapshot(index, now))
            .collect();
        json!({"clock_basis":"client-monotonic-seconds",
            "elapsed_seconds":now.duration_since(self.started).as_secs_f64(),
            "requested_agents":self.states.len(),
            "connected":self.states.len()-missing.len(),
            "missing_count":missing.len(),"missing_vessels":missing})
    }
    fn followup(&self, now: Instant) -> Value {
        let cutoff = self.cutoff.unwrap();
        let mut rows: Vec<Value> = self
            .cohort
            .iter()
            .map(|(index, followed)| {
                let mut row = self.states[*index].snapshot(*index, now);
                row["first_registration_after_cutoff_seconds"] = json!(
                    followed
                        .registered
                        .filter(|at| *at <= now)
                        .map(|at| at.duration_since(cutoff).as_secs_f64())
                );
                row["first_reply_after_cutoff_seconds"] = json!(
                    followed
                        .replied
                        .filter(|at| *at <= now)
                        .map(|at| at.duration_since(cutoff).as_secs_f64())
                );
                row
            })
            .collect();
        rows.sort_by_key(|row| row["agent"].as_u64().unwrap());
        json!({"elapsed_seconds":now.duration_since(cutoff).as_secs_f64(),"cohort_size":rows.len(),
            "registered":self.cohort.values().filter(|value| value.registered.is_some_and(|at| at <= now)).count(),
            "useful_reply_observed": self.cohort.values()
                .filter(|value| value.replied.is_some_and(|at| at <= now)).count(),
            "vessels":rows})
    }
}
impl Presence {
    fn update(&mut self, now: Instant) {
        if self.active {
            self.seconds += self.current as f64 * now.duration_since(self.last).as_secs_f64();
        }
        self.last = now;
    }
}

#[derive(Clone)]
struct RetryPolicy {
    enabled: bool,
    response_timeout: Option<Duration>,
    attempt_seconds: u64,
    base_ms: u64,
    max_ms: u64,
    seed: u64,
    ips: Vec<Ipv4Addr>,
}
impl RetryPolicy {
    fn disabled(ip: Ipv4Addr) -> Self {
        Self {
            enabled: false,
            response_timeout: None,
            attempt_seconds: 45,
            base_ms: 1000,
            max_ms: 30000,
            seed: 20261120,
            ips: vec![ip],
        }
    }
    fn delay(&self, failures: u32, agent: usize) -> Duration {
        let ceiling = self
            .base_ms
            .saturating_mul(1_u64 << failures.min(16))
            .min(self.max_ms);
        let mut bits =
            self.seed ^ (agent as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ u64::from(failures);
        bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
        let floor = (ceiling / 2).max(1);
        Duration::from_millis(floor + (bits ^ (bits >> 31)) % (ceiling - floor + 1))
    }
}

impl Stats {
    fn record_reply(&self, index: usize) {
        if !self.diagnostic_enabled.load(Ordering::Relaxed) {
            return;
        }
        if let Some(book) = self.presence.lock().unwrap().diagnostics.as_mut() {
            let now = Instant::now();
            if !book.accepts(now) {
                return;
            }
            book.states[index].last_reply = Some(now);
            if let Some(followed) = book.cohort.get_mut(&index) {
                followed.replied.get_or_insert(now);
            }
        }
    }
    fn stage(&self, index: usize, stage: &'static str) {
        let _gate = self.cutoff_gate.lock().unwrap();
        if !self.diagnostic_enabled.load(Ordering::Relaxed) {
            return;
        }
        if let Some(book) = self.presence.lock().unwrap().diagnostics.as_mut() {
            let now = Instant::now();
            if !book.accepts(now) {
                return;
            }
            let state = &mut book.states[index];
            state.stage = stage;
            state.stage_started = now;
        }
    }
    fn timing(&self, name: &str, elapsed: Duration) {
        let _gate = self.cutoff_gate.lock().unwrap();
        self.timing_locked(name, elapsed);
    }
    fn timing_locked(&self, name: &str, elapsed: Duration) {
        let phase = match self.phase.load(Ordering::Relaxed) {
            1 => "settling",
            2 => "holding",
            3 => "followup",
            _ => "registration",
        };
        self.timings
            .lock()
            .unwrap()
            .entry(format!("{phase}:{name}"))
            .or_default()
            .push(elapsed.as_secs_f64() * 1000.0);
    }
    fn connected_seconds(&self) -> f64 {
        let mut meter = self.presence.lock().unwrap();
        meter.update(Instant::now());
        meter.seconds
    }
    fn error(&self, operation: &str, agent: usize, detail: String) {
        let _gate = self.cutoff_gate.lock().unwrap();
        self.error_locked(operation, agent, detail);
    }
    fn error_locked(&self, operation: &str, agent: usize, detail: String) {
        if self.diagnostic_enabled.load(Ordering::Relaxed) && !operation.starts_with("timing-") {
            if let Some(book) = self.presence.lock().unwrap().diagnostics.as_mut() {
                let now = Instant::now();
                if book.accepts(now) {
                    let state = &mut book.states[agent];
                    let event = (now, operation.to_string(), detail.clone());
                    if operation.starts_with("heartbeat-") {
                        state.last_disconnect = Some(event.clone());
                    }
                    state.last_error = Some(event);
                }
            }
        }
        if operation.starts_with("timing-") {
            self.timing_failures.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failures.fetch_add(1, Ordering::Relaxed);
        }
        let reason = if detail.contains("408") {
            "http-408"
        } else if detail.contains("inactivity") {
            "inactivity"
        } else if detail.contains("attempt timed out") {
            "attempt-timeout"
        } else if detail.contains("Address already in use") {
            "address-in-use"
        } else if detail.contains("reset") {
            "reset"
        } else {
            "other"
        };
        *self
            .error_counts
            .lock()
            .unwrap()
            .entry(format!("{operation}:{reason}"))
            .or_default() += 1;
        let mut errors = self.errors.lock().unwrap();
        if errors.len() < 16 {
            errors.push(json!({"operation": operation, "agent": agent, "detail": detail}));
        }
    }
    fn sample(&self, elapsed: f64, phase: &str) -> Value {
        let _gate = self.cutoff_gate.lock().unwrap();
        self.sample_locked(elapsed, phase)
    }
    fn sample_locked(&self, elapsed: f64, phase: &str) -> Value {
        json!({"type": "sample", "elapsed_seconds": elapsed, "phase": phase,
            "attempted": self.attempted.load(Ordering::Relaxed),
            "finished_registration_attempts": self.finished.load(Ordering::Relaxed),
            "registered_total": self.registered.load(Ordering::Relaxed),
            "connected": self.connected.load(Ordering::Relaxed),
            "sent_pings": self.sent_pings.load(Ordering::Relaxed),
            "received_pongs": self.pongs.load(Ordering::Relaxed),
            "failures": self.failures.load(Ordering::Relaxed),
            "cancelled_registrations": self.cancelled_registrations.load(Ordering::Relaxed),
            "timing_probes_registered": self.timing_registered.load(Ordering::Relaxed),
            "timing_sent": self.timing_sent.load(Ordering::Relaxed),
            "timing_matched": self.timing_matched.load(Ordering::Relaxed),
            "timing_failures": self.timing_failures.load(Ordering::Relaxed),
            "runtime_lag_recent_ms": self.runtime_lag_ms.lock().unwrap().last().copied(),
            "registration_attempts": self.registration_attempts.load(Ordering::Relaxed),
            "registration_successes": self.registration_successes.load(Ordering::Relaxed),
            "retry_attempts": self.retry_attempts.load(Ordering::Relaxed),
            "reconnections": self.reconnections.load(Ordering::Relaxed),
            "useful_recoveries": self.recovery_completed.load(Ordering::Relaxed),
            "admission": self.admission.lock().unwrap().as_ref().map(Admission::snapshot),
            "connected_vessel_seconds": self.connected_seconds(),
            "hold_sent_pings": self.hold_pings.load(Ordering::Relaxed),
            "hold_received_pongs": self.hold_pongs.load(Ordering::Relaxed)})
    }
}

async fn register(
    address: SocketAddr,
    url: &str,
    local_ip: Ipv4Addr,
    source_port: u16,
    identity: &str,
) -> Result<Socket, String> {
    register_observed(address, url, local_ip, source_port, identity, None).await
}

async fn observed<T>(
    observation: Option<&(Arc<Stats>, bool, usize)>,
    stage: &'static str,
    operation: impl std::future::Future<Output = Result<T, String>>,
) -> Result<T, String> {
    let began = Instant::now();
    if let Some((stats, _, index)) = observation {
        stats.stage(*index, stage);
    }
    let result = operation.await;
    if let Some((stats, repeated, _)) = observation {
        let kind = if *repeated { "retry" } else { "initial" };
        let status = if result.is_ok() { "ok" } else { "error" };
        stats.timing(&format!("{kind}:{stage}:{status}"), began.elapsed());
    }
    result
}

async fn register_observed(
    address: SocketAddr,
    url: &str,
    local_ip: Ipv4Addr,
    source_port: u16,
    identity: &str,
    observation: Option<(Arc<Stats>, bool, usize)>,
) -> Result<Socket, String> {
    let socket = TcpSocket::new_v4().map_err(|e| e.to_string())?;
    socket.set_reuseaddr(true).map_err(|e| e.to_string())?;
    socket2::SockRef::from(&socket)
        .set_linger(Some(Duration::ZERO))
        .map_err(|e| e.to_string())?;
    socket
        .bind((local_ip, source_port).into())
        .map_err(|e| e.to_string())?;
    let stream = observed(observation.as_ref(), "tcp-connect", async {
        socket.connect(address).await.map_err(|e| e.to_string())
    })
    .await?;
    stream.set_nodelay(true).map_err(|e| e.to_string())?;
    socket2::SockRef::from(&stream)
        .set_linger(Some(Duration::ZERO))
        .map_err(|e| e.to_string())?;
    let (mut ws, _) = observed(observation.as_ref(), "http-upgrade", async {
        client_async(url, stream).await.map_err(|e| e.to_string())
    })
    .await?;
    observed(observation.as_ref(), "identity-ack", async {
        if let Some((stats, _, index)) = observation.as_ref() {
            stats.stage(*index, "identity-send");
        }
        ws.send(Message::Text(
            json!({"type": "et-connect", "agent_id": identity})
                .to_string()
                .into(),
        ))
        .await
        .map_err(|e| e.to_string())?;
        if let Some((stats, _, index)) = observation.as_ref() {
            stats.stage(*index, "identity-ack");
        }
        while let Some(frame) = ws.next().await {
            match frame.map_err(|e| e.to_string())? {
                Message::Text(text) => {
                    let message: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
                    if message["type"] == "et-connect-ack" {
                        if message["agent_id"] != identity {
                            return Err("registration identity mismatch".into());
                        }
                        return Ok(ws);
                    }
                    if message["type"] == "et-invalid" {
                        return Err(message.to_string());
                    }
                }
                Message::Close(reason) => {
                    return Err(format!("closed before registration: {reason:?}"));
                }
                _ => {}
            }
        }
        Err("connection ended before registration".into())
    })
    .await
}

#[derive(Clone)]
struct Admission {
    initial: Arc<Semaphore>,
    retry: Arc<Semaphore>,
    capacity: Arc<Semaphore>,
    total: usize,
    reserved: usize,
    borrow: bool,
    metrics: Arc<AdmissionMetrics>,
}

#[derive(Default)]
struct AdmissionMetrics {
    gate: Mutex<()>,
    active: AtomicUsize,
    active_retry: AtomicUsize,
    queued_initial: AtomicUsize,
    queued_retry: AtomicUsize,
    peak_active: AtomicUsize,
    peak_retry: AtomicUsize,
    peak_queued_retry: AtomicUsize,
}

struct QueuedAdmission {
    metrics: Arc<AdmissionMetrics>,
    repeated: bool,
    queued: bool,
}
impl Drop for QueuedAdmission {
    fn drop(&mut self) {
        if !self.queued {
            return;
        }
        let _gate = self.metrics.gate.lock().unwrap();
        let count = if self.repeated {
            &self.metrics.queued_retry
        } else {
            &self.metrics.queued_initial
        };
        count.fetch_sub(1, Ordering::Relaxed);
    }
}

struct Admitted {
    _class: Option<OwnedSemaphorePermit>,
    _capacity: Option<OwnedSemaphorePermit>,
    metrics: Arc<AdmissionMetrics>,
    repeated: bool,
}
impl Drop for Admitted {
    fn drop(&mut self) {
        let _gate = self.metrics.gate.lock().unwrap();
        self.metrics.active.fetch_sub(1, Ordering::Relaxed);
        if self.repeated {
            self.metrics.active_retry.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Admission {
    fn new(total: usize, reserved: usize) -> Result<Self, String> {
        if total == 0 || reserved >= total {
            return Err("invalid admission limits".into());
        }
        let initial = Arc::new(Semaphore::new(total - reserved));
        let retry = if reserved == 0 {
            Arc::clone(&initial)
        } else {
            Arc::new(Semaphore::new(reserved))
        };
        Ok(Self {
            initial,
            retry,
            capacity: Arc::new(Semaphore::new(total)),
            total,
            reserved,
            borrow: false,
            metrics: Arc::new(AdmissionMetrics::default()),
        })
    }
    #[cfg(all(test, target_os = "linux"))]
    fn gate(&self, repeated: bool) -> &Semaphore {
        if repeated { &self.retry } else { &self.initial }
    }
    async fn acquire(&self, repeated: bool) -> Admitted {
        let count = if repeated {
            &self.metrics.queued_retry
        } else {
            &self.metrics.queued_initial
        };
        {
            let _gate = self.metrics.gate.lock().unwrap();
            let queued = count.fetch_add(1, Ordering::Relaxed) + 1;
            if repeated {
                self.metrics
                    .peak_queued_retry
                    .fetch_max(queued, Ordering::Relaxed);
            }
        }
        let mut waiting = QueuedAdmission {
            metrics: Arc::clone(&self.metrics),
            repeated,
            queued: true,
        };
        // Acquire the initial-class limit before joining the global queue. Otherwise the whole initial fleet
        // could queue ahead of retries again, defeating reservation even with a global semaphore.
        let class = if self.borrow && repeated {
            None
        } else {
            let gate = if repeated { &self.retry } else { &self.initial };
            Some(Arc::clone(gate).acquire_owned().await.unwrap())
        };
        let capacity = if self.borrow {
            Some(Arc::clone(&self.capacity).acquire_owned().await.unwrap())
        } else {
            None
        };
        let _gate = self.metrics.gate.lock().unwrap();
        count.fetch_sub(1, Ordering::Relaxed);
        waiting.queued = false;
        let active = self.metrics.active.fetch_add(1, Ordering::Relaxed) + 1;
        self.metrics
            .peak_active
            .fetch_max(active, Ordering::Relaxed);
        if repeated {
            let active = self.metrics.active_retry.fetch_add(1, Ordering::Relaxed) + 1;
            self.metrics.peak_retry.fetch_max(active, Ordering::Relaxed);
        }
        Admitted {
            _class: class,
            _capacity: capacity,
            metrics: Arc::clone(&self.metrics),
            repeated,
        }
    }
    fn snapshot(&self) -> Value {
        let _gate = self.metrics.gate.lock().unwrap();
        json!({"total_slots":self.total,"reserved_retry_slots":self.reserved,
            "mode":if self.borrow { "borrowing" } else { "fixed" },
            "active":self.metrics.active.load(Ordering::Relaxed),
            "active_retry":self.metrics.active_retry.load(Ordering::Relaxed),
            "queued_initial":self.metrics.queued_initial.load(Ordering::Relaxed),
            "queued_retry":self.metrics.queued_retry.load(Ordering::Relaxed),
            "peak_active":self.metrics.peak_active.load(Ordering::Relaxed),
            "peak_active_retry":self.metrics.peak_retry.load(Ordering::Relaxed),
            "peak_queued_retry":self.metrics.peak_queued_retry.load(Ordering::Relaxed)})
    }
}

struct Connected(Arc<Stats>, Option<usize>, bool);
impl Connected {
    #[cfg(all(test, target_os = "linux"))]
    fn new(stats: Arc<Stats>) -> Self {
        Self::for_vessel(stats, None)
    }
    #[cfg(test)]
    fn for_vessel(stats: Arc<Stats>, index: Option<usize>) -> Self {
        let gate = stats.cutoff_gate.lock().unwrap();
        let connected = Self::insert(Arc::clone(&stats), index, Instant::now());
        drop(gate);
        connected
    }
    // Caller holds cutoff_gate so connection and registration bookkeeping share one boundary.
    fn insert(stats: Arc<Stats>, index: Option<usize>, now: Instant) -> Self {
        let mut meter = stats.presence.lock().unwrap();
        meter.update(now);
        meter.current += 1;
        if let (Some(book), Some(index)) = (meter.diagnostics.as_mut(), index) {
            if book.accepts(now) {
                let state = &mut book.states[index];
                state.connected = true;
                state.stage = "heartbeat";
                state.stage_started = now;
                if let Some(followed) = book.cohort.get_mut(&index) {
                    followed.registered.get_or_insert(now);
                }
            }
        }
        stats.connected.store(meter.current, Ordering::Relaxed);
        drop(meter);
        Self(stats, index, true)
    }

    fn end_session(mut self, new_outage: bool, error: Option<HeartbeatError>) {
        let stats = Arc::clone(&self.0);
        let _gate = stats.cutoff_gate.lock().unwrap();
        self.remove_locked();
        if let (Some(index), Some((kind, detail, silence))) = (self.1, error) {
            if let Some(silence) = silence {
                stats.timing_locked("response-silence-at-disconnect", silence);
            }
            stats.error_locked(kind, index, detail);
        }
        if new_outage {
            stats.recovery_started.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn remove_locked(&mut self) {
        self.2 = false;
        let mut meter = self.0.presence.lock().unwrap();
        let now = Instant::now();
        meter.update(now);
        meter.current -= 1;
        if let (Some(book), Some(index)) = (meter.diagnostics.as_mut(), self.1) {
            if book.accepts(now) {
                let state = &mut book.states[index];
                state.connected = false;
                state.stage = "disconnected";
                state.stage_started = now;
            }
        }
        if meter.active {
            meter.minimum = meter.minimum.min(meter.current);
        }
        self.0.connected.store(meter.current, Ordering::Relaxed);
    }
}
impl Drop for Connected {
    fn drop(&mut self) {
        if self.2 {
            let stats = Arc::clone(&self.0);
            let _gate = stats.cutoff_gate.lock().unwrap();
            self.remove_locked();
        }
    }
}

async fn retry_deadline(stats: &Stats, initial: tokio::time::Instant) {
    loop {
        let notified = stats.hold_ready.notified();
        let boundary = if stats.measurement_started.load(Ordering::Acquire) {
            *stats.hold_deadline.lock().unwrap()
        } else {
            Some(initial)
        };
        if let Some(boundary) = boundary {
            tokio::select! {
                _ = tokio::time::sleep_until(boundary) => {
                    if registration_expired(stats, initial, true) { return; }
                },
                _ = notified => {},
            }
        } else {
            notified.await;
        }
    }
}

fn registration_expired(stats: &Stats, initial: tokio::time::Instant, retry: bool) -> bool {
    active_registration_deadline(stats, initial, retry)
        .is_some_and(|end| tokio::time::Instant::now() >= end)
}

fn active_registration_deadline(
    stats: &Stats,
    initial: tokio::time::Instant,
    retry: bool,
) -> Option<tokio::time::Instant> {
    if retry && stats.measurement_started.load(Ordering::Acquire) {
        *stats.hold_deadline.lock().unwrap()
    } else {
        Some(initial)
    }
}

fn cancel_registration(stats: &Stats, first: bool, retry: bool) {
    if !retry {
        stats.registration_censored.store(true, Ordering::Relaxed);
    }
    if first {
        stats
            .cancelled_registrations
            .fetch_add(1, Ordering::Relaxed);
    }
}

fn registration_cohort(stats: &Stats) -> (usize, usize) {
    let _gate = stats.cutoff_gate.lock().unwrap();
    (
        stats.registered.load(Ordering::Relaxed),
        stats.connected.load(Ordering::Relaxed),
    )
}

async fn agent(
    index: usize,
    base: u16,
    address: SocketAddr,
    local_ip: Ipv4Addr,
    url: Arc<String>,
    admission: Admission,
    stats: Arc<Stats>,
    deadline: tokio::time::Instant,
    retry: RetryPolicy,
) {
    let mut first_result = true;
    let mut attempted = false;
    let mut registered = false;
    let mut attempt = 0_usize;
    let mut failures = 0_u32;
    let mut outage = None;
    let identity = format!("vessel-{index:05}");
    loop {
        let source = if retry.enabled {
            retry.ips[attempt % retry.ips.len()]
        } else {
            local_ip
        };
        let operation = async {
            let repeated = attempted;
            stats.stage(index, "admission-queue");
            let waiting = Instant::now();
            let _permit = admission.acquire(repeated).await;
            stats.timing(
                if repeated {
                    "retry:queue"
                } else {
                    "initial:queue"
                },
                waiting.elapsed(),
            );
            {
                let _gate = stats.cutoff_gate.lock().unwrap();
                if registration_expired(&stats, deadline, retry.enabled) {
                    return None;
                }
                if !attempted {
                    attempted = true;
                    stats.attempted.fetch_add(1, Ordering::Relaxed);
                }
                stats.registration_attempts.fetch_add(1, Ordering::Relaxed);
                if stats.diagnostic_enabled.load(Ordering::Relaxed) {
                    if let Some(book) = stats.presence.lock().unwrap().diagnostics.as_mut() {
                        if book.accepts(Instant::now()) {
                            let state = &mut book.states[index];
                            state.attempts += 1;
                            state.source = Some(format!("{source}:{}", base + index as u16));
                        }
                    }
                }
                if repeated {
                    stats.retry_attempts.fetch_add(1, Ordering::Relaxed);
                }
            }
            let began = Instant::now();
            let registration = register_observed(
                address,
                &url,
                source,
                base + index as u16,
                &identity,
                Some((Arc::clone(&stats), repeated, index)),
            );
            let result = if retry.enabled {
                let result =
                    tokio::time::timeout(Duration::from_secs(retry.attempt_seconds), registration)
                        .await
                        .unwrap_or_else(|_| Err("registration attempt timed out".into()));
                stats.timing(
                    if repeated {
                        "retry:attempt"
                    } else {
                        "initial:attempt"
                    },
                    began.elapsed(),
                );
                result
            } else {
                registration.await
            };
            Some(result)
        };
        let result = tokio::select! {
            result = operation => result,
            _ = retry_deadline(&stats, deadline), if retry.enabled => None,
            _ = tokio::time::sleep_until(deadline), if !retry.enabled => None,
        };
        match result {
            Some(Ok(ws)) => {
                let connected = {
                    let _gate = stats.cutoff_gate.lock().unwrap();
                    let committed_at = Instant::now();
                    let expired = active_registration_deadline(&stats, deadline, retry.enabled)
                        .is_some_and(|limit| tokio::time::Instant::from_std(committed_at) >= limit);
                    if expired {
                        cancel_registration(&stats, first_result, retry.enabled);
                        if first_result {
                            stats.finished.fetch_add(1, Ordering::Release);
                        }
                        return;
                    }
                    stats.registration_successes.fetch_add(1, Ordering::Relaxed);
                    if !registered {
                        stats.registered.fetch_add(1, Ordering::Release);
                        registered = true;
                    } else {
                        stats.reconnections.fetch_add(1, Ordering::Relaxed);
                        if let Some(started) = outage {
                            stats.timing_locked(
                                "reconnect-ready",
                                Instant::now().duration_since(started),
                            );
                            stats.reconnect_ms.lock().unwrap().push(
                                Instant::now().duration_since(started).as_secs_f64() * 1000.0,
                            );
                        }
                    }
                    if first_result {
                        stats.finished.fetch_add(1, Ordering::Release);
                        first_result = false;
                    }
                    Connected::insert(Arc::clone(&stats), Some(index), committed_at)
                };
                let (useful, error) = heartbeat_session(
                    ws,
                    index,
                    Arc::clone(&stats),
                    outage,
                    retry.response_timeout,
                )
                .await;
                if useful {
                    outage = None;
                    failures = 0;
                }
                let new_outage = outage.is_none();
                if new_outage {
                    outage = Some(Instant::now());
                }
                connected.end_session(new_outage, Some(error));
            }
            Some(Err(detail)) => {
                stats.error("registration", index, detail);
            }
            None => {
                let _gate = stats.cutoff_gate.lock().unwrap();
                cancel_registration(&stats, first_result, retry.enabled);
            }
        }
        if first_result {
            let _gate = stats.cutoff_gate.lock().unwrap();
            stats.finished.fetch_add(1, Ordering::Release);
        }
        first_result = false;
        if stats
            .hold_deadline
            .lock()
            .unwrap()
            .is_some_and(|end| tokio::time::Instant::now() >= end)
        {
            return;
        }
        if !retry.enabled {
            return;
        }
        stats.stage(index, "backoff");
        tokio::time::sleep(retry.delay(failures, index)).await;
        failures = failures.saturating_add(1);
        attempt += 1;
    }
}

async fn heartbeat_session(
    mut ws: Socket,
    index: usize,
    stats: Arc<Stats>,
    mut recovery: Option<Instant>,
    response_timeout: Option<Duration>,
) -> (bool, HeartbeatError) {
    let mut useful = false;
    let mut outstanding_pings = 0usize;
    let mut last_pong = Instant::now();
    let mut response_deadline =
        tokio::time::Instant::now() + response_timeout.unwrap_or(Duration::from_secs(86400));
    let mut beat = tokio::time::interval(Duration::from_secs(5));
    beat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    beat.tick().await;
    let terminal = loop {
        tokio::select! {
            _ = tokio::time::sleep_until(response_deadline), if response_timeout.is_some() => {
                break (
                    "heartbeat-response-timeout",
                    format!("no heartbeat reply for {:?}", response_timeout.unwrap()),
                    Some(last_pong.elapsed()),
                );
            }
            scheduled = beat.tick() => {
                let started = Instant::now();
                {
                let _gate = stats.cutoff_gate.lock().unwrap();
                if stats.holding.load(Ordering::Relaxed) {
                    stats.heartbeat_lag_ms.lock().unwrap().push(
                        tokio::time::Instant::now().saturating_duration_since(scheduled).as_secs_f64()*1000.0);
                }
                }
                let sending = ws.send(Message::Ping(Default::default()));
                let sent = if response_timeout.is_some() {
                    match tokio::time::timeout_at(response_deadline, sending).await {
                        Ok(value) => value,
                        Err(_) => {
                            break (
                                "heartbeat-response-timeout",
                                "heartbeat send blocked past response deadline".into(),
                                Some(last_pong.elapsed()),
                            );
                        }
                    }
                } else { sending.await };
                if let Err(error) = sent {
                    break ("heartbeat-send", error.to_string(), None);
                }
                outstanding_pings += 1;
                let _gate = stats.cutoff_gate.lock().unwrap();
                stats.sent_pings.fetch_add(1, Ordering::Relaxed);
                if stats.holding.load(Ordering::Relaxed) {
                    stats.hold_pings.fetch_add(1, Ordering::Relaxed);
                    stats.send_ms.lock().unwrap().push(started.elapsed().as_secs_f64()*1000.0);
                }
            }
            frame = ws.next() => {
                match frame {
                    Some(Ok(Message::Pong(_))) => {
                        if outstanding_pings == 0 {
                            continue;
                        }
                        outstanding_pings -= 1;
                        let _gate = stats.cutoff_gate.lock().unwrap();
                        stats.record_reply(index);
                        let previous = stats.last_vessel_pong.lock().unwrap().insert(index, Instant::now());
                        if let Some(previous) = previous { stats.timing_locked("vessel-pong-gap", previous.elapsed()); }
                        stats.timing_locked("pong-gap", last_pong.elapsed());
                        last_pong = Instant::now();
                        response_deadline = tokio::time::Instant::now()
                            + response_timeout.unwrap_or(Duration::from_secs(86400));
                        stats.pongs.fetch_add(1, Ordering::Relaxed);
                        if stats.holding.load(Ordering::Relaxed) { stats.hold_pongs.fetch_add(1, Ordering::Relaxed); }
                        useful = true;
                        if let Some(started) = recovery.take() {
                            stats.timing_locked("useful-recovery", started.elapsed());
                            stats.useful_recovery_ms.lock().unwrap().push(started.elapsed().as_secs_f64()*1000.0);
                            stats.recovery_completed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Some(Ok(Message::Close(reason))) => {
                        break ("heartbeat-close", format!("{reason:?}"), None);
                    }
                    Some(Ok(_)) => {}
                    Some(Err(error)) => { break ("heartbeat-read", error.to_string(), None); }
                    None => { break ("heartbeat-read", "stream ended".into(), None); }
                }
            }
        }
    };
    (useful, terminal)
}

fn record_timing_pong(stats: &Stats, pending: &mut HashMap<u64, Instant>, payload: &[u8]) {
    let Ok(bytes) = <[u8; 8]>::try_from(payload) else {
        return;
    };
    if let Some(started) = pending.remove(&u64::from_be_bytes(bytes)) {
        let _gate = stats.cutoff_gate.lock().unwrap();
        stats.timing_matched.fetch_add(1, Ordering::Relaxed);
        stats
            .timing_ms
            .lock()
            .unwrap()
            .push(started.elapsed().as_secs_f64() * 1000.0);
    }
}

async fn timing_probe(
    index: usize,
    address: SocketAddr,
    local_ip: Ipv4Addr,
    url: Arc<String>,
    stats: Arc<Stats>,
) {
    let mut socket = match register(
        address,
        &url,
        local_ip,
        45000 + index as u16,
        &format!("identified-timing-{index}"),
    )
    .await
    {
        Ok(value) => value,
        Err(error) => {
            stats.error("timing-registration", index, error);
            return;
        }
    };
    {
        let _gate = stats.cutoff_gate.lock().unwrap();
        stats.timing_registered.fetch_add(1, Ordering::Relaxed);
    }
    let mut beat = tokio::time::interval(Duration::from_secs(5));
    beat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    beat.tick().await;
    let mut sequence = 0_u64;
    let mut pending = HashMap::new();
    loop {
        tokio::select! {
            _ = beat.tick() => {
                sequence += 1;
                pending.insert(sequence, Instant::now());
                if let Err(error) = socket.send(Message::Ping(sequence.to_be_bytes().to_vec().into())).await {
                    stats.error("timing-send", index, error.to_string()); return;
                }
                let _gate = stats.cutoff_gate.lock().unwrap();
                stats.timing_sent.fetch_add(1, Ordering::Relaxed);
            }
            frame = socket.next() => {
                match frame {
                    Some(Ok(Message::Pong(payload))) => record_timing_pong(&stats, &mut pending, &payload),
                    Some(Ok(Message::Close(reason))) => {
                        stats.error("timing-close", index, format!("{reason:?}")); return;
                    }
                    Some(Ok(_)) => {}
                    Some(Err(error)) => { stats.error("timing-read", index, error.to_string()); return; }
                    None => { stats.error("timing-read", index, "stream ended".into()); return; }
                }
            }
        }
    }
}

fn distribution(values: &mut [f64]) -> Value {
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return json!({"count": 0});
    }
    let quantile = |q: f64| values[(values.len() as f64 * q).ceil().max(1.0) as usize - 1];
    json!({"count": values.len(), "min_ms": values[0], "p50_ms": quantile(0.5),
        "p95_ms": quantile(0.95), "max_ms": values[values.len()-1]})
}

struct Cutoff {
    seconds: f64,
    connected: usize,
    minimum: usize,
    diagnostics: Option<Value>,
    at: Instant,
    report: Value,
}

fn capture_cutoff(stats: &Stats, began: Instant, count: usize, followup: u64) -> Cutoff {
    let gate = stats.cutoff_gate.lock().unwrap();
    stats.holding.store(false, Ordering::Relaxed);
    stats.phase.store(3, Ordering::Relaxed);
    let now = Instant::now();
    *stats.hold_deadline.lock().unwrap() = Some(tokio::time::Instant::from_std(
        now + Duration::from_secs(followup),
    ));
    let mut meter = stats.presence.lock().unwrap();
    meter.update(now);
    meter.active = false;
    let diagnostics = meter.diagnostics.as_mut().map(|book| {
        book.followup_end = Some(now + Duration::from_secs(followup));
        book.cutoff(now)
    });
    let (seconds, connected, minimum) = (meter.seconds, meter.current, meter.minimum);
    drop(meter);
    let mut report = stats.sample_locked(began.elapsed().as_secs_f64(), "finished");
    report["unimpaired_control"] = stats.control_status.lock().unwrap().clone();
    let mut silence: Vec<f64> = stats
        .last_vessel_pong
        .lock()
        .unwrap()
        .values()
        .map(|at| now.saturating_duration_since(*at).as_secs_f64() * 1000.0)
        .collect();
    report["vessels_with_reply_within_30s_at_end"] =
        json!(silence.iter().filter(|age| **age <= 30000.0).count());
    report["vessels_without_any_reply"] = json!(count.saturating_sub(silence.len()));
    let vectors: Vec<_> = [
        ("identified_probe_rtt", &stats.timing_ms),
        ("heartbeat_scheduling_lag", &stats.heartbeat_lag_ms),
        ("heartbeat_send_duration", &stats.send_ms),
        ("runtime_scheduling_lag", &stats.runtime_lag_ms),
        ("unimpaired_alive_response", &stats.alive_ms),
        ("reconnect_ready_time", &stats.reconnect_ms),
        ("useful_recovery_time", &stats.useful_recovery_ms),
    ]
    .into_iter()
    .map(|(key, values)| (key, values.lock().unwrap().clone()))
    .collect();
    report["timing_unanswered"] = json!(
        report["timing_sent"]
            .as_u64()
            .unwrap()
            .saturating_sub(report["timing_matched"].as_u64().unwrap())
    );
    report["unresolved_disconnections"] = json!(
        stats
            .recovery_started
            .load(Ordering::Relaxed)
            .saturating_sub(stats.recovery_completed.load(Ordering::Relaxed))
    );
    let timings = stats.timings.lock().unwrap().clone();
    report["errors"] = json!(*stats.errors.lock().unwrap());
    report["error_counts"] = json!(*stats.error_counts.lock().unwrap());
    drop(gate);
    stats.hold_ready.notify_waiters();
    report["reply_silence_at_end"] = distribution(&mut silence);
    for (key, mut values) in vectors {
        report[key] = distribution(&mut values);
    }
    report["timings_by_completion_phase"] = json!(
        timings
            .into_iter()
            .map(|(name, mut values)| (name, distribution(&mut values)))
            .collect::<HashMap<_, _>>()
    );
    Cutoff {
        seconds,
        connected,
        minimum,
        diagnostics,
        at: now,
        report,
    }
}

fn valid_ports(count: usize, base: u16, timing_count: usize) -> bool {
    count > 0
        && base > 0
        && count <= 25000
        && base as usize + count <= 50000
        && timing_count > 0
        && timing_count <= 32
        && !((base as usize) < 45000 + timing_count && base as usize + count > 45000)
}

fn watchdog_timeout(seconds: u64, reconnect: bool) -> Result<Option<Duration>, String> {
    if seconds == 0 {
        return Ok(None);
    }
    if !reconnect || seconds <= 5 {
        return Err("response watchdog requires reconnects and a timeout greater than the 5s heartbeat interval".into());
    }
    Ok(Some(Duration::from_secs(seconds)))
}

fn set_control_status(stats: &Stats, registered: bool, status: &str, error: Option<String>) {
    let _gate = stats.cutoff_gate.lock().unwrap();
    *stats.control_status.lock().unwrap() =
        json!({"registered":registered,"status":status,"error":error});
}

fn observe_cohort(gate_file: &str, stats: &Stats, began: Instant) -> Result<(), String> {
    let parent = Path::new(gate_file)
        .parent()
        .ok_or("hold gate has no parent")?;
    let request = parent.join("cohort-request.json");
    if !request.exists() {
        return Ok(());
    }
    let request: Value = serde_json::from_slice(&fs::read(request).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let row = {
        let _gate = stats.cutoff_gate.lock().unwrap();
        json!({"request":request["request"],
            "cohort":stats.sample_locked(began.elapsed().as_secs_f64(), "awaiting-impairment")})
    };
    let temporary = parent.join("cohort-observation.tmp");
    fs::write(&temporary, serde_json::to_vec(&row).unwrap()).map_err(|e| e.to_string())?;
    fs::rename(temporary, parent.join("cohort-observation.json")).map_err(|e| e.to_string())?;
    Ok(())
}

async fn run(args: HashMap<String, String>) -> Result<(), String> {
    let host = args.get("server").map(String::as_str).unwrap_or("hub:8080");
    let address = tokio::net::lookup_host(host)
        .await
        .map_err(|e| e.to_string())?
        .find(SocketAddr::is_ipv4)
        .ok_or("no IPv4 server address")?;
    let url = Arc::new(format!("ws://{host}/ws"));
    // Select the routed container address without sending a datagram. Binding 0.0.0.0
    // would also claim Docker DNS's 127.0.0.11 listener port when it overlaps the fleet.
    let route = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).map_err(|e| e.to_string())?;
    route.connect(address).map_err(|e| e.to_string())?;
    let IpAddr::V4(local_ip) = route.local_addr().map_err(|e| e.to_string())?.ip() else {
        return Err("no IPv4 source address".into());
    };
    drop(route);
    let count: usize = args
        .get("agents")
        .map(String::as_str)
        .unwrap_or("25000")
        .parse()
        .map_err(|_| "invalid agents")?;
    let base: u16 = args
        .get("source-base")
        .map(String::as_str)
        .unwrap_or("20000")
        .parse()
        .map_err(|_| "invalid source-base")?;
    if count == 0 || count > 25000 || base as usize + count > 50000 {
        return Err("invalid fleet/source range".into());
    }
    let timing_count: usize = args
        .get("timing-probes")
        .map(String::as_str)
        .unwrap_or("8")
        .parse()
        .map_err(|_| "invalid timing-probes")?;
    if !valid_ports(count, base, timing_count) {
        return Err("invalid or overlapping fleet/timing-probe source ports".into());
    }
    if let Some(bytes) = args.get("verify-bytes") {
        let bytes: usize = bytes.parse().map_err(|_| "invalid verify-bytes")?;
        let mut socket = register(address, &url, local_ip, base, "vessel-00000").await?;
        let started = Instant::now();
        socket
            .send(Message::Text(
                json!({"type": "et-alive", "timestamp": "x".repeat(bytes)})
                    .to_string()
                    .into(),
            ))
            .await
            .map_err(|e| e.to_string())?;
        while let Some(frame) = socket.next().await {
            if let Message::Text(text) = frame.map_err(|e| e.to_string())? {
                let value: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
                if value["type"] == "et-response" {
                    println!(
                        "{}",
                        json!({
                            "type": "verification",
                            "application_bytes": bytes,
                            "seconds": started.elapsed().as_secs_f64()
                        })
                    );
                    return Ok(());
                }
            }
        }
        return Err("verification connection ended".into());
    }
    let hold: u64 = args
        .get("hold-seconds")
        .map(String::as_str)
        .unwrap_or("120")
        .parse()
        .map_err(|_| "invalid hold")?;
    let setup: u64 = args
        .get("setup-seconds")
        .map(String::as_str)
        .unwrap_or("1800")
        .parse()
        .map_err(|_| "invalid setup")?;
    let stats = Arc::new(Stats::default());
    let diagnostic = args.get("diagnostics").is_some_and(|value| value == "true");
    let followup: u64 = args
        .get("followup-seconds")
        .map(String::as_str)
        .unwrap_or("0")
        .parse()
        .map_err(|_| "invalid followup-seconds")?;
    if followup > 0 && !diagnostic {
        return Err("follow-up requires diagnostics".into());
    }
    stats
        .diagnostic_enabled
        .store(diagnostic, Ordering::Relaxed);
    let mut retry = RetryPolicy::disabled(local_ip);
    retry.enabled = args.get("reconnect").is_some_and(|value| value == "true");
    if let Some(value) = args.get("response-timeout-seconds") {
        let seconds: u64 = value
            .parse()
            .map_err(|_| "invalid response-timeout-seconds")?;
        retry.response_timeout = watchdog_timeout(seconds, retry.enabled)?;
    }
    for (key, target) in [
        ("attempt-seconds", &mut retry.attempt_seconds),
        ("backoff-base-ms", &mut retry.base_ms),
        ("backoff-max-ms", &mut retry.max_ms),
        ("retry-seed", &mut retry.seed),
    ] {
        if let Some(value) = args.get(key) {
            *target = value.parse().map_err(|_| format!("invalid {key}"))?;
        }
    }
    if retry.attempt_seconds == 0 || retry.base_ms == 0 || retry.max_ms < retry.base_ms {
        return Err("invalid retry limits".into());
    }
    if let Some(ips) = args.get("source-ips") {
        retry.ips = ips
            .split(',')
            .map(str::parse)
            .collect::<Result<Vec<Ipv4Addr>, _>>()
            .map_err(|e| e.to_string())?;
    }
    let total: usize = args
        .get("admission-slots")
        .map(String::as_str)
        .unwrap_or("100")
        .parse()
        .map_err(|_| "invalid admission-slots")?;
    let reserved: usize = args
        .get("retry-slots")
        .map(String::as_str)
        .unwrap_or("0")
        .parse()
        .map_err(|_| "invalid retry-slots")?;
    let mut admission = Admission::new(total, reserved)?;
    admission.borrow = args
        .get("borrow-admission")
        .is_some_and(|value| value == "true");
    *stats.admission.lock().unwrap() = Some(admission.clone());
    let mut agents = JoinSet::new();
    let began = Instant::now();
    if diagnostic {
        stats.presence.lock().unwrap().diagnostics = Some(DiagnosticBook::new(count, began));
    }
    let deadline = tokio::time::Instant::now() + Duration::from_secs(setup);
    let lag_stats = Arc::clone(&stats);
    let lag_monitor = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(100));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        interval.tick().await;
        loop {
            let scheduled = interval.tick().await;
            let _gate = lag_stats.cutoff_gate.lock().unwrap();
            lag_stats.runtime_lag_ms.lock().unwrap().push(
                tokio::time::Instant::now()
                    .saturating_duration_since(scheduled)
                    .as_secs_f64()
                    * 1000.0,
            );
        }
    });
    for index in 0..count {
        agents.spawn(agent(
            index,
            base,
            address,
            local_ip,
            Arc::clone(&url),
            admission.clone(),
            Arc::clone(&stats),
            deadline,
            retry.clone(),
        ));
    }
    while if retry.enabled {
        stats.registered.load(Ordering::Acquire) < count && began.elapsed().as_secs() < setup
    } else {
        stats.finished.load(Ordering::Acquire) < count
    } {
        println!(
            "{}",
            stats.sample(began.elapsed().as_secs_f64(), "registration")
        );
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    let setup_seconds = began.elapsed().as_secs_f64();
    let registration_censored = if retry.enabled {
        stats.registered.load(Ordering::Acquire) < count
    } else {
        stats.registration_censored.load(Ordering::Relaxed)
    };
    let (registered_at_hold, connected_at_gate) = registration_cohort(&stats);
    stats.measurement_started.store(true, Ordering::Release);
    stats.hold_ready.notify_waiters();
    if let Some(file) = args.get("hold-gate") {
        if registered_at_hold != count || connected_at_gate != count {
            return Err(
                "Pre-registration cohort is incomplete; refusing to start the impairment window"
                    .into(),
            );
        }
        while !Path::new(file).exists() {
            observe_cohort(file, &stats, began)?;
            println!(
                "{}",
                stats.sample(began.elapsed().as_secs_f64(), "awaiting-impairment")
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        if stats.connected.load(Ordering::Relaxed) != count {
            return Err(
                "Cohort incomplete when impairment gate opened; refusing the window".into(),
            );
        }
    }
    let settling: u64 = args
        .get("settle-seconds")
        .map(String::as_str)
        .unwrap_or("0")
        .parse()
        .map_err(|_| "invalid settle-seconds")?;
    let before_settling = stats.sample(began.elapsed().as_secs_f64(), "before-settling");
    stats.phase.store(1, Ordering::Relaxed);
    let settle_start = Instant::now();
    while settle_start.elapsed().as_secs() < settling {
        println!(
            "{}",
            stats.sample(began.elapsed().as_secs_f64(), "settling")
        );
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    let (before_hold, hold_start) = {
        let _gate = stats.cutoff_gate.lock().unwrap();
        if args.contains_key("hold-gate") && stats.connected.load(Ordering::Relaxed) != count {
            return Err("Cohort incomplete at hold start; refusing the window".into());
        }
        let before = stats.sample_locked(began.elapsed().as_secs_f64(), "before-hold");
        stats.phase.store(2, Ordering::Relaxed);
        let started = Instant::now();
        let mut meter = stats.presence.lock().unwrap();
        meter.last = started;
        meter.active = true;
        meter.seconds = 0.0;
        meter.minimum = meter.current;
        stats.holding.store(true, Ordering::Relaxed);
        (before, started)
    };
    let cutoff_deadline = tokio::time::Instant::now() + Duration::from_secs(hold);
    let mut minimum_connected = stats.connected.load(Ordering::Relaxed);
    let mut timing_tasks = JoinSet::new();
    for index in 0..timing_count {
        timing_tasks.spawn(timing_probe(
            index,
            address,
            local_ip,
            Arc::clone(&url),
            Arc::clone(&stats),
        ));
    }
    let probe_stats = Arc::clone(&stats);
    let probe_url = Arc::clone(&url);
    set_control_status(&stats, false, "registering", None);
    let probe = tokio::spawn(async move {
        match register(address, &probe_url, local_ip, 50001, "unimpaired-hub-probe").await {
            Err(error) => {
                set_control_status(&probe_stats, false, "registration-error", Some(error))
            }
            Ok(mut socket) => {
                set_control_status(&probe_stats, true, "connected", None);
                loop {
                    let started = Instant::now();
                    if let Err(error) = socket
                        .send(Message::Text(
                            json!({"type":"et-alive", "timestamp":"bench"})
                                .to_string()
                                .into(),
                        ))
                        .await
                    {
                        set_control_status(
                            &probe_stats,
                            true,
                            "send-error",
                            Some(error.to_string()),
                        );
                        break;
                    }
                    loop {
                        match socket.next().await {
                            Some(Ok(Message::Text(text))) => {
                                if serde_json::from_str::<Value>(&text)
                                    .ok()
                                    .is_some_and(|v| v["type"] == "et-response")
                                {
                                    let _gate = probe_stats.cutoff_gate.lock().unwrap();
                                    probe_stats
                                        .alive_ms
                                        .lock()
                                        .unwrap()
                                        .push(started.elapsed().as_secs_f64() * 1000.0);
                                    break;
                                }
                            }
                            Some(Ok(Message::Close(reason))) => {
                                set_control_status(
                                    &probe_stats,
                                    true,
                                    "closed",
                                    Some(format!("{reason:?}")),
                                );
                                return;
                            }
                            Some(Ok(_)) => {}
                            Some(Err(error)) => {
                                set_control_status(
                                    &probe_stats,
                                    true,
                                    "read-error",
                                    Some(error.to_string()),
                                );
                                return;
                            }
                            None => {
                                set_control_status(
                                    &probe_stats,
                                    true,
                                    "ended",
                                    Some("stream ended".into()),
                                );
                                return;
                            }
                        }
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    });
    while tokio::time::Instant::now() < cutoff_deadline {
        minimum_connected = minimum_connected.min(stats.connected.load(Ordering::Relaxed));
        println!("{}", stats.sample(began.elapsed().as_secs_f64(), "holding"));
        tokio::time::sleep_until(
            (tokio::time::Instant::now() + Duration::from_secs(1)).min(cutoff_deadline),
        )
        .await;
    }
    minimum_connected = minimum_connected.min(stats.connected.load(Ordering::Relaxed));
    let cutoff = capture_cutoff(&stats, began, count, followup);
    let (connected_seconds, cutoff_connected, cutoff_diagnostics, cutoff_at, mut report) = (
        cutoff.seconds,
        cutoff.connected,
        cutoff.diagnostics,
        cutoff.at,
        cutoff.report,
    );
    minimum_connected = minimum_connected.min(cutoff.minimum);
    report["connected"] = json!(cutoff_connected);
    let observed_hold_seconds = cutoff_at.duration_since(hold_start).as_secs_f64();
    report["cutoff_diagnostics"] = json!(cutoff_diagnostics);
    if diagnostic {
        std::fs::write(
            "/results/cutoff-vessels.json",
            serde_json::to_vec_pretty(&report["cutoff_diagnostics"]).unwrap(),
        )
        .map_err(|error| error.to_string())?;
    }
    report["type"] = json!("report");
    report["requested_agents"] = json!(count);
    report["setup_seconds"] = json!(setup_seconds);
    report["before_settling"] = before_settling;
    report["registered_at_hold_start"] = before_hold["registered_total"].clone();
    report["before_hold"] = before_hold;
    report["settle_seconds"] = json!(settling);
    report["response_timeout_seconds"] = json!(retry.response_timeout.map(|value| value.as_secs()));
    report["registration_attempt_timeout_seconds"] = json!(retry.attempt_seconds);
    report["registration_censored"] = json!(registration_censored);
    report["hold_seconds"] = json!(observed_hold_seconds);
    report["connected_vessel_seconds"] = json!(connected_seconds);
    report["connected_vessel_fraction"] =
        json!((connected_seconds / (count as f64 * observed_hold_seconds)).min(1.0));
    report["end_fleet_complete"] = json!(report["connected"] == json!(count));
    report["minimum_connected_during_hold"] = json!(minimum_connected);
    report["target_held"] = json!(!registration_censored && minimum_connected == count);
    report["timing_probes_requested"] = json!(timing_count);
    report["empty_heartbeat_rtt_measured"] = json!(false);
    report["followup_requested_seconds"] = json!(followup);
    let followup_end = tokio::time::Instant::from_std(cutoff_at + Duration::from_secs(followup));
    while tokio::time::Instant::now() < followup_end {
        println!(
            "{}",
            stats.sample(began.elapsed().as_secs_f64(), "diagnostic-followup")
        );
        tokio::time::sleep_until(
            (tokio::time::Instant::now() + Duration::from_secs(1)).min(followup_end),
        )
        .await;
    }
    if diagnostic {
        let followup = stats
            .presence
            .lock()
            .unwrap()
            .diagnostics
            .as_ref()
            .unwrap()
            .followup(cutoff_at + Duration::from_secs(followup));
        std::fs::write(
            "/results/cutoff-followup.json",
            serde_json::to_vec_pretty(&followup).unwrap(),
        )
        .map_err(|error| error.to_string())?;
        report["cutoff_followup"] = followup;
    }
    println!("{report}");
    probe.abort();
    lag_monitor.abort();
    timing_tasks.abort_all();
    agents.abort_all();
    while agents.join_next().await.is_some() {}
    Ok(())
}

fn main() {
    let mut args = HashMap::new();
    let mut values = std::env::args().skip(1);
    while let Some(key) = values.next() {
        let value = values.next().expect("each option needs a value");
        args.insert(key.trim_start_matches("--").to_owned(), value);
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    if let Err(error) = runtime.block_on(run(args)) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod measurement_tests {
    use super::*;

    #[test]
    fn cohort_observation_answers_the_current_request_with_current_counts() {
        let folder = std::env::temp_dir().join(format!(
            "cohort-{}-{:?}",
            std::process::id(),
            Instant::now()
        ));
        fs::create_dir(&folder).unwrap();
        let stats = Arc::new(Stats::default());
        let mut connected = Some(Connected::for_vessel(Arc::clone(&stats), None));
        let gate = folder.join("impairment-applied.json");
        for (request, expected) in [("before", 1), ("after", 0)] {
            if expected == 0 {
                drop(connected.take());
            }
            fs::write(
                folder.join("cohort-request.json"),
                json!({"request":request}).to_string(),
            )
            .unwrap();
            observe_cohort(gate.to_str().unwrap(), &stats, Instant::now()).unwrap();
            let row: Value =
                serde_json::from_slice(&fs::read(folder.join("cohort-observation.json")).unwrap())
                    .unwrap();
            assert_eq!(row["request"], request);
            assert_eq!(row["cohort"]["connected"], expected);
        }
        drop(connected);
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn control_failure_keeps_successful_samples_separate_from_fleet_errors() {
        let stats = Stats::default();
        set_control_status(&stats, true, "connected", None);
        stats.alive_ms.lock().unwrap().push(4.0);
        set_control_status(&stats, true, "read-error", Some("reset".into()));
        let cutoff = capture_cutoff(&stats, Instant::now(), 0, 0);
        assert_eq!(cutoff.report["unimpaired_control"]["status"], "read-error");
        assert_eq!(cutoff.report["unimpaired_control"]["registered"], true);
        assert_eq!(cutoff.report["failures"], 0);
        assert_eq!(stats.alive_ms.lock().unwrap().as_slice(), [4.0]);
        set_control_status(&stats, false, "registration-error", Some("refused".into()));
        let failed = capture_cutoff(&stats, Instant::now(), 0, 0);
        assert_eq!(failed.report["unimpaired_control"]["registered"], false);
    }

    #[test]
    fn registration_commit_uses_one_timestamp_for_presence_and_followup() {
        let stats = Arc::new(Stats::default());
        let began = Instant::now();
        let mut book = DiagnosticBook::new(1, began);
        book.cutoff(began);
        book.followup_end = Some(began + Duration::from_secs(1));
        stats.presence.lock().unwrap().diagnostics = Some(book);
        let committed = began + Duration::from_millis(1);
        let connected = {
            let _gate = stats.cutoff_gate.lock().unwrap();
            Connected::insert(Arc::clone(&stats), Some(0), committed)
        };
        let meter = stats.presence.lock().unwrap();
        assert_eq!(meter.last, committed);
        assert_eq!(
            meter.diagnostics.as_ref().unwrap().cohort[&0].registered,
            Some(committed)
        );
        drop(meter);
        drop(connected);
    }

    #[test]
    fn cohort_read_waits_for_the_last_registration_to_publish_connected() {
        let stats = Arc::new(Stats::default());
        let gate = stats.cutoff_gate.lock().unwrap();
        stats.registered.store(1, Ordering::Relaxed);
        let reader = Arc::clone(&stats);
        let (sender, receiver) = std::sync::mpsc::channel();
        let task = std::thread::spawn(move || sender.send(registration_cohort(&reader)).unwrap());
        assert!(receiver.recv_timeout(Duration::from_millis(20)).is_err());
        stats.connected.store(1, Ordering::Relaxed);
        drop(gate);
        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(1)).unwrap(),
            (1, 1)
        );
        task.join().unwrap();
    }

    #[tokio::test]
    async fn an_expired_ready_admission_cannot_start_a_dial() {
        let stats = Arc::new(Stats::default());
        agent(
            0,
            20000,
            (Ipv4Addr::LOCALHOST, 9).into(),
            Ipv4Addr::LOCALHOST,
            Arc::new("ws://127.0.0.1:9/ws".into()),
            Admission::new(1, 0).unwrap(),
            Arc::clone(&stats),
            tokio::time::Instant::now() - Duration::from_secs(1),
            RetryPolicy::disabled(Ipv4Addr::LOCALHOST),
        )
        .await;
        assert_eq!(stats.registration_attempts.load(Ordering::Relaxed), 0);
        assert_eq!(stats.registered.load(Ordering::Relaxed), 0);
        assert_eq!(stats.cancelled_registrations.load(Ordering::Relaxed), 1);
        assert_eq!(stats.failures.load(Ordering::Relaxed), 0);
        assert!(stats.registration_censored.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn admission_snapshot_keeps_active_retry_within_active_total() {
        let admission = Admission::new(4, 2).unwrap();
        let mut workers = JoinSet::new();
        for repeated in [false, true, false, true] {
            let admission = admission.clone();
            workers.spawn(async move {
                for _ in 0..500 {
                    let permit = admission.acquire(repeated).await;
                    tokio::task::yield_now().await;
                    drop(permit);
                }
            });
        }
        for _ in 0..2000 {
            let sample = admission.snapshot();
            assert!(sample["active_retry"].as_u64().unwrap() <= sample["active"].as_u64().unwrap());
            tokio::task::yield_now().await;
        }
        while workers.join_next().await.is_some() {}
        assert_eq!(admission.snapshot()["active"], 0);
    }

    #[test]
    fn quantiles_are_nondecreasing_for_small_samples() {
        assert_eq!(distribution(&mut []), json!({"count": 0}));
        for n in 1..12 {
            let mut values: Vec<f64> = (0..n).map(|i| i as f64 * 99.0 + 1.0).collect();
            let result = distribution(&mut values);
            assert!(result["min_ms"].as_f64().unwrap() <= result["p50_ms"].as_f64().unwrap());
            assert!(result["p50_ms"].as_f64().unwrap() <= result["p95_ms"].as_f64().unwrap());
            assert!(result["p95_ms"].as_f64().unwrap() <= result["max_ms"].as_f64().unwrap());
        }
        let pair = distribution(&mut [1.0, 100.0]);
        assert_eq!(pair["p50_ms"], 1.0);
        assert_eq!(pair["p95_ms"], 100.0);
    }

    #[test]
    fn fleet_and_probe_ranges_do_not_overlap() {
        assert!(!valid_ports(1, 0, 1));
        assert!(valid_ports(25000, 20000, 8));
        assert!(!valid_ports(25000, 20001, 1));
        assert!(!valid_ports(1, 45000, 1));
        assert!(valid_ports(1, 44999, 1));
    }

    #[test]
    fn watchdog_must_allow_the_first_heartbeat() {
        assert_eq!(watchdog_timeout(0, false).unwrap(), None);
        for seconds in [1, 5] {
            assert!(watchdog_timeout(seconds, true).is_err());
        }
        assert_eq!(
            watchdog_timeout(6, true).unwrap(),
            Some(Duration::from_secs(6))
        );
        assert!(watchdog_timeout(6, false).is_err());
    }

    #[test]
    fn actual_cutoff_sets_the_shared_followup_deadline() {
        let stats = Stats::default();
        let scheduled = Instant::now() - Duration::from_secs(2);
        let cutoff = capture_cutoff(&stats, scheduled, 1, 5);
        let expected = cutoff.at + Duration::from_secs(5);
        assert_eq!(
            stats.hold_deadline.lock().unwrap().unwrap().into_std(),
            expected
        );
        assert!(expected > scheduled + Duration::from_secs(5));
    }

    #[test]
    fn disconnect_and_new_outage_have_one_cutoff_state() {
        let stats = Arc::new(Stats::default());
        let first = Connected::for_vessel(Arc::clone(&stats), Some(0));
        first.end_session(true, None);
        let cutoff = capture_cutoff(&stats, Instant::now(), 1, 0);
        assert_eq!(cutoff.connected, 0);
        assert_eq!(cutoff.report["unresolved_disconnections"], 1);
        let replacement = Connected::for_vessel(Arc::clone(&stats), Some(0));
        replacement.end_session(false, None);
        assert_eq!(stats.recovery_started.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn retry_cancellation_waits_for_the_actual_cutoff_deadline() {
        let stats = Arc::new(Stats::default());
        stats.measurement_started.store(true, Ordering::Release);
        let task_stats = Arc::clone(&stats);
        let waiter = tokio::spawn(async move {
            retry_deadline(&task_stats, tokio::time::Instant::now()).await;
        });
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(!waiter.is_finished());
        let cutoff = capture_cutoff(&stats, Instant::now(), 1, 0);
        tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .unwrap()
            .unwrap();
        assert!(cutoff.at <= Instant::now());
    }

    #[tokio::test]
    async fn an_old_setup_timer_cannot_cancel_the_active_measurement_window() {
        for _ in 0..16 {
            let stats = Stats::default();
            let deadline = tokio::time::Instant::now() + Duration::from_millis(1);
            let waiting = retry_deadline(&stats, deadline);
            tokio::pin!(waiting);
            assert!(futures_util::poll!(waiting.as_mut()).is_pending());
            tokio::time::sleep(Duration::from_millis(3)).await;
            stats.measurement_started.store(true, Ordering::Release);
            stats.hold_ready.notify_waiters();
            assert!(futures_util::poll!(waiting.as_mut()).is_pending());
            *stats.hold_deadline.lock().unwrap() = Some(tokio::time::Instant::now());
            stats.hold_ready.notify_waiters();
            tokio::time::timeout(Duration::from_secs(1), waiting)
                .await
                .unwrap();
        }
    }

    #[test]
    fn terminal_heartbeat_error_and_disconnect_are_one_cutoff_event() {
        let stats = Arc::new(Stats::default());
        let connected = Connected::for_vessel(Arc::clone(&stats), Some(0));
        let before = capture_cutoff(&stats, Instant::now(), 1, 0);
        assert_eq!(before.report["connected"], 1);
        assert_eq!(before.report["failures"], 0);
        connected.end_session(true, Some(("heartbeat-close", "peer closed".into(), None)));
        let after = capture_cutoff(&stats, Instant::now(), 1, 0);
        assert_eq!(after.report["connected"], 0);
        assert_eq!(after.report["failures"], 1);
        assert_eq!(after.report["unresolved_disconnections"], 1);
    }

    #[test]
    fn concurrent_probe_replies_have_one_frozen_boundary() {
        let stats = Arc::new(Stats::default());
        let writer_stats = Arc::clone(&stats);
        let writer = std::thread::spawn(move || {
            for sequence in 0..1000_u64 {
                let mut pending = HashMap::from([(sequence, Instant::now())]);
                {
                    let _gate = writer_stats.cutoff_gate.lock().unwrap();
                    writer_stats.timing_sent.fetch_add(1, Ordering::Relaxed);
                }
                record_timing_pong(&writer_stats, &mut pending, &sequence.to_be_bytes());
            }
        });
        for _ in 0..100 {
            let frozen = capture_cutoff(&stats, Instant::now(), 1, 0).report;
            assert_eq!(
                frozen["timing_matched"],
                frozen["identified_probe_rtt"]["count"]
            );
            assert_eq!(
                frozen["timing_unanswered"].as_u64().unwrap(),
                frozen["timing_sent"].as_u64().unwrap()
                    - frozen["timing_matched"].as_u64().unwrap()
            );
        }
        writer.join().unwrap();
        assert_eq!(stats.timing_ms.lock().unwrap().len(), 1000);
    }

    #[test]
    fn zero_followup_excludes_later_registration_and_reply() {
        let stats = Arc::new(Stats::default());
        stats.diagnostic_enabled.store(true, Ordering::Relaxed);
        stats.presence.lock().unwrap().diagnostics = Some(DiagnosticBook::new(1, Instant::now()));
        let frozen = capture_cutoff(&stats, Instant::now(), 1, 0);
        let _connection = Connected::for_vessel(Arc::clone(&stats), Some(0));
        stats.record_reply(0);
        stats.stage(0, "late-stage");
        let followup = stats
            .presence
            .lock()
            .unwrap()
            .diagnostics
            .as_ref()
            .unwrap()
            .followup(frozen.at);
        assert_eq!(followup["registered"], 0);
        assert_eq!(followup["useful_reply_observed"], 0);
        assert_eq!(followup["vessels"][0]["connected"], false);
        assert_eq!(followup["vessels"][0]["stage"], "not-started");
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn cutoff_cohort_and_count_stay_frozen_when_recovery_and_new_failures_follow() {
        let stats = Arc::new(Stats::default());
        stats.diagnostic_enabled.store(true, Ordering::Relaxed);
        stats.presence.lock().unwrap().diagnostics = Some(DiagnosticBook::new(3, Instant::now()));
        let first = Connected::for_vessel(Arc::clone(&stats), Some(0));
        let _second = Connected::for_vessel(Arc::clone(&stats), Some(1));
        stats.stage(2, "tcp-connect");
        stats.error("heartbeat-close", 2, "inactivity".into());
        let cutoff = {
            let mut presence = stats.presence.lock().unwrap();
            let snapshot = presence
                .diagnostics
                .as_mut()
                .unwrap()
                .cutoff(Instant::now());
            assert_eq!(snapshot["connected"], presence.current);
            snapshot
        };
        assert_eq!(cutoff["missing_vessels"][0]["vessel_id"], "vessel-00002");
        assert_eq!(cutoff["missing_vessels"][0]["stage"], "tcp-connect");
        let _replacement = Connected::for_vessel(Arc::clone(&stats), Some(2));
        stats.record_reply(2);
        drop(first);
        let followup = stats
            .presence
            .lock()
            .unwrap()
            .diagnostics
            .as_ref()
            .unwrap()
            .followup(Instant::now());
        assert_eq!(followup["cohort_size"], 1);
        assert_eq!(followup["registered"], 1);
        assert_eq!(followup["useful_reply_observed"], 1);
        assert_eq!(followup["vessels"][0]["vessel_id"], "vessel-00002");
        assert_eq!(cutoff["connected"], 2);
        assert_eq!(cutoff["missing_count"], 1);
    }

    #[tokio::test]
    async fn cancellation_keeps_the_last_observed_network_stage_for_diagnosis() {
        let stats = Arc::new(Stats::default());
        stats.diagnostic_enabled.store(true, Ordering::Relaxed);
        stats.presence.lock().unwrap().diagnostics = Some(DiagnosticBook::new(1, Instant::now()));
        let observation = (Arc::clone(&stats), true, 0);
        assert!(
            tokio::time::timeout(
                Duration::from_millis(10),
                observed(
                    Some(&observation),
                    "http-upgrade",
                    std::future::pending::<Result<(), String>>()
                )
            )
            .await
            .is_err()
        );
        let cutoff = stats
            .presence
            .lock()
            .unwrap()
            .diagnostics
            .as_mut()
            .unwrap()
            .cutoff(Instant::now());
        assert_eq!(cutoff["missing_vessels"][0]["stage"], "http-upgrade");
        assert!(
            cutoff["missing_vessels"][0]["stage_seconds"]
                .as_f64()
                .unwrap()
                >= 0.01
        );
    }

    #[tokio::test]
    async fn adaptive_admission_bypasses_backlog_borrows_idle_slots_and_cleans_cancellation() {
        let mut admission = Admission::new(100, 20).unwrap();
        admission.borrow = true;
        let mut initial = Vec::new();
        for _ in 0..80 {
            initial.push(admission.acquire(false).await);
        }
        let mut backlog = JoinSet::new();
        for _ in 0..250 {
            let admission = admission.clone();
            backlog.spawn(async move { admission.acquire(false).await });
        }
        tokio::time::timeout(Duration::from_secs(1), async {
            while admission.metrics.queued_initial.load(Ordering::Relaxed) < 250 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let mut retries = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), async {
            for _ in 0..20 {
                retries.push(admission.acquire(true).await);
            }
        })
        .await
        .unwrap();
        assert_eq!(admission.metrics.active.load(Ordering::Relaxed), 100);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), admission.acquire(true))
                .await
                .is_err()
        );
        assert_eq!(admission.metrics.queued_retry.load(Ordering::Relaxed), 0);
        backlog.abort_all();
        while backlog.join_next().await.is_some() {}
        assert_eq!(admission.metrics.queued_initial.load(Ordering::Relaxed), 0);
        drop(initial);
        for _ in 0..80 {
            retries.push(admission.acquire(true).await);
        }
        assert_eq!(admission.metrics.active_retry.load(Ordering::Relaxed), 100);
        assert_eq!(admission.metrics.peak_active.load(Ordering::Relaxed), 100);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), admission.acquire(false))
                .await
                .is_err()
        );
        assert_eq!(admission.initial.available_permits(), 80);
        assert_eq!(admission.metrics.queued_initial.load(Ordering::Relaxed), 0);
        drop(retries);
        assert_eq!(admission.capacity.available_permits(), 100);
        assert_eq!(admission.metrics.active.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn response_watchdog_replaces_an_open_but_silent_connection() {
        let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for episode in 0..2 {
                let (stream, _) = listener.accept().await.unwrap();
                stream.set_nodelay(true).unwrap();
                let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
                let _connect = socket.next().await.unwrap().unwrap();
                socket
                    .send(Message::Text(
                        json!({"type":"et-connect-ack", "agent_id":"vessel-00000"})
                            .to_string()
                            .into(),
                    ))
                    .await
                    .unwrap();
                if episode == 0 {
                    tokio::time::sleep(Duration::from_secs(7)).await;
                } else {
                    while let Some(Ok(message)) = socket.next().await {
                        if let Message::Ping(payload) = message {
                            socket.send(Message::Pong(payload)).await.unwrap();
                        }
                    }
                }
            }
        });
        let stats = Arc::new(Stats::default());
        let mut retry = RetryPolicy::disabled(Ipv4Addr::LOCALHOST);
        retry.enabled = true;
        retry.response_timeout = watchdog_timeout(6, true).unwrap();
        retry.ips = vec![Ipv4Addr::LOCALHOST, Ipv4Addr::new(127, 0, 0, 2)];
        retry.base_ms = 5;
        retry.max_ms = 5;
        let task = tokio::spawn(agent(
            0,
            20000,
            address,
            Ipv4Addr::LOCALHOST,
            Arc::new(format!("ws://{address}/ws")),
            Admission::new(2, 1).unwrap(),
            Arc::clone(&stats),
            tokio::time::Instant::now() + Duration::from_secs(20),
            retry,
        ));
        tokio::time::timeout(Duration::from_secs(20), async {
            while stats.recovery_completed.load(Ordering::Relaxed) == 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "watchdog did not recover: {} errors={:?}",
                stats.sample(2.0, "test-timeout"),
                stats.errors.lock().unwrap()
            )
        });
        assert_eq!(stats.registered.load(Ordering::Relaxed), 1);
        assert_eq!(stats.reconnections.load(Ordering::Relaxed), 1);
        assert_eq!(
            stats
                .error_counts
                .lock()
                .unwrap()
                .get("heartbeat-response-timeout:other"),
            Some(&1)
        );
        task.abort();
        let _ = task.await;
        server.abort();
    }

    #[tokio::test]
    async fn reserved_retries_bypass_initial_backlog_without_raising_total_slots() {
        let admission = Admission::new(100, 20).unwrap();
        let _initial = admission.initial.acquire_many(80).await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(10), admission.gate(false).acquire())
                .await
                .is_err()
        );
        let _retry = tokio::time::timeout(
            Duration::from_millis(10),
            admission.gate(true).acquire_many(20),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(admission.retry.available_permits(), 0);
        assert!(Admission::new(100, 100).is_err());
        let shared = Admission::new(1, 0).unwrap();
        let _initial = shared.initial.acquire().await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(10), shared.gate(true).acquire())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn reconnect_requires_a_useful_response_and_preserves_unique_vessels() {
        let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let (unsolicited, observed) = tokio::sync::oneshot::channel();
        let mut unsolicited = Some(unsolicited);
        let server = tokio::spawn(async move {
            for episode in 0..2 {
                let (stream, _) = listener.accept().await.unwrap();
                let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
                let _connect = socket.next().await.unwrap().unwrap();
                socket
                    .send(Message::Text(
                        json!({"type":"et-connect-ack", "agent_id":"vessel-00000"})
                            .to_string()
                            .into(),
                    ))
                    .await
                    .unwrap();
                if episode == 0 {
                    socket.send(Message::Close(None)).await.unwrap();
                } else {
                    socket
                        .send(Message::Pong(Default::default()))
                        .await
                        .unwrap();
                    unsolicited.take().unwrap().send(()).unwrap();
                    loop {
                        if let Some(Ok(Message::Ping(payload))) = socket.next().await {
                            socket.send(Message::Pong(payload)).await.unwrap();
                            break;
                        }
                    }
                    let _closed = socket.next().await;
                }
            }
        });
        let stats = Arc::new(Stats::default());
        let mut retry = RetryPolicy::disabled(Ipv4Addr::LOCALHOST);
        retry.enabled = true;
        retry.base_ms = 10;
        retry.max_ms = 10;
        retry.attempt_seconds = 1;
        let task = tokio::spawn(agent(
            0,
            20000,
            address,
            Ipv4Addr::LOCALHOST,
            Arc::new(format!("ws://{address}/ws")),
            Admission::new(1, 0).unwrap(),
            Arc::clone(&stats),
            tokio::time::Instant::now() + Duration::from_secs(10),
            retry,
        ));
        observed.await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(stats.recovery_completed.load(Ordering::Relaxed), 0);
        assert_eq!(stats.pongs.load(Ordering::Relaxed), 0);
        tokio::time::timeout(Duration::from_secs(7), async {
            while stats.recovery_completed.load(Ordering::Relaxed) == 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(stats.registered.load(Ordering::Relaxed), 1);
        assert_eq!(stats.registration_successes.load(Ordering::Relaxed), 2);
        assert_eq!(stats.reconnections.load(Ordering::Relaxed), 1);
        assert_eq!(stats.useful_recovery_ms.lock().unwrap().len(), 1);
        task.abort();
        let _ = task.await;
        server.abort();
        assert_eq!(stats.connected.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn backoff_is_bounded_and_spreads_agents() {
        let mut retry = RetryPolicy::disabled(Ipv4Addr::LOCALHOST);
        retry.enabled = true;
        let mut values = std::collections::HashSet::new();
        for index in 0..100 {
            values.insert(retry.delay(0, index).as_millis());
            for failures in 0..32 {
                let value = retry.delay(failures, index);
                assert!(value >= Duration::from_millis(500));
                assert!(value <= Duration::from_millis(30000));
            }
        }
        assert!(values.len() > 50);
    }

    #[test]
    fn presence_integrates_connected_time_and_keeps_short_drops() {
        let stats = Arc::new(Stats::default());
        let socket = Connected::new(Arc::clone(&stats));
        {
            let mut meter = stats.presence.lock().unwrap();
            meter.active = true;
            meter.minimum = 1;
            meter.last = Instant::now() - Duration::from_secs(2);
        }
        drop(socket);
        let replacement = Connected::new(Arc::clone(&stats));
        let meter = stats.presence.lock().unwrap();
        assert!(meter.seconds >= 2.0);
        assert_eq!(meter.minimum, 0);
        assert_eq!(meter.current, 1);
        drop(meter);
        drop(replacement);
    }

    #[test]
    fn timing_matches_identifiers_instead_of_reply_order() {
        let stats = Stats::default();
        let mut pending = HashMap::from([(1_u64, Instant::now()), (2_u64, Instant::now())]);
        record_timing_pong(&stats, &mut pending, &2_u64.to_be_bytes());
        assert!(pending.contains_key(&1));
        assert!(!pending.contains_key(&2));
        record_timing_pong(&stats, &mut pending, &2_u64.to_be_bytes());
        record_timing_pong(&stats, &mut pending, b"");
        assert_eq!(stats.timing_matched.load(Ordering::Relaxed), 1);
        assert_eq!(stats.timing_ms.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn registration_budget_cancels_a_queued_attempt() {
        let stats = Arc::new(Stats::default());
        let admission = Admission::new(1, 0).unwrap();
        let _blocked = admission.initial.acquire().await.unwrap();
        agent(
            0,
            20000,
            "127.0.0.1:1".parse().unwrap(),
            Ipv4Addr::LOCALHOST,
            Arc::new("ws://127.0.0.1:1/ws".into()),
            admission.clone(),
            Arc::clone(&stats),
            tokio::time::Instant::now() + Duration::from_millis(10),
            RetryPolicy::disabled(Ipv4Addr::LOCALHOST),
        )
        .await;
        assert!(stats.registration_censored.load(Ordering::Relaxed));
        assert_eq!(stats.finished.load(Ordering::Relaxed), 1);
        assert_eq!(stats.cancelled_registrations.load(Ordering::Relaxed), 1);
        assert_eq!(stats.connected.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn register_shares_port_with_listener_on_another_local_address() {
        let dns = std::net::TcpListener::bind((Ipv4Addr::new(127, 0, 0, 11), 0)).unwrap();
        let source_port = dns.local_addr().unwrap().port();
        assert!(
            TcpSocket::new_v4()
                .unwrap()
                .bind((Ipv4Addr::UNSPECIFIED, source_port).into())
                .is_err()
        );
        let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
            let _registration = socket.next().await.unwrap().unwrap();
            socket
                .send(Message::Text(
                    json!({"type":"et-connect-ack", "agent_id":"test-vessel"})
                        .to_string()
                        .into(),
                ))
                .await
                .unwrap();
            let _closed = socket.next().await;
        });
        let url = format!("ws://{address}/ws");
        let client = tokio::time::timeout(
            Duration::from_secs(5),
            register(
                address,
                &url,
                Ipv4Addr::LOCALHOST,
                source_port,
                "test-vessel",
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            client.get_ref().local_addr().unwrap().ip(),
            IpAddr::V4(Ipv4Addr::LOCALHOST)
        );
        drop(client);
        server.abort();
    }
}
