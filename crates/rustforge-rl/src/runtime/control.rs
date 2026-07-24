//! Request-ID based, runtime-neutral trainer control state.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ControlRequestId(u64);

impl ControlRequestId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlKind {
    Pause,
    Resume,
    GracefulStop,
    ForceStop,
    Checkpoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum StopMode {
    None,
    Graceful,
    Force,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlRejection {
    Stopping,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlApplyResult {
    Applied,
    Superseded,
    AlreadyInState,
    Unsupported,
    Rejected(ControlRejection),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlResolution {
    pub request_id: ControlRequestId,
    pub control: ControlKind,
    pub result: ControlApplyResult,
    pub applied_at_step: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlObservation {
    pub revision: u64,
    pub effective_paused: bool,
    pub stop_mode: StopMode,
    pub resolutions: Vec<ControlResolution>,
}

#[derive(Clone)]
pub struct TrainerControl {
    inner: Arc<ControlInner>,
}

struct ControlInner {
    next_request: AtomicU64,
    state: Mutex<ControlState>,
    changed: Condvar,
}

struct ControlState {
    revision: u64,
    pause_desired: bool,
    pause_applied: bool,
    pause_pending: Option<PendingPause>,
    stop_mode: StopMode,
    stop_pending: Option<PendingStop>,
    checkpoint_pending: Option<ControlRequestId>,
    queued: VecDeque<QueuedResolution>,
}

struct PendingPause {
    id: ControlRequestId,
    desired: bool,
    kind: ControlKind,
}

struct PendingStop {
    id: ControlRequestId,
    mode: StopMode,
    kind: ControlKind,
}

struct QueuedResolution {
    id: ControlRequestId,
    kind: ControlKind,
    result: ControlApplyResult,
}

impl TrainerControl {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ControlInner {
                next_request: AtomicU64::new(0),
                state: Mutex::new(ControlState {
                    revision: 0,
                    pause_desired: false,
                    pause_applied: false,
                    pause_pending: None,
                    stop_mode: StopMode::None,
                    stop_pending: None,
                    checkpoint_pending: None,
                    queued: VecDeque::new(),
                }),
                changed: Condvar::new(),
            }),
        }
    }

    pub fn request_pause(&self) -> ControlRequestId {
        self.request_pause_state(true)
    }

    pub fn request_resume(&self) -> ControlRequestId {
        self.request_pause_state(false)
    }

    pub fn request_graceful_stop(&self) -> ControlRequestId {
        self.request_stop(StopMode::Graceful, ControlKind::GracefulStop)
    }

    pub fn request_force_stop(&self) -> ControlRequestId {
        self.request_stop(StopMode::Force, ControlKind::ForceStop)
    }

    pub fn request_checkpoint(&self) -> ControlRequestId {
        let id = self.next_id();
        let mut state = lock_recover(&self.inner.state);
        if let Some(previous) = state.checkpoint_pending.replace(id) {
            state.queued.push_back(QueuedResolution {
                id: previous,
                kind: ControlKind::Checkpoint,
                result: ControlApplyResult::Superseded,
            });
        }
        state.revision += 1;
        self.inner.changed.notify_all();
        id
    }

    pub fn observe(&self, applied_at_step: u64, checkpoint_supported: bool) -> ControlObservation {
        let mut state = lock_recover(&self.inner.state);
        observe_locked(&mut state, applied_at_step, checkpoint_supported)
    }

    pub fn wait_while_paused(
        &self,
        applied_at_step: u64,
        checkpoint_supported: bool,
    ) -> ControlObservation {
        loop {
            let observation = self.observe(applied_at_step, checkpoint_supported);
            if !observation.effective_paused || observation.stop_mode != StopMode::None {
                return observation;
            }
            let mut state = lock_recover(&self.inner.state);
            while state.pause_desired && state.stop_mode == StopMode::None {
                state = self
                    .inner
                    .changed
                    .wait(state)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
        }
    }

    fn request_pause_state(&self, desired: bool) -> ControlRequestId {
        let id = self.next_id();
        let kind = if desired {
            ControlKind::Pause
        } else {
            ControlKind::Resume
        };
        let mut state = lock_recover(&self.inner.state);
        if state.stop_mode != StopMode::None {
            state.queued.push_back(QueuedResolution {
                id,
                kind,
                result: ControlApplyResult::Rejected(ControlRejection::Stopping),
            });
        } else {
            if let Some(previous) = state.pause_pending.take() {
                state.queued.push_back(QueuedResolution {
                    id: previous.id,
                    kind: previous.kind,
                    result: ControlApplyResult::Superseded,
                });
            }
            state.pause_desired = desired;
            state.pause_pending = Some(PendingPause { id, desired, kind });
        }
        state.revision += 1;
        self.inner.changed.notify_all();
        id
    }

    fn request_stop(&self, mode: StopMode, kind: ControlKind) -> ControlRequestId {
        let id = self.next_id();
        let mut state = lock_recover(&self.inner.state);
        if mode <= state.stop_mode {
            state.queued.push_back(QueuedResolution {
                id,
                kind,
                result: ControlApplyResult::AlreadyInState,
            });
        } else {
            if let Some(previous) = state.stop_pending.take() {
                state.queued.push_back(QueuedResolution {
                    id: previous.id,
                    kind: previous.kind,
                    result: ControlApplyResult::Superseded,
                });
            }
            if let Some(previous) = state.pause_pending.take() {
                state.queued.push_back(QueuedResolution {
                    id: previous.id,
                    kind: previous.kind,
                    result: ControlApplyResult::Superseded,
                });
            }
            state.stop_mode = mode;
            state.stop_pending = Some(PendingStop { id, mode, kind });
            state.pause_applied = false;
        }
        state.revision += 1;
        self.inner.changed.notify_all();
        id
    }

    fn next_id(&self) -> ControlRequestId {
        ControlRequestId(self.inner.next_request.fetch_add(1, Ordering::Relaxed) + 1)
    }
}

impl Default for TrainerControl {
    fn default() -> Self {
        Self::new()
    }
}

fn observe_locked(
    state: &mut ControlState,
    applied_at_step: u64,
    checkpoint_supported: bool,
) -> ControlObservation {
    let mut resolutions: Vec<ControlResolution> = state
        .queued
        .drain(..)
        .map(|queued| ControlResolution {
            request_id: queued.id,
            control: queued.kind,
            result: queued.result,
            applied_at_step,
        })
        .collect();

    if let Some(pending) = state.stop_pending.take() {
        debug_assert_eq!(pending.mode, state.stop_mode);
        resolutions.push(ControlResolution {
            request_id: pending.id,
            control: pending.kind,
            result: ControlApplyResult::Applied,
            applied_at_step,
        });
        state.pause_applied = false;
    }

    if let Some(pending) = state.pause_pending.take() {
        let result = if state.stop_mode != StopMode::None {
            ControlApplyResult::Rejected(ControlRejection::Stopping)
        } else if state.pause_applied == pending.desired {
            ControlApplyResult::AlreadyInState
        } else {
            state.pause_applied = pending.desired;
            ControlApplyResult::Applied
        };
        resolutions.push(ControlResolution {
            request_id: pending.id,
            control: pending.kind,
            result,
            applied_at_step,
        });
    }

    if let Some(id) = state.checkpoint_pending.take() {
        resolutions.push(ControlResolution {
            request_id: id,
            control: ControlKind::Checkpoint,
            result: if checkpoint_supported {
                ControlApplyResult::Applied
            } else {
                ControlApplyResult::Unsupported
            },
            applied_at_step,
        });
    }

    ControlObservation {
        revision: state.revision,
        effective_paused: state.pause_applied && state.stop_mode == StopMode::None,
        stop_mode: state.stop_mode,
        resolutions,
    }
}

fn lock_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
