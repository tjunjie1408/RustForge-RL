//! Independent terminal input, source, progress, render, and outcome cadences.

use std::io;
use std::time::Duration;

use crossterm::event::{Event, EventStream};
use futures_util::StreamExt;
use tokio::time::{interval_at, Instant, Interval, MissedTickBehavior};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EventCadence {
    pub source_poll: Duration,
    pub progress_sample: Duration,
    pub render: Duration,
    pub outcome_poll: Duration,
}

impl Default for EventCadence {
    fn default() -> Self {
        Self {
            source_poll: Duration::from_millis(250),
            progress_sample: Duration::from_millis(250),
            render: Duration::from_millis(67),
            outcome_poll: Duration::from_secs(1),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AppEvent {
    Terminal(Event),
    SourcePoll,
    ProgressSample,
    Render,
    OutcomePoll,
}

pub struct TerminalEventLoop {
    input: EventStream,
    source_poll: Interval,
    progress_sample: Interval,
    render: Interval,
    outcome_poll: Interval,
}

impl TerminalEventLoop {
    pub fn new(cadence: EventCadence) -> Self {
        Self {
            input: EventStream::new(),
            source_poll: skipping_interval(cadence.source_poll),
            progress_sample: skipping_interval(cadence.progress_sample),
            render: skipping_interval(cadence.render),
            outcome_poll: skipping_interval(cadence.outcome_poll),
        }
    }

    pub async fn next(&mut self) -> io::Result<AppEvent> {
        tokio::select! {
            input = self.input.next() => match input {
                Some(Ok(event)) => Ok(AppEvent::Terminal(event)),
                Some(Err(error)) => Err(error),
                None => Err(io::Error::new(io::ErrorKind::UnexpectedEof, "terminal event stream closed")),
            },
            _ = self.source_poll.tick() => Ok(AppEvent::SourcePoll),
            _ = self.progress_sample.tick() => Ok(AppEvent::ProgressSample),
            _ = self.render.tick() => Ok(AppEvent::Render),
            _ = self.outcome_poll.tick() => Ok(AppEvent::OutcomePoll),
        }
    }
}

fn skipping_interval(period: Duration) -> Interval {
    let mut interval = interval_at(Instant::now() + period, period);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
    interval
}
