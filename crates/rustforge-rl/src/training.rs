//! Small training-loop helpers shared by examples and integration tests.
//!
//! These helpers keep common RL control-flow decisions explicit at call sites.
//! In particular, Gymnasium-style environments separate real terminal states
//! (`terminated`) from time-limit cutoffs (`truncated`), and DQN should only
//! stop bootstrapping on real terminal states.

/// Returns whether the current episode should be reset.
///
/// Both true terminal states and time-limit truncations end the episode from
/// the environment loop's perspective.
#[inline]
pub fn episode_done(terminated: bool, truncated: bool) -> bool {
    terminated || truncated
}

/// Returns the `done` flag that should be stored in replay buffers for TD targets.
///
/// A time-limit truncation is not a true terminal state, so DQN should continue
/// bootstrapping from the final observation. Only `terminated` disables the
/// future-value term.
#[inline]
pub fn replay_done(terminated: bool, _truncated: bool) -> bool {
    terminated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn episode_done_ends_on_terminated_or_truncated() {
        assert!(episode_done(true, false));
        assert!(episode_done(false, true));
        assert!(episode_done(true, true));
        assert!(!episode_done(false, false));
    }

    #[test]
    fn replay_done_only_tracks_true_terminal_states() {
        assert!(replay_done(true, false));
        assert!(replay_done(true, true));
        assert!(!replay_done(false, true));
        assert!(!replay_done(false, false));
    }
}
