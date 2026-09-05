pub(crate) fn derive_seed(base_seed: u64, stream_id: u64) -> u64 {
    let mut value =
        base_seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(stream_id.wrapping_add(1)));
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EpisodeBoundary {
    None,
    Terminated,
    Truncated,
    StepLimit,
}

impl EpisodeBoundary {
    pub(crate) fn classify(terminated: bool, truncated: bool, step_limit: bool) -> Self {
        if terminated {
            Self::Terminated
        } else if truncated {
            Self::Truncated
        } else if step_limit {
            Self::StepLimit
        } else {
            Self::None
        }
    }

    pub(crate) fn bootstrap_value(self, estimate: impl FnOnce() -> f32) -> Option<f32> {
        match self {
            Self::None => None,
            Self::Terminated => Some(0.0),
            Self::Truncated | Self::StepLimit => Some(estimate()),
        }
    }

    pub(crate) fn done_mask(self) -> f32 {
        match self {
            Self::Terminated => 1.0,
            Self::None | Self::Truncated | Self::StepLimit => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::{derive_seed, EpisodeBoundary};

    #[test]
    fn seed_streams_are_stable_and_distinct() {
        let first = [
            derive_seed(2026, 0),
            derive_seed(2026, 1),
            derive_seed(2026, 2),
        ];
        let repeated = [
            derive_seed(2026, 0),
            derive_seed(2026, 1),
            derive_seed(2026, 2),
        ];

        assert_eq!(first, repeated);
        assert_ne!(first[0], first[1]);
        assert_ne!(first[1], first[2]);
        assert_ne!(first[0], first[2]);
    }

    #[test]
    fn termination_has_precedence_and_skips_bootstrap() {
        let boundary = EpisodeBoundary::classify(true, true, true);
        let calls = Cell::new(0);

        assert_eq!(boundary, EpisodeBoundary::Terminated);
        assert_eq!(boundary.done_mask(), 1.0);
        assert_eq!(
            boundary.bootstrap_value(|| {
                calls.set(calls.get() + 1);
                7.0
            }),
            Some(0.0)
        );
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn truncation_precedes_step_limit_and_retains_bootstrap() {
        let boundary = EpisodeBoundary::classify(false, true, true);
        assert_eq!(boundary, EpisodeBoundary::Truncated);
        assert_eq!(boundary.done_mask(), 0.0);
        assert_eq!(boundary.bootstrap_value(|| 3.5), Some(3.5));
    }

    #[test]
    fn step_limit_retains_bootstrap_and_none_does_not_end_episode() {
        let boundary = EpisodeBoundary::classify(false, false, true);
        assert_eq!(boundary, EpisodeBoundary::StepLimit);
        assert_eq!(boundary.done_mask(), 0.0);
        assert_eq!(boundary.bootstrap_value(|| -2.25), Some(-2.25));

        assert_eq!(EpisodeBoundary::None.done_mask(), 0.0);
        assert_eq!(EpisodeBoundary::None.bootstrap_value(|| 9.0), None);
    }
}
