//! Deterministic two-state chain used by the GPU example and convergence test.
//! State 0 transitions to state 1 (reward 0 for action 0, -1 for action 1).
//! State 1 terminates (reward -1 for action 0, +1 for action 1).
//! With gamma=0.9 the optimal Q-values are [0.9,-0.1] and [-1,1].
use rustforge_rl::{
    buffer::TransitionBatch,
    env::{Environment, Space},
    training::replay_done,
};
#[derive(Default)]
pub struct TinyChain {
    state: usize,
}
impl TinyChain {
    fn observation(&self) -> [f32; 2] {
        if self.state == 0 {
            [1., 0.]
        } else {
            [0., 1.]
        }
    }
    /// Enumerates the four state/action transitions through the Environment API.
    /// Full replay avoids sampling noise; reset seeds choose reproducible starts.
    pub fn full_replay(&mut self) -> TransitionBatch {
        let mut batch = TransitionBatch::new(4, 2);
        for state in 0..2 {
            for action in 0..2 {
                let (observation, _) = self.reset(Some(state));
                let (next, reward, terminated, truncated, _) = self.step(action);
                let row = state as usize * 2 + action;
                for col in 0..2 {
                    batch.states.data_mut()[[row, col]] = observation[col];
                    batch.next_states.data_mut()[[row, col]] = next[col];
                }
                batch.actions[row] = action;
                batch.rewards.data_mut()[[row, 0]] = reward;
                batch.dones.data_mut()[[row, 0]] = if replay_done(terminated, truncated) {
                    1.
                } else {
                    0.
                };
            }
        }
        batch.size = 4;
        batch
    }
}
impl Environment for TinyChain {
    type Obs = [f32; 2];
    type Act = usize;
    type Info = ();
    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, ()) {
        self.state = seed.map_or(0, |seed| (seed % 2) as usize);
        (self.observation(), ())
    }
    fn step(&mut self, action: usize) -> (Self::Obs, f32, bool, bool, ()) {
        assert!(action < 2);
        if self.state == 0 {
            self.state = 1;
            (
                self.observation(),
                if action == 0 { 0. } else { -1. },
                false,
                false,
                (),
            )
        } else {
            (
                [0., 0.],
                if action == 1 { 1. } else { -1. },
                true,
                false,
                (),
            )
        }
    }
    fn action_space(&self) -> Space {
        Space::Discrete(2)
    }
    fn observation_space(&self) -> Space {
        Space::continuous(vec![0.; 2], vec![1.; 2])
    }
}
