//! Returns and advantage estimation for policy gradient algorithms.
//!
//! Provides pure-function utilities for computing discounted returns (Monte Carlo)
//! and Generalized Advantage Estimation (GAE). These are shared by REINFORCE and A2C.
//!
//! ## `dones` Semantics
//!
//! The `dones` parameter uses `f32` encoding (`1.0` = done, `0.0` = not done) and
//! represents **true terminal states only** (`terminated`), NOT time-limit truncations.
//! This matches the `replay_done(terminated, truncated)` convention from `training.rs`.
//!
//! Time-limit truncation should **not** set `done=1.0` — the bootstrap term
//! `γ·V(s_{t+1})` must remain active so GAE does not underestimate truncated tails.
//! The environment loop resets separately based on `episode_done(terminated, truncated)`.
//!
//! ## Usage with REINFORCE vs A2C
//!
//! - **REINFORCE** (no critic): pass `values` as all zeros, `lambda=1.0`,
//!   `last_value=0.0`. Result: `advantages == returns` = MC discounted returns.
//!   Baseline subtraction is handled at the training stage, not here.
//! - **A2C** (with critic): pass critic's detached V(s_t) estimates as `values`.
//!   Result: `advantages` = GAE, `returns` = `advantages + values` (TD(λ) targets).

/// Computes discounted returns G_t for each timestep, scanning backwards.
///
/// ## Formula
///
/// ```text
/// G_T = r_T                                          (if done_T = 1.0)
/// G_T = r_T + γ * 0                                  (last step, no future)
/// G_t = r_t + γ * (1 - done_t) * G_{t+1}            (general case)
/// ```
///
/// At `done_t = 1.0`, the future return is zeroed (true terminal state).
///
/// ## Arguments
///
/// - `rewards`: Per-step rewards, length `T`.
/// - `dones`: Per-step terminal flags (f32: 1.0 = terminated, 0.0 = not), length `T`.
/// - `gamma`: Discount factor γ ∈ [0, 1].
///
/// ## Returns
///
/// `Vec<f32>` of length `T` with discounted returns for each timestep.
///
/// ## Panics
///
/// - If `rewards.len() != dones.len()`.
/// - If `gamma` is outside [0, 1] (debug-only).
pub fn compute_discounted_returns(rewards: &[f32], dones: &[f32], gamma: f32) -> Vec<f32> {
    debug_assert!(
        (0.0..=1.0).contains(&gamma),
        "gamma must be in [0, 1], got {}",
        gamma
    );
    assert_eq!(
        rewards.len(),
        dones.len(),
        "rewards and dones must have the same length: {} vs {}",
        rewards.len(),
        dones.len()
    );

    let t = rewards.len();
    let mut returns = vec![0.0_f32; t];

    if t == 0 {
        return returns;
    }

    // Backward scan: G_t = r_t + γ * (1 - done_t) * G_{t+1}
    returns[t - 1] = rewards[t - 1];
    for i in (0..t - 1).rev() {
        returns[i] = rewards[i] + gamma * (1.0 - dones[i]) * returns[i + 1];
    }

    returns
}

/// Computes Generalized Advantage Estimation (GAE) and returns.
///
/// ## Formula
///
/// ```text
/// δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
/// A_t^GAE(λ) = Σ_{l=0}^{T-t-1} (γλ)^l * δ_{t+l}
/// returns_t = A_t + V(s_t)
/// ```
///
/// ## Special cases
///
/// - `λ = 0`: `A_t = δ_t` (one-step TD error)
/// - `λ = 1, γ = 1`: `returns = MC returns` (Monte Carlo)
///
/// ## Arguments
///
/// - `rewards`: Per-step rewards, length `T`.
/// - `values`: Per-step value estimates V(s_t) from critic, length `T`.
///   For REINFORCE (no critic), pass all zeros.
/// - `dones`: Per-step terminal flags (f32: 1.0 = terminated only), length `T`.
/// - `last_value`: V(s_{T+1}) — the value of the state after the last step.
///   Use 0.0 if the episode terminated. Use critic's estimate if truncated.
/// - `gamma`: Discount factor γ ∈ [0, 1].
/// - `lambda`: GAE smoothing parameter λ ∈ [0, 1].
///
/// ## Returns
///
/// `(advantages, returns)` where:
/// - `advantages[t] = A_t^GAE` (for policy loss weighting)
/// - `returns[t] = advantages[t] + values[t]` (for value function regression target)
///
/// ## Panics
///
/// - If `rewards`, `values`, `dones` have different lengths.
/// - If `gamma` or `lambda` is outside [0, 1] (debug-only).
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    last_value: f32,
    gamma: f32,
    lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert!(
        (0.0..=1.0).contains(&gamma),
        "gamma must be in [0, 1], got {}",
        gamma
    );
    debug_assert!(
        (0.0..=1.0).contains(&lambda),
        "lambda must be in [0, 1], got {}",
        lambda
    );
    assert_eq!(
        rewards.len(),
        values.len(),
        "rewards and values must have the same length: {} vs {}",
        rewards.len(),
        values.len()
    );
    assert_eq!(
        rewards.len(),
        dones.len(),
        "rewards and dones must have the same length: {} vs {}",
        rewards.len(),
        dones.len()
    );

    let t = rewards.len();
    let mut advantages = vec![0.0_f32; t];
    let mut returns = vec![0.0_f32; t];

    if t == 0 {
        return (advantages, returns);
    }

    // Backward scan: A_t = δ_t + γλ * (1 - done_t) * A_{t+1}
    let mut gae = 0.0_f32;
    for i in (0..t).rev() {
        let next_value = if i + 1 < t { values[i + 1] } else { last_value };
        let not_done = 1.0 - dones[i];

        // δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        let delta = rewards[i] + gamma * next_value * not_done - values[i];

        // A_t = δ_t + γλ * (1 - done_t) * A_{t+1}
        gae = delta + gamma * lambda * not_done * gae;

        advantages[i] = gae;
        returns[i] = gae + values[i];
    }

    (advantages, returns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // ── compute_discounted_returns ──

    #[test]
    fn discounted_returns_single_step_terminal() {
        // One step, terminal: G_0 = r_0 = 5.0
        let returns = compute_discounted_returns(&[5.0], &[1.0], 0.99);
        assert_abs_diff_eq!(returns[0], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn discounted_returns_single_step_non_terminal() {
        // One step, not terminal: G_0 = r_0 = 5.0 (no future regardless)
        let returns = compute_discounted_returns(&[5.0], &[0.0], 0.99);
        assert_abs_diff_eq!(returns[0], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn discounted_returns_multi_step_no_terminal() {
        // rewards = [1, 1, 1], dones = [0, 0, 0], gamma = 0.5
        // G_2 = 1
        // G_1 = 1 + 0.5 * 1 = 1.5
        // G_0 = 1 + 0.5 * 1.5 = 1.75
        let returns = compute_discounted_returns(&[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0], 0.5);
        assert_abs_diff_eq!(returns[0], 1.75, epsilon = 1e-6);
        assert_abs_diff_eq!(returns[1], 1.5, epsilon = 1e-6);
        assert_abs_diff_eq!(returns[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn discounted_returns_terminal_in_middle() {
        // rewards = [1, 1, 1, 1], dones = [0, 1, 0, 0], gamma = 1.0
        // G_3 = 1
        // G_2 = 1 + 1*1 = 2
        // G_1 = 1 (done=1 zeroes future)
        // G_0 = 1 + 1*(1-0)*1 = 1 + 1 = 2
        let returns =
            compute_discounted_returns(&[1.0, 1.0, 1.0, 1.0], &[0.0, 1.0, 0.0, 0.0], 1.0);
        assert_abs_diff_eq!(returns[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(returns[1], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(returns[2], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(returns[3], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn discounted_returns_empty() {
        let returns = compute_discounted_returns(&[], &[], 0.99);
        assert!(returns.is_empty());
    }

    #[test]
    #[should_panic(expected = "rewards and dones must have the same length")]
    fn discounted_returns_length_mismatch_panics() {
        compute_discounted_returns(&[1.0, 2.0], &[0.0], 0.99);
    }

    // ── compute_gae ──

    #[test]
    fn gae_lambda_zero_equals_td_error() {
        // λ=0 → A_t = δ_t = r_t + γ*V(s_{t+1})*(1-done) - V(s_t)
        let rewards = [2.0, 3.0];
        let values = [1.0, 2.0];
        let dones = [0.0, 0.0];
        let last_value = 4.0;
        let gamma = 0.9;

        let (advantages, returns) = compute_gae(&rewards, &values, &dones, last_value, gamma, 0.0);

        // δ_0 = 2 + 0.9*2*1 - 1 = 2 + 1.8 - 1 = 2.8
        assert_abs_diff_eq!(advantages[0], 2.8, epsilon = 1e-6);
        // δ_1 = 3 + 0.9*4*1 - 2 = 3 + 3.6 - 2 = 4.6
        assert_abs_diff_eq!(advantages[1], 4.6, epsilon = 1e-6);

        // returns = advantages + values
        assert_abs_diff_eq!(returns[0], 2.8 + 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(returns[1], 4.6 + 2.0, epsilon = 1e-6);
    }

    #[test]
    fn gae_lambda_one_gamma_one_returns_equal_mc_returns() {
        // λ=1, γ=1 → returns[t] = advantages[t] + values[t] should equal MC returns
        // MC returns with γ=1: G_t = Σ r_k from t to T
        let rewards = [1.0, 2.0, 3.0];
        let values = [0.5, 1.5, 2.5]; // arbitrary critic estimates
        let dones = [0.0, 0.0, 0.0];
        let last_value = 0.0; // pretend episode ends after step 2

        let (advantages, returns) =
            compute_gae(&rewards, &values, &dones, last_value, 1.0, 1.0);

        // MC returns: G_2=3, G_1=2+3=5, G_0=1+2+3=6
        assert_abs_diff_eq!(returns[0], 6.0, epsilon = 1e-5);
        assert_abs_diff_eq!(returns[1], 5.0, epsilon = 1e-5);
        assert_abs_diff_eq!(returns[2], 3.0, epsilon = 1e-5);

        // advantages = returns - values
        assert_abs_diff_eq!(advantages[0], 6.0 - 0.5, epsilon = 1e-5);
        assert_abs_diff_eq!(advantages[1], 5.0 - 1.5, epsilon = 1e-5);
        assert_abs_diff_eq!(advantages[2], 3.0 - 2.5, epsilon = 1e-5);
    }

    #[test]
    fn gae_terminal_resets_bootstrap() {
        // done=1 at step 1 → δ_1 does not bootstrap from step 2's value
        let rewards = [1.0, 2.0, 3.0];
        let values = [0.0, 0.0, 0.0];
        let dones = [0.0, 1.0, 0.0];
        let last_value = 10.0;
        let gamma = 0.99;
        let lambda = 0.95;

        let (advantages, _returns) =
            compute_gae(&rewards, &values, &dones, last_value, gamma, lambda);

        // δ_2 = 3 + 0.99*10*1 - 0 = 12.9
        // A_2 = 12.9
        assert_abs_diff_eq!(advantages[2], 12.9, epsilon = 1e-5);

        // δ_1 = 2 + 0.99*0*(1-1) - 0 = 2 (done zeroes bootstrap)
        // A_1 = 2 + 0.99*0.95*0*A_2 = 2 (done zeroes GAE carry)
        assert_abs_diff_eq!(advantages[1], 2.0, epsilon = 1e-5);

        // δ_0 = 1 + 0.99*0*1 - 0 = 1
        // A_0 = 1 + 0.99*0.95*1*2 = 1 + 1.881 = 2.881
        assert_abs_diff_eq!(advantages[0], 1.0 + 0.99 * 0.95 * 2.0, epsilon = 1e-5);
    }

    #[test]
    fn gae_last_value_bootstrap() {
        // Non-terminal episode end: last_value should bootstrap
        let rewards = [1.0];
        let values = [0.0];
        let dones = [0.0]; // NOT terminal
        let last_value = 5.0;
        let gamma = 0.9;
        let lambda = 1.0;

        let (advantages, returns) =
            compute_gae(&rewards, &values, &dones, last_value, gamma, lambda);

        // δ_0 = 1 + 0.9*5 - 0 = 5.5
        assert_abs_diff_eq!(advantages[0], 5.5, epsilon = 1e-6);
        assert_abs_diff_eq!(returns[0], 5.5, epsilon = 1e-6); // + values[0]=0
    }

    #[test]
    fn gae_last_value_ignored_when_terminal() {
        // Terminal at last step: last_value should NOT be used
        let rewards = [1.0];
        let values = [0.0];
        let dones = [1.0]; // TERMINAL
        let last_value = 100.0; // should be ignored

        let (advantages, _returns) =
            compute_gae(&rewards, &values, &dones, last_value, 0.99, 1.0);

        // δ_0 = 1 + 0.99*100*(1-1) - 0 = 1 (last_value zeroed by done)
        // Wait: done is at step 0 — it means the transition at step 0 ended the episode.
        // next_value = last_value (since i+1 >= t), but (1-done)=0, so:
        // δ_0 = 1 + 0.99*100*0 - 0 = 1
        assert_abs_diff_eq!(advantages[0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn gae_multi_episode_in_single_batch() {
        // Two episodes in one batch: [ep1: r=1,1 done=0,1] [ep2: r=2,2 done=0,0]
        let rewards = [1.0, 1.0, 2.0, 2.0];
        let values = [0.0, 0.0, 0.0, 0.0];
        let dones = [0.0, 1.0, 0.0, 0.0];
        let last_value = 0.0;
        let gamma = 1.0;
        let lambda = 1.0;

        let (_advantages, returns) =
            compute_gae(&rewards, &values, &dones, last_value, gamma, lambda);

        // Episode 2: G_3=2, G_2=2+2=4
        assert_abs_diff_eq!(returns[2], 4.0, epsilon = 1e-5);
        assert_abs_diff_eq!(returns[3], 2.0, epsilon = 1e-5);

        // Episode 1: done at step 1 → G_1=1, G_0=1+1=2
        assert_abs_diff_eq!(returns[0], 2.0, epsilon = 1e-5);
        assert_abs_diff_eq!(returns[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn gae_reinforce_pattern_values_zero() {
        // REINFORCE: values=0, λ=1 → advantages == returns == MC returns
        let rewards = [1.0, 2.0, 3.0];
        let values = [0.0, 0.0, 0.0];
        let dones = [0.0, 0.0, 0.0];
        let gamma = 0.9;

        let (advantages, returns) = compute_gae(&rewards, &values, &dones, 0.0, gamma, 1.0);

        // MC returns: G_2=3, G_1=2+0.9*3=4.7, G_0=1+0.9*4.7=5.23
        let expected_g2 = 3.0;
        let expected_g1 = 2.0 + 0.9 * expected_g2;
        let expected_g0 = 1.0 + 0.9 * expected_g1;

        // With values=0: advantages == returns
        assert_abs_diff_eq!(advantages[0], expected_g0, epsilon = 1e-5);
        assert_abs_diff_eq!(advantages[1], expected_g1, epsilon = 1e-5);
        assert_abs_diff_eq!(advantages[2], expected_g2, epsilon = 1e-5);

        assert_abs_diff_eq!(returns[0], advantages[0], epsilon = 1e-5);
        assert_abs_diff_eq!(returns[1], advantages[1], epsilon = 1e-5);
        assert_abs_diff_eq!(returns[2], advantages[2], epsilon = 1e-5);
    }

    #[test]
    fn gae_empty() {
        let (adv, ret) = compute_gae(&[], &[], &[], 0.0, 0.99, 0.95);
        assert!(adv.is_empty());
        assert!(ret.is_empty());
    }

    #[test]
    #[should_panic(expected = "rewards and values must have the same length")]
    fn gae_length_mismatch_values_panics() {
        compute_gae(&[1.0, 2.0], &[1.0], &[0.0, 0.0], 0.0, 0.99, 0.95);
    }

    #[test]
    #[should_panic(expected = "rewards and dones must have the same length")]
    fn gae_length_mismatch_dones_panics() {
        compute_gae(&[1.0, 2.0], &[1.0, 2.0], &[0.0], 0.0, 0.99, 0.95);
    }
}
