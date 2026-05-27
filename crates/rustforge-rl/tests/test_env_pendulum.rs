use approx::assert_relative_eq;
use rustforge_rl::env::pendulum::{angle_normalize, compute_reward};
use rustforge_rl::env::{Environment, Pendulum, PendulumAction};

#[test]
fn pendulum_reset_returns_valid_obs() {
    let mut env = Pendulum::new();
    let (obs, _) = env.reset(Some(42));

    // [cos(θ), sin(θ), θ̇]
    assert!(
        obs[0].abs() <= 1.0 + 1e-5,
        "cos(θ) out of range: {}",
        obs[0]
    );
    assert!(
        obs[1].abs() <= 1.0 + 1e-5,
        "sin(θ) out of range: {}",
        obs[1]
    );
    assert!(
        obs[2].abs() <= 1.0 + 1e-5,
        "initial θ̇ out of range: {}",
        obs[2]
    );
}

#[test]
fn pendulum_obs_unit_circle_invariant() {
    let mut env = Pendulum::new();

    // Test for multiple resets and steps
    for seed in 0..10 {
        let (obs, _) = env.reset(Some(seed));
        let norm_sq = obs[0] * obs[0] + obs[1] * obs[1];
        assert_relative_eq!(norm_sq, 1.0, epsilon = 1e-4);

        for _ in 0..50 {
            let (obs, _, _, _, _) = env.step(PendulumAction(0.5));
            let norm_sq = obs[0] * obs[0] + obs[1] * obs[1];
            assert_relative_eq!(norm_sq, 1.0, epsilon = 1e-4);
        }
    }
}

#[test]
fn pendulum_theta_dot_bounded() {
    let mut env = Pendulum::new();
    env.reset(Some(123));

    // Apply large torque to push speed limits
    for _ in 0..100 {
        let (obs, _, _, _, _) = env.step(PendulumAction(10.0));
        assert!(obs[2].abs() <= 8.0 + 1e-5, "θ̇ exceeded limit: {}", obs[2]);
    }

    for _ in 0..100 {
        let (obs, _, _, _, _) = env.step(PendulumAction(-10.0));
        assert!(obs[2].abs() <= 8.0 + 1e-5, "θ̇ exceeded limit: {}", obs[2]);
    }
}

#[test]
fn pendulum_action_clamped() {
    let mut env = Pendulum::new();
    env.reset(Some(0));

    // Check extreme positive torque
    let (obs, reward, _, _, _) = env.step(PendulumAction(100.0));
    assert!(obs[0].is_finite());
    assert!(obs[1].is_finite());
    assert!(obs[2].is_finite());
    assert_relative_eq!(obs[0] * obs[0] + obs[1] * obs[1], 1.0, epsilon = 1e-4);
    assert!(obs[2].abs() <= 8.0);
    assert!(reward.is_finite());

    // Check extreme negative torque
    let (obs, reward, _, _, _) = env.step(PendulumAction(-100.0));
    assert!(obs[0].is_finite());
    assert!(obs[1].is_finite());
    assert!(obs[2].is_finite());
    assert_relative_eq!(obs[0] * obs[0] + obs[1] * obs[1], 1.0, epsilon = 1e-4);
    assert!(obs[2].abs() <= 8.0);
    assert!(reward.is_finite());

    // Check NaN action
    let (obs, reward, _, _, _) = env.step(PendulumAction(f32::NAN));
    assert!(obs[0].is_finite());
    assert!(obs[1].is_finite());
    assert!(obs[2].is_finite());
    assert_relative_eq!(obs[0] * obs[0] + obs[1] * obs[1], 1.0, epsilon = 1e-4);
    assert!(obs[2].abs() <= 8.0);
    assert!(reward.is_finite());

    // Check Infinity action
    let (obs, reward, _, _, _) = env.step(PendulumAction(f32::INFINITY));
    assert!(obs[0].is_finite());
    assert!(obs[1].is_finite());
    assert!(obs[2].is_finite());
    assert_relative_eq!(obs[0] * obs[0] + obs[1] * obs[1], 1.0, epsilon = 1e-4);
    assert!(obs[2].abs() <= 8.0);
    assert!(reward.is_finite());
}

#[test]
fn pendulum_deterministic_with_same_seed() {
    let mut env1 = Pendulum::new();
    let mut env2 = Pendulum::new();

    let (obs1, _) = env1.reset(Some(999));
    let (obs2, _) = env2.reset(Some(999));
    assert_eq!(obs1, obs2, "Same seed must give same obs");

    for _ in 0..50 {
        let (next1, r1, t1, tr1, _) = env1.step(PendulumAction(1.5));
        let (next2, r2, t2, tr2, _) = env2.step(PendulumAction(1.5));
        assert_eq!(next1, next2);
        assert_relative_eq!(r1, r2, epsilon = 1e-6);
        assert_eq!(t1, t2);
        assert_eq!(tr1, tr2);
    }
}

#[test]
fn pendulum_step_truncates_at_max_steps() {
    let mut env = Pendulum::with_max_steps(5);
    env.reset(Some(0));
    for i in 0..4 {
        let (_, _, _, truncated, _) = env.step(PendulumAction(0.0));
        assert!(!truncated, "step {} should not truncate yet", i + 1);
    }
    let (_, _, _, truncated, _) = env.step(PendulumAction(0.0));
    assert!(truncated, "step 5 should truncate");
}

#[test]
fn pendulum_reward_is_negative() {
    let mut env = Pendulum::new();
    env.reset(Some(1));
    for _ in 0..20 {
        let (_, reward, _, _, _) = env.step(PendulumAction(0.5));
        assert!(reward <= 0.0 + 1e-5, "Reward should be ≤ 0, got {}", reward);
    }

    // Check theta=0, theta_dot=0, u=0 is 0.0
    let r_zero = compute_reward(0.0, 0.0, 0.0);
    assert_relative_eq!(r_zero, 0.0, epsilon = 1e-9);
}

#[test]
fn pendulum_reward_lower_bound() {
    // theta=pi, theta_dot=8, u=2
    // reward = -(pi^2 + 0.1 * 64 + 0.001 * 4) = -(9.8696044 + 6.4 + 0.004) = -16.2736044
    let r = compute_reward(std::f32::consts::PI, 8.0, 2.0);
    assert_relative_eq!(r, -16.273_605, epsilon = 1e-6);
}

#[test]
fn pendulum_unstable_equilibrium_at_theta_zero() {
    let mut env = Pendulum::new();
    env.reset(Some(42));

    // Set near unstable equilibrium (top, theta = 0)
    env.set_state(0.1, 0.0);

    // Let it fall under zero torque
    for _ in 0..25 {
        env.step(PendulumAction(0.0));
    }

    // The pendulum should swing away from the top
    let state = env.get_state();
    let theta = state[0];
    assert!(
        theta.abs() > 0.5,
        "Pendulum did not swing away from unstable top, theta: {}",
        theta
    );
}

#[test]
fn pendulum_stable_equilibrium_at_theta_pi() {
    let mut env = Pendulum::new();
    env.reset(Some(42));

    // Set near stable equilibrium (bottom, theta = pi)
    let pi = std::f32::consts::PI;
    env.set_state(pi, 0.0);

    // Let it stay under zero torque
    for _ in 0..200 {
        env.step(PendulumAction(0.0));
    }

    // The pendulum should stay close to the bottom (stable equilibrium)
    let state = env.get_state();
    let theta = state[0];
    let diff = (angle_normalize(theta).abs() - pi).abs();
    assert!(
        diff < 0.05,
        "Pendulum deviated from stable bottom, theta: {}, normalized: {}, diff: {}",
        theta,
        angle_normalize(theta),
        diff
    );
}

#[test]
fn pendulum_gymnasium_numerical_alignment() {
    let mut env = Pendulum::new();
    env.reset(Some(42));
    env.set_state(1.0, 0.5);

    let actions = [0.5, -0.5, 1.0, -1.0, 0.0];
    let expected = [
        (1.060_305_1, 1.206_103_2, -1.025_25),
        (1.149_579_3, 1.785_481_7, -1.269_965_5),
        (1.280_575_5, 2.619_925_7, -1.641_326_9),
        (1.440_003_6, 3.188_561_2, -2.327_274_8),
        (1.636_611_3, 3.932_155_4, -3.090_302_7),
    ];

    for (i, &u) in actions.iter().enumerate() {
        let (_, reward, _, _, _) = env.step(PendulumAction(u));
        let state = env.get_state();
        let (exp_theta, exp_theta_dot, exp_reward) = expected[i];

        assert_relative_eq!(state[0], exp_theta, epsilon = 1e-5);
        assert_relative_eq!(state[1], exp_theta_dot, epsilon = 1e-5);
        assert_relative_eq!(reward, exp_reward, epsilon = 1e-4);
    }
}
