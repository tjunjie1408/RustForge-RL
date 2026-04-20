//! Exhaustive tests for GridWorld environment.
//!
//! Covers: wall collision, goal reaching, boundary clamping, determinism,
//! and action enum exhaustiveness.

use rustforge_rl::env::{Environment, GridAction, GridWorld};

// Movement & Collision

mod movement {
    use super::*;

    #[test]
    fn wall_collision_no_move() {
        // Default 5×5 grid has walls at [1,1], [1,3], [2,1], [3,3]
        let mut env = GridWorld::new();
        env.reset(Some(42));

        // Agent starts at [0,0]. Move down to [1,0], then try to move right into wall [1,1]
        env.step(GridAction::Down);
        let (obs, _, _, _, _) = env.step(GridAction::Right);

        // Agent should still be at [1,0] because [1,1] is a wall
        assert_eq!(obs, [1, 0], "Agent should not move into wall at [1,1]");
    }

    #[test]
    fn goal_reached_terminates() {
        let mut env = GridWorld::new();
        env.reset(Some(42));

        // Navigate to goal at [4,4] via a wall-free path
        // Path: (0,0) → down×4 → right×4
        for _ in 0..4 {
            let (_, _, terminated, _, _) = env.step(GridAction::Down);
            assert!(!terminated, "Should not terminate before reaching goal");
        }
        // Now at [4,0], move right 4 times to [4,4]
        for i in 0..4 {
            let (_, _, terminated, _, _) = env.step(GridAction::Right);
            if i < 3 {
                assert!(!terminated, "Should not terminate at column {}", i + 1);
            } else {
                assert!(terminated, "Should terminate when reaching goal at [4,4]");
            }
        }
    }

    #[test]
    fn goal_reward_is_positive() {
        let mut env = GridWorld::new();
        env.reset(Some(42));

        // Navigate direct path to goal
        for _ in 0..4 {
            env.step(GridAction::Down);
        }
        for i in 0..4 {
            let (_, reward, terminated, _, _) = env.step(GridAction::Right);
            if terminated {
                assert_eq!(reward, 1.0, "Goal reward should be 1.0");
            } else {
                assert!(
                    reward < 0.0,
                    "Step penalty should be negative at step {}",
                    i
                );
            }
        }
    }

    #[test]
    fn step_penalty_is_negative() {
        let mut env = GridWorld::new();
        env.reset(Some(42));

        let (_, reward, _, _, _) = env.step(GridAction::Right);
        assert!(
            reward < 0.0,
            "Non-goal step reward should be negative, got {}",
            reward
        );
    }
}

// Boundary Clamping

mod boundaries {
    use super::*;

    #[test]
    fn upper_left_corner_clamping() {
        let mut env = GridWorld::new();
        env.reset(Some(42));
        // Agent at [0,0] — try to go up (should stay at [0,0])
        let (obs, _, _, _, _) = env.step(GridAction::Up);
        assert_eq!(
            obs,
            [0, 0],
            "Should stay at [0,0] when moving up from top edge"
        );

        // Try to go left (should also stay at [0,0])
        let (obs, _, _, _, _) = env.step(GridAction::Left);
        assert_eq!(
            obs,
            [0, 0],
            "Should stay at [0,0] when moving left from left edge"
        );
    }

    #[test]
    fn bottom_right_corner_clamping() {
        let mut env = GridWorld::new();
        env.reset(Some(42));

        // Navigate to [4,3] (one left of goal to avoid termination)
        for _ in 0..4 {
            env.step(GridAction::Down);
        }
        for _ in 0..3 {
            env.step(GridAction::Right);
        }

        // Now at [4,3]. Move down — should stay at row 4
        let (obs, _, _, _, _) = env.step(GridAction::Down);
        assert_eq!(obs[0], 4, "Should stay at bottom edge");
    }
}

// Determinism

mod determinism {
    use super::*;

    #[test]
    fn seeded_runs_identical() {
        let mut env1 = GridWorld::new();
        let mut env2 = GridWorld::new();

        env1.reset(Some(42));
        env2.reset(Some(42));

        let actions = [
            GridAction::Down,
            GridAction::Right,
            GridAction::Down,
            GridAction::Right,
        ];

        for (i, &action) in actions.iter().enumerate() {
            let r1 = env1.step(action);
            let r2 = env2.step(action);
            assert_eq!(r1.0, r2.0, "Obs diverged at step {}", i);
            assert_eq!(r1.1, r2.1, "Reward diverged at step {}", i);
            assert_eq!(r1.2, r2.2, "Terminated diverged at step {}", i);
        }
    }
}

// Action Enum & Space

mod action_space {
    use super::*;

    #[test]
    fn action_enum_exhaustive() {
        // Compile-time check: all 4 directions covered
        let action = GridAction::Up;
        let _delta = match action {
            GridAction::Up => (-1i32, 0i32),
            GridAction::Down => (1, 0),
            GridAction::Left => (0, -1),
            GridAction::Right => (0, 1),
        };
    }

    #[test]
    fn action_space_is_discrete_4() {
        let env = GridWorld::new();
        assert_eq!(env.action_space(), rustforge_rl::env::Space::discrete(4));
    }

    #[test]
    fn observation_space_dimensionality() {
        let env = GridWorld::new();
        assert_eq!(env.observation_space().dim(), 2);
    }
}

// Grid Configuration

mod configuration {
    use super::*;
    use rustforge_rl::env::gridworld::CellType;

    #[test]
    fn custom_grid_works() {
        let grid = vec![
            vec![CellType::Empty, CellType::Empty],
            vec![CellType::Empty, CellType::Goal],
        ];
        let mut env = GridWorld::with_grid(grid, [0, 0], [1, 1]);
        env.reset(Some(1));

        // Move to goal: down + right
        let (_, _, terminated, _, _) = env.step(GridAction::Down);
        assert!(!terminated);
        let (_, _, terminated, _, _) = env.step(GridAction::Right);
        assert!(terminated, "Should reach goal at [1,1]");
    }

    #[test]
    fn dimensions_correct() {
        let env = GridWorld::new();
        assert_eq!(env.dimensions(), (5, 5));
    }
}
