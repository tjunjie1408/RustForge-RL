//! GridWorld environment — discrete 2D grid maze with walls, start, and goal.
//!
//! # Overview
//!
//! A simple grid-based environment where an agent navigates from a start position
//! to a goal position, avoiding walls. Useful for testing tabular RL methods
//! and simple policy gradient algorithms.
//!
//! # State Space
//!
//! `[usize; 2]` = `[row, col]` — the agent's current position on the grid.
//!
//! # Action Space
//!
//! `GridAction` enum: `Up`, `Down`, `Left`, `Right`.
//! Compile-time exhaustive — no invalid actions possible.
//!
//! # Reward Structure
//!
//! - Reaching the goal: +1.0 (episode terminates)
//! - Each step: -0.01 (encourages shortest path)
//! - Walking into a wall or boundary: position unchanged, step penalty still applied

use rand::rngs::StdRng;
use rand::SeedableRng;

use super::spaces::Space;
use super::traits::Environment;

/// Grid cell types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellType {
    /// Empty cell — agent can walk here.
    Empty,
    /// Wall cell — agent cannot enter.
    Wall,
    /// Goal cell — episode terminates with +1.0 reward.
    Goal,
}

/// GridWorld action: four cardinal directions.
///
/// Using an enum instead of `usize` guarantees compile-time action validity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridAction {
    Up,
    Down,
    Left,
    Right,
}

/// GridWorld environment with configurable grid layout.
///
/// Default grid (5×5):
/// ```text
/// S . . . .
/// . # . # .
/// . # . . .
/// . . . # .
/// . . . . G
/// ```
/// Where S=start, G=goal, #=wall, .=empty
pub struct GridWorld {
    /// Grid layout (row-major)
    grid: Vec<Vec<CellType>>,
    /// Current agent position [row, col]
    agent_pos: [usize; 2],
    /// Starting position [row, col]
    start_pos: [usize; 2],
    /// Goal position [row, col]
    goal_pos: [usize; 2],
    /// Grid dimensions
    rows: usize,
    cols: usize,
    /// Internal PRNG for reproducibility
    rng: StdRng,
}

impl GridWorld {
    /// Create a new GridWorld with the default 5×5 layout.
    pub fn new() -> Self {
        let grid = vec![
            vec![
                CellType::Empty,
                CellType::Empty,
                CellType::Empty,
                CellType::Empty,
                CellType::Empty,
            ],
            vec![
                CellType::Empty,
                CellType::Wall,
                CellType::Empty,
                CellType::Wall,
                CellType::Empty,
            ],
            vec![
                CellType::Empty,
                CellType::Wall,
                CellType::Empty,
                CellType::Empty,
                CellType::Empty,
            ],
            vec![
                CellType::Empty,
                CellType::Empty,
                CellType::Empty,
                CellType::Wall,
                CellType::Empty,
            ],
            vec![
                CellType::Empty,
                CellType::Empty,
                CellType::Empty,
                CellType::Empty,
                CellType::Goal,
            ],
        ];
        let rows = grid.len();
        let cols = grid[0].len();

        GridWorld {
            grid,
            agent_pos: [0, 0],
            start_pos: [0, 0],
            goal_pos: [rows - 1, cols - 1],
            rows,
            cols,
            rng: StdRng::from_entropy(),
        }
    }

    /// Create a GridWorld with a custom grid layout.
    ///
    /// # Arguments
    /// - `grid`: 2D vector of `CellType`
    /// - `start`: Starting position `[row, col]`
    /// - `goal`: Goal position `[row, col]`
    pub fn with_grid(grid: Vec<Vec<CellType>>, start: [usize; 2], goal: [usize; 2]) -> Self {
        let rows = grid.len();
        let cols = if rows > 0 { grid[0].len() } else { 0 };

        GridWorld {
            grid,
            agent_pos: start,
            start_pos: start,
            goal_pos: goal,
            rows,
            cols,
            rng: StdRng::from_entropy(),
        }
    }

    /// Get the agent's current position.
    pub fn agent_position(&self) -> [usize; 2] {
        self.agent_pos
    }

    /// Get the grid dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

impl Default for GridWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for GridWorld {
    type Obs = [usize; 2];
    type Act = GridAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> (Self::Obs, Self::Info) {
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        }

        self.agent_pos = self.start_pos;
        (self.agent_pos, ())
    }

    fn step(&mut self, action: Self::Act) -> (Self::Obs, f32, bool, bool, Self::Info) {
        let [row, col] = self.agent_pos;

        // Compute candidate position based on action
        let (new_row, new_col) = match action {
            GridAction::Up => (row.saturating_sub(1), col),
            GridAction::Down => ((row + 1).min(self.rows - 1), col),
            GridAction::Left => (row, col.saturating_sub(1)),
            GridAction::Right => (row, (col + 1).min(self.cols - 1)),
        };

        // Only move if the target cell is not a wall
        if self.grid[new_row][new_col] != CellType::Wall {
            self.agent_pos = [new_row, new_col];
        }

        // Check if goal reached
        let terminated = self.agent_pos == self.goal_pos;

        // Reward: +1.0 for goal, -0.01 step penalty otherwise
        let reward = if terminated { 1.0 } else { -0.01 };

        (self.agent_pos, reward, terminated, false, ())
    }

    fn action_space(&self) -> Space {
        Space::discrete(4)
    }

    fn observation_space(&self) -> Space {
        Space::continuous(
            vec![0.0, 0.0],
            vec![(self.rows - 1) as f32, (self.cols - 1) as f32],
        )
    }
}
