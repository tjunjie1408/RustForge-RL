pub mod monitor;
pub mod run;
pub mod train;

use rustforge_autograd::Variable;
use rustforge_nn::Module;
use rustforge_rl::agent::{DQNConfig, DQN};
use rustforge_tensor::Tensor;

pub fn export_graph() -> anyhow::Result<()> {
    let agent = DQN::new(DQNConfig::default());
    let input = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[1, 4]);
    let output = agent.q_net().forward(&Variable::new(input, true));
    println!("{}", output.export_graphviz());
    Ok(())
}
