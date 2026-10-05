//! Host-side GPU training instrumentation, with real CPU environments and replay.
//! Usage: `cargo run --release -p rustforge-rl --features gpu
//! --example gpu_training_profile -- <dqn|td3|sac> <steps>`.
use rand::{rngs::StdRng, SeedableRng};
use rustforge_rl::{
    agent::{
        gpu_sac::GpuSac, gpu_td3::GpuTd3, pendulum_sac_config, pendulum_td3_config, DQNConfig,
        GpuDqn,
    },
    buffer::{ContinuousReplayBuffer, ContinuousTransitionBatch, ReplayBuffer, TransitionBatch},
    env::{CartPole, CartPoleAction, Environment, Pendulum, PendulumAction},
    training::{episode_done, replay_done},
};
use rustforge_tensor::gpu::GpuContext;
use serde_json::{json, Value};
use std::{error::Error, time::Instant};

const SEED: u64 = 42;
const BATCH: usize = 64;
type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn dqn(context: &GpuContext, steps: usize) -> Result<Value> {
    let config = DQNConfig::default();
    let config_json = json!({"obs_dim":config.obs_dim, "num_actions":config.num_actions,
        "hidden_dim":config.hidden_dim, "lr":config.lr, "gamma":config.gamma,
        "target_update_freq":config.target_update_freq, "double_dqn":config.double_dqn,
        "use_per":config.use_per, "per_beta_annealing_steps":config.per_beta_annealing_steps});
    let mut agent = {
        let _scope = context.profile_scope("initialization");
        GpuDqn::new_seeded(context, config, SEED)?
    };
    let mut env = CartPole::with_max_steps(500);
    let mut replay = ReplayBuffer::new(16_384, 4);
    let mut batch = TransitionBatch::new(BATCH, 4);
    let (mut state, _) = env.reset(Some(SEED));
    let mut episodes = 0;
    let mut last_loss = None;
    for _ in 0..steps {
        let action = {
            let _scope = context.profile_scope("inference");
            agent.select_greedy_action(&state)?
        };
        let (next, reward, terminated, truncated, _) = {
            let _scope = context.profile_scope("environment");
            env.step(CartPoleAction::try_from(action)?)
        };
        {
            let _scope = context.profile_scope("replay");
            replay.push(
                &state,
                action,
                reward,
                &next,
                replay_done(terminated, truncated),
            );
            if replay.len() >= BATCH {
                replay.sample(BATCH, &mut batch);
            }
        }
        if replay.len() >= BATCH {
            let uploaded = {
                let _scope = context.profile_scope("batch_upload");
                agent.upload_batch(&batch)?
            };
            let _scope = context.profile_scope("training");
            last_loss = Some(agent.train_device_batch(&uploaded)?);
        }
        state = if episode_done(terminated, truncated) {
            episodes += 1;
            let _scope = context.profile_scope("environment");
            env.reset(None).0
        } else {
            next
        };
    }
    Ok(
        json!({"environment":"cartpole", "config":config_json, "updates":agent.train_steps(),
        "episodes_completed":episodes, "last_loss":last_loss}),
    )
}

enum ContinuousAgent {
    Td3(GpuTd3),
    Sac(GpuSac),
}

fn continuous(context: &GpuContext, algorithm: &str, steps: usize) -> Result<Value> {
    let (mut agent, config) = {
        let _scope = context.profile_scope("initialization");
        if algorithm == "td3" {
            let config = pendulum_td3_config();
            (
                ContinuousAgent::Td3(GpuTd3::new_seeded(context, config.clone(), SEED)?),
                serde_json::to_value(config)?,
            )
        } else {
            let config = pendulum_sac_config();
            (
                ContinuousAgent::Sac(GpuSac::new_seeded(context, config.clone(), SEED)?),
                serde_json::to_value(config)?,
            )
        }
    };
    let mut env = Pendulum::with_max_steps(200);
    let mut replay = ContinuousReplayBuffer::new(16_384, 3, 1);
    let mut batch = ContinuousTransitionBatch::new(BATCH, 3, 1);
    let mut collection = StdRng::seed_from_u64(SEED);
    let mut sampling = StdRng::seed_from_u64(SEED + 1);
    let mut target = StdRng::seed_from_u64(SEED + 2);
    let mut actor = StdRng::seed_from_u64(SEED + 3);
    let (mut state, _) = env.reset(Some(SEED));
    let mut episodes = 0;
    let mut last_metrics = Value::Null;
    for _ in 0..steps {
        let action = {
            let _scope = context.profile_scope("inference");
            match &agent {
                ContinuousAgent::Td3(a) => {
                    a.select_action_with_rng(&state, 0.1, &mut collection)?
                }
                ContinuousAgent::Sac(a) => a.select_action_with_rng(&state, &mut collection)?,
            }
        };
        let (next, reward, terminated, truncated, _) = {
            let _scope = context.profile_scope("environment");
            env.step(PendulumAction::new(action[0]))
        };
        {
            let _scope = context.profile_scope("replay");
            replay.push(
                &state,
                &action,
                reward,
                &next,
                replay_done(terminated, truncated),
            );
            if replay.len() >= BATCH {
                replay.sample_with_rng(BATCH, &mut batch, &mut sampling);
            }
        }
        if replay.len() >= BATCH {
            let _scope = context.profile_scope("training");
            last_metrics = match &mut agent {
                ContinuousAgent::Td3(a) => {
                    let (critic_loss, actor_loss) = a.train_step_with_rng(&batch, &mut target)?;
                    json!({"critic_loss":critic_loss, "actor_loss":actor_loss})
                }
                ContinuousAgent::Sac(a) => {
                    let (critic_loss, actor_loss, alpha_loss, alpha) =
                        a.train_step_with_rngs(&batch, &mut target, &mut actor)?;
                    json!({"critic_loss":critic_loss, "actor_loss":actor_loss, "alpha_loss":alpha_loss, "alpha":alpha})
                }
            };
        }
        state = if episode_done(terminated, truncated) {
            episodes += 1;
            let _scope = context.profile_scope("environment");
            env.reset(None).0
        } else {
            next
        };
    }
    let updates = match &agent {
        ContinuousAgent::Td3(a) => a.updates(),
        ContinuousAgent::Sac(a) => a.updates(),
    };
    Ok(
        json!({"environment":"pendulum", "config":config, "updates":updates,
        "episodes_completed":episodes, "last_metrics":last_metrics}),
    )
}

fn main() -> Result<()> {
    // Validate arguments before initializing an adapter or allocating training state.
    let mut args = std::env::args().skip(1);
    let algorithm = args
        .next()
        .unwrap_or_else(|| "dqn".into())
        .to_ascii_lowercase();
    if !matches!(algorithm.as_str(), "dqn" | "td3" | "sac") {
        return Err("algorithm must be dqn, td3 or sac".into());
    }
    let steps = args
        .next()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(128);
    if steps == 0 || args.next().is_some() {
        return Err("usage: gpu_training_profile [dqn|td3|sac] [positive steps]".into());
    }
    let start = Instant::now();
    let context = GpuContext::new()?.with_profiling();
    let device_initialization_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    let training = match algorithm.as_str() {
        "dqn" => dqn(&context, steps)?,
        _ => continuous(&context, &algorithm, steps)?,
    };
    {
        let _scope = context.profile_scope("final_completion");
        context.synchronize();
    }
    let host_elapsed_ns = start.elapsed().as_nanos();
    let adapter = context.adapter_info();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"rustforge-gpu-training-profile-v1", "algorithm":algorithm,
        "steps":steps, "seed":SEED, "batch_size":BATCH,
        "replay_seed":if algorithm == "dqn" { None } else { Some(SEED + 1) },
        "collection_policy":"inference_every_step", "timing_kind":"host_wall_inclusive",
        "phase_times_overlap":true,
            "device_initialization_ns":device_initialization_ns, "host_elapsed_ns":host_elapsed_ns,
            "adapter":{"name":adapter.name, "backend":format!("{:?}", adapter.backend),
                "device_type":format!("{:?}", adapter.device_type)},
            "training":training, "profile":context.profile_snapshot().expect("profiling enabled")
        }))?
    );
    Ok(())
}
