#![cfg(feature = "gpu")]
use rand::{rngs::StdRng, SeedableRng};
use rustforge_rl::agent::{gpu_sac::GpuSac, gpu_td3::GpuTd3, sac::SACConfig, td3::TD3Config};
use rustforge_tensor::gpu::GpuContext;

#[test]
#[ignore = "requires a hardware or software wgpu adapter"]
fn td3_and_sac_inference_keep_network_checks_and_pack_final_actions() {
    let c = GpuContext::new().unwrap().with_profiling();
    let mut td3_config = TD3Config::new(2, 2, vec![-2., 1.], vec![4., 5.]);
    td3_config.hidden_dim = 8;
    let td3 = GpuTd3::new_seeded(&c, td3_config, 42).unwrap();
    let before = c.profile_snapshot().unwrap().counters;
    let action = td3
        .select_action_with_rng(&[0.2, -0.3], 0., &mut StdRng::seed_from_u64(9))
        .unwrap();
    let after = c.profile_snapshot().unwrap().counters;
    assert_eq!(
        (
            after.readbacks - before.readbacks,
            after.host_waits - before.host_waits
        ),
        (2, 2)
    );
    assert!((-2. ..=4.).contains(&action[0]) && (1. ..=5.).contains(&action[1]));
    let mut sac_config = SACConfig::new(2, 2, vec![-2., 1.], vec![4., 5.]);
    sac_config.hidden_dim = 8;
    let sac = GpuSac::new_seeded(&c, sac_config, 42).unwrap();
    for deterministic in [true, false] {
        let before = c.profile_snapshot().unwrap().counters;
        let action = if deterministic {
            sac.deterministic_action(&[0.2, -0.3]).unwrap()
        } else {
            sac.select_action_with_rng(&[0.2, -0.3], &mut StdRng::seed_from_u64(9))
                .unwrap()
        };
        let after = c.profile_snapshot().unwrap().counters;
        assert_eq!(
            (
                after.readbacks - before.readbacks,
                after.host_waits - before.host_waits
            ),
            (2, 2)
        );
        assert!((-2. ..=4.).contains(&action[0]) && (1. ..=5.).contains(&action[1]));
    }
}
