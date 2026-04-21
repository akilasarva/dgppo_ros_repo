import argparse
import datetime
import functools as ft
import os
import pathlib

import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml
import json

from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.trainer.utils import test_rollout
from dgppo.utils.graph import GraphsTuple
from dgppo.utils.utils import jax_jit_np, jax_vmap
from dgppo.utils.typing import Array, PRNGKey
from typing import Optional


def test(args):
    print(f"> Running test.py {args}")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    with open(os.path.join(args.path, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    num_agents = config.get('num_agents', args.num_agents) if args.num_agents is not None else config.get('num_agents')
    env_id = config.get('env', args.env) if args.env is not None else config.get('env')
    num_obs = config.get('obs', args.obs) if args.obs is not None else config.get('obs')
    full_observation = config.get('full_observation', args.full_observation) if args.full_observation is not None else config.get('full_observation')
    n_rays = config.get('n_rays', 32)

    env = make_env(
        env_id=env_id,
        num_agents=num_agents,
        num_obs=num_obs,
        max_step=args.max_step,
        full_observation=full_observation,
        n_rays=n_rays,
    )

    path = args.path
    model_path = os.path.join(path, "models")

    if args.step is not None:
        step = args.step
        print(f"Loading model from specified step: {step}")
    elif args.load_best:
        best_model_info_path = os.path.join(path, 'best_model_info.json')
        if os.path.exists(best_model_info_path):
            with open(best_model_info_path, 'r') as f:
                best_info = json.load(f)
            step = best_info.get('best_step', None)
            if step is not None:
                print(f"Loading BEST model found at step: {step} (reward: {best_info.get('best_eval_reward'):.3f})")
            else:
                print(f"Warning: 'best_step' not found in best_model_info.json. Falling back to latest step.")
        else:
            print(f"Warning: best_model_info.json not found in {path}. Falling back to latest step.")
        
        if step is None:
            models = os.listdir(model_path)
            step = max([int(model) for model in models if model.isdigit()])
            print("Falling back to latest step: ", step)
    else:
        models = os.listdir(model_path)
        step = max([int(model) for model in models if model.isdigit()])
        print("Loading latest step: ", step)
    
    if step is None:
        raise ValueError("Could not determine which model step to load. Ensure a valid checkpoint path and/or --step/--load-best arguments.")

    algo = make_algo(
        algo=config.get('algo'),
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        cost_weight=config.get('cost_weight', 0.),
        cbf_weight=config.get('cbf_weight', 1.0),
        actor_gnn_layers=config.get('actor_gnn_layers', 2),
        Vl_gnn_layers=config.get('Vl_gnn_layers', 2),
        Vh_gnn_layers=config.get('Vh_gnn_layers', 1),
        rnn_layers=config.get('rnn_layers', 1),
        lr_actor=config.get('lr_actor', 3e-4),
        lr_Vl=config.get('lr_Vl', 1e-3),
        lr_Vh=config.get('lr_Vh', 1e-3),
        max_grad_norm=config.get('max_grad_norm', 2.0),
        alpha=config.get('alpha', 10.0),
        cbf_eps=config.get('cbf_eps', 1e-2),
        seed=config.get('seed', 0),
        batch_size=config.get('batch_size', 16384),
        use_rnn=config.get('use_rnn', True),
        use_lstm=config.get('use_lstm', False),
        coef_ent=config.get('coef_ent', 1e-2),
        rnn_step=config.get('rnn_step', 16),
        gamma=config.get('gamma', 0.99),
        clip_eps=config.get('clip_eps', 0.25),
        lagr_init=config.get('lagr_init', 0.5),
        lr_lagr=config.get('lr_lagr', 1e-7),
        train_steps=config.get('steps', 200000),
        cbf_schedule=config.get('cbf_schedule', True),
        cost_schedule=config.get('cost_schedule', False)
    )
    
    # algo.load(model_path, step)

    # if args.stochastic:
    #     def act_fn(x, rnn_state, key):
    #         action, _, new_rnn_state = algo.step(x, rnn_state, key, params=algo.params)
    #         return action, new_rnn_state
    # else:
    #     def act_fn(x, rnn_state, key):
    #         action, _, new_rnn_state = algo.step(x, rnn_state, key=key, params=algo.params)
    #         return action, new_rnn_state
    
    # act_fn = jax.jit(act_fn)
    # init_rnn_state = algo.init_rnn_state
    
    algo.load(model_path, step)
    if args.stochastic:
        def act_fn(x, z, rnn_state, key):
            action, _, new_rnn_state = algo.step(x, z, rnn_state, key)
            return action, new_rnn_state
        act_fn = jax.jit(act_fn)
    else:
        act_fn = algo.act
    act_fn = jax.jit(act_fn)
    init_rnn_state = algo.init_rnn_state

    # set up keys
    test_key = jr.PRNGKey(args.seed)

    @jax_jit_np
    def rollout_fn(key: PRNGKey, start_graph: GraphsTuple, transition_index: Optional[int]):
        return test_rollout(env, act_fn, init_rnn_state, key, stochastic=args.stochastic, start_graph=start_graph, transition_index=transition_index)
    
    def unsafe_mask(graph_: GraphsTuple) -> Array:
        cost = env.get_cost(graph_)
        return jnp.any(cost >= 0.0, axis=-1)

    is_unsafe_fn = jax_jit_np(jax_vmap(unsafe_mask))
    
    rewards = []
    costs = []
    rollouts = []
    is_unsafes = []
    rates = []

    if args.test_all_transitions:
        print(f"Testing {args.epi} long episodes, each covering all 3 transitions (approach -> on -> exit).")
        
        for i_epi in range(args.epi):
            print(f"\n--- Running Full Episode {i_epi} ---")

            test_key, reset_key = jr.split(test_key, 2)
            
            # Start the episode at the beginning of the curriculum (approach_bridge_0).
            start_graph = env.reset(reset_key, transition_index=0)

            # Run one full rollout from start to finish.
            # The final transition_index tells the environment where the ultimate goal is located.
            rollout = rollout_fn(reset_key, start_graph=start_graph, transition_index=2)
                
            is_unsafe = is_unsafe_fn(rollout.graph)
            epi_reward = rollout.rewards.sum()
            epi_cost = rollout.costs.max() if rollout.costs.size > 0 else 0.0
            safe_rate = 1 - jnp.any(is_unsafe, axis=-1).mean()
            
            rewards.append(epi_reward)
            costs.append(epi_cost)
            rollouts.append(rollout)
            is_unsafes.append(is_unsafe)
            rates.append(np.array(safe_rate))
                
            print(f"  > Episode {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, safe rate: {safe_rate * 100:.3f}%")
    else:
        test_keys = jr.split(test_key, 1_000)[: args.epi]
        test_keys = test_keys[args.offset:]
        
        for i_epi in range(args.epi):
            key_x0, _ = jr.split(test_keys[i_epi], 2)
            initial_graph = env.reset(key_x0)
            rollout = rollout_fn(key_x0, start_graph=initial_graph, transition_index=None)
            
            is_unsafe = is_unsafe_fn(rollout.graph)
            is_unsafes.append(is_unsafe)

            epi_reward = rollout.rewards.sum()
            epi_cost = rollout.costs.max() if rollout.costs.size > 0 else 0.0
            
            rewards.append(epi_reward)
            costs.append(epi_cost)
            rollouts.append(rollout)
            safe_rate = 1 - jnp.any(is_unsafe, axis=-1).mean()
            print(f"epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, safe rate: {safe_rate * 100:.3f}%")
            rates.append(np.array(safe_rate))
    
    is_unsafe = np.max(np.stack(is_unsafes), axis=1)
    safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()

    print(
        f"reward: {np.mean(rewards):.3f}, min/max reward: {np.min(rewards):.3f}/{np.max(rewards):.3f}, "
        f"cost: {np.mean(costs):.3f}, min/max cost: {np.min(costs):.3f}/{np.max(costs):.3f}, "
        f"safe_rate: {safe_mean * 100:.3f}%"
    )

    if args.log:
        env_area_size = getattr(env, 'area_size', 'N/A')
        env_n_obs = env.params.get('n_obs', 'N/A') if hasattr(env, 'params') and isinstance(env.params, dict) else 'N/A'
        with open(os.path.join(path, "test_log.csv"), "a") as f:
            f.write(f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                    f"{env_area_size},{env_n_obs},"
                    f"{safe_mean * 100:.3f},{safe_std * 100:.3f}\n")

    if args.no_video:
        return

    videos_dir = pathlib.Path(path) / "videos" / f"{step}"
    videos_dir.mkdir(exist_ok=True, parents=True)
    for ii, (rollout, Ta_is_unsafe) in enumerate(zip(rollouts, is_unsafes)):
        safe_rate = rates[ii] * 100
        video_name = f"n{num_agents}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}_sr{safe_rate:.0f}"
        viz_opts = {}
        video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
        env.render_video(rollout, video_path, Ta_is_unsafe, viz_opts, dpi=args.dpi)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, required=True,
                        help="Path to the log directory of a trained model.")

    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--epi", type=int, default=1,
                        help="Number of episodes to run for testing.")
    parser.add_argument("--step", type=int, default=None,
                        help="Specific training step to load the model from.")
    parser.add_argument("--obs", type=int, default=None,
                        help="Override observation dimension from config.yaml.")
    parser.add_argument("--stochastic", action="store_true", default=False)
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--log", action="store_true", default=False)

    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument('--load-best', action='store_true')
    parser.add_argument("--test-all-transitions", action="store_true", default=False,
                        help="Test all curriculum transitions sequentially instead of random episodes.")

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()