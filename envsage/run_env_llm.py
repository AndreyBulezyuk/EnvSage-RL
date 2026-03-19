from __future__ import annotations

import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'envsage.settings')
from django.conf import settings

# Configure Django
if not settings.configured:
    django.setup()

import argparse
import importlib.util
import json
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces.utils import flatten, flatdim
from torch.utils.tensorboard import SummaryWriter

from agent.llm_scientist_agent import EnvSageAgent
from agent.models import Experiment, Episode
import logging
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generic environment runner for the LLM scientist agent")
    p.add_argument("--env-id", type=str, default="LunarLander-v3")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--session-name", type=str, default=f"session_{int(time())}")
    p.add_argument("--model", type=str, default="gpt-5.1")
    p.add_argument("--render", action="store_true")
    p.add_argument("--export-every", type=int, default=10)
    p.add_argument("--use-exported-policy", type=str, default="")
    p.add_argument("--env-kwargs-json", type=str, default="{}")
    return p.parse_args()


def make_session_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_programmatic_policy(policy_path: str):
    policy_path = str(policy_path)
    spec = importlib.util.spec_from_file_location("exported_programmatic_policy", policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load programmatic policy from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "act"):
        raise RuntimeError(f"policy file {policy_path} does not define act(obs)")
    return module.act


def make_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    return value


# ------------------------------------------------------------------
# Space helpers
# ------------------------------------------------------------------
def describe_space(space: spaces.Space) -> Dict[str, Any]:
    if isinstance(space, spaces.Discrete):
        return {"space_type": "Discrete", "n": int(space.n), "flat_dim": 1}
    if isinstance(space, spaces.Box):
        low = np.asarray(space.low, dtype=float).reshape(-1)
        high = np.asarray(space.high, dtype=float).reshape(-1)
        return {
            "space_type": "Box",
            "shape": list(space.shape),
            "dtype": str(space.dtype),
            "flat_dim": int(flatdim(space)),
            "low_flat": low.tolist(),
            "high_flat": high.tolist(),
        }
    if isinstance(space, spaces.MultiDiscrete):
        return {
            "space_type": "MultiDiscrete",
            "nvec": np.asarray(space.nvec, dtype=int).reshape(-1).tolist(),
            "flat_dim": int(flatdim(space)),
        }
    if isinstance(space, spaces.MultiBinary):
        n = int(np.prod(space.shape)) if hasattr(space, "shape") else int(space.n)
        return {"space_type": "MultiBinary", "n": n, "flat_dim": n}
    if isinstance(space, spaces.Dict):
        return {
            "space_type": "Dict",
            "flat_dim": int(flatdim(space)),
            "keys": list(space.spaces.keys()),
            "subspaces": {k: describe_space(v) for k, v in space.spaces.items()},
        }
    if isinstance(space, spaces.Tuple):
        return {
            "space_type": "Tuple",
            "flat_dim": int(flatdim(space)),
            "subspaces": [describe_space(s) for s in space.spaces],
        }
    return {"space_type": type(space).__name__, "flat_dim": int(flatdim(space))}


def build_env_schema(env: gym.Env, env_id: str, env_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "env_id": env_id,
        "env_kwargs": make_jsonable(env_kwargs),
        "observation_space": describe_space(env.observation_space),
        "action_space": describe_space(env.action_space),
    }


def flatten_observation(obs: Any, obs_space: spaces.Space) -> List[float]:
    # round observations to 4 decimals to reduce LLM input size and noise from float precision
    arr = np.asarray(flatten(obs_space, obs), dtype=float).reshape(-1)
    arr = arr.round(4)
    return arr.tolist()


def decode_action(action_literal: Any, action_space: spaces.Space) -> Any:
    if isinstance(action_space, spaces.Discrete):
        return int(np.clip(int(action_literal), 0, action_space.n - 1))
    if isinstance(action_space, spaces.Box):
        arr = np.asarray(action_literal, dtype=action_space.dtype).reshape(action_space.shape)
        return np.clip(arr, action_space.low, action_space.high).astype(action_space.dtype)
    if isinstance(action_space, spaces.MultiDiscrete):
        arr = np.asarray(action_literal, dtype=int).reshape(action_space.nvec.shape)
        return np.minimum(np.maximum(arr, 0), action_space.nvec - 1).astype(action_space.dtype)
    if isinstance(action_space, spaces.MultiBinary):
        shape = action_space.shape if hasattr(action_space, "shape") else (action_space.n,)
        arr = np.asarray(action_literal, dtype=int).reshape(shape)
        return (arr > 0).astype(action_space.dtype)
    raise NotImplementedError(f"Unsupported action space for execution: {type(action_space).__name__}")


def action_to_jsonable(action: Any, action_space: spaces.Space) -> Any:
    if isinstance(action_space, spaces.Discrete):
        return int(action)
    arr = np.asarray(action).reshape(-1)
    if isinstance(action_space, spaces.Box):
        return arr.astype(float).tolist()
    return arr.astype(int).tolist()


# ------------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------------
def add_obs_logs(writer: SummaryWriter, trajectory: List[Dict[str, Any]], episode_idx: int) -> None:
    if not trajectory:
        return
    obs_arr = np.asarray([x["obs_flat"] for x in trajectory], dtype=float)
    next_arr = np.asarray([x["next_obs_flat"] for x in trajectory], dtype=float)
    deltas = next_arr - obs_arr
    writer.add_histogram("obs/all_flat", obs_arr.flatten(), episode_idx)
    writer.add_histogram("obs/deltas_flat", deltas.flatten(), episode_idx)
    for i in range(obs_arr.shape[1]):
        writer.add_scalar(f"obs/mean_{i}", float(obs_arr[:, i].mean()), episode_idx)
        writer.add_scalar(f"obs/std_{i}", float(obs_arr[:, i].std()), episode_idx)
        writer.add_scalar(f"deltas/mean_{i}", float(deltas[:, i].mean()), episode_idx)
        writer.add_scalar(f"deltas/std_{i}", float(deltas[:, i].std()), episode_idx)


def add_action_logs(writer: SummaryWriter, trajectory: List[Dict[str, Any]], action_space: spaces.Space, episode_idx: int) -> None:
    if not trajectory:
        return
    if isinstance(action_space, spaces.Discrete):
        arr = np.asarray([int(x["action_json"]) for x in trajectory], dtype=int)
        counts = np.bincount(arr, minlength=action_space.n).astype(float)
        counts /= max(counts.sum(), 1.0)
        for i in range(action_space.n):
            writer.add_scalar(f"actions/share_{i}", float(counts[i]), episode_idx)
        return
    flat_actions = np.asarray([
        np.asarray(x["action_json"], dtype=float).reshape(-1) if isinstance(x["action_json"], list) else np.asarray([x["action_json"]], dtype=float)
        for x in trajectory
    ], dtype=float)
    writer.add_histogram("actions/all_flat", flat_actions.flatten(), episode_idx)
    for i in range(flat_actions.shape[1]):
        writer.add_scalar(f"actions/mean_{i}", float(flat_actions[:, i].mean()), episode_idx)
        writer.add_scalar(f"actions/std_{i}", float(flat_actions[:, i].std()), episode_idx)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    session_dir = Path("runs") / args.session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(session_dir / "tensorboard" / f"run_{args.env_id}_{args.model}_{args.seed}"))

    env_kwargs = json.loads(args.env_kwargs_json)
    if args.render:
        env_kwargs["render_mode"] = "human"
    env = gym.make(args.env_id, **env_kwargs)

    env_schema = build_env_schema(env, args.env_id, env_kwargs)
    agent = EnvSageAgent(
        gym_env=env,
        session_name=args.session_name,
        model=args.model,
    )
    agent.set_env_schema(env_schema)
    last_export_path: Optional[str] = None
    all_returns: List[float] = []
    writer.add_text("run/config", json.dumps({
        "env_schema": env_schema,
        "model": args.model,
        "seed": args.seed,
    }, indent=2), 0)
    # Create dummy programmatic policy file
    export_dir = session_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    default_policy_path = export_dir / "exported_programmatic_policy.py"
    default_policy_path.write_text(
        "def act(obs, step):\n"
        "    return 0\n"
    )
    

    try:
        for episode_idx in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode_idx)
            # round observations to 4 decimals to reduce LLM input size and noise from float precision
            obs = np.asarray(obs, dtype=float).round(4).tolist() if isinstance(obs, (np.ndarray, list, tuple)) else obs
            flat_obs = flatten_observation(obs, env.observation_space)
            trajectory: List[Dict[str, Any]] = []
            reward_trace: List[float] = []
            
            if episode_idx == 0:
                logger.info(f"Starting run with config: {json.dumps({'env_schema': env_schema, 'model': args.model, 'seed': args.seed}, indent=2)}")
                logger.info("Creating default hypothesis and experiments.")
                
            
            # load policy every episode to fetch the newest python file version     
            exported_policy = None
            if args.use_exported_policy:
                exported_policy = load_programmatic_policy(args.use_exported_policy)
                episode_mode = "programmatic_policy"
                experiment_plan = None
                episode_mode_reason = f"using imported policy: {args.use_exported_policy}"
            else:
                episode_mode, experiment_plan, episode_mode_reason = agent.decide_episode_mode(flat_obs, episode_idx)

            terminated = False
            truncated = False
            episode_return = 0.0
            last_action_reason = ""
            
            for step_idx in range(args.max_steps):
                if exported_policy is not None:
                    action_literal = exported_policy(flat_obs, step_idx)
                    action_meta = {"reason": "exported policy", "source": "programmatic_policy"}
                elif episode_mode == "experiment" and experiment_plan is not None:
                    #update_experiment = Experiment.objects.filter(session=args.session_name, name=experiment_plan.name).update(status='running')
                    #if update_experiment == 0:
                    #    logger.error(f"Experiment with name {experiment_plan.name} not found in database {update_experiment} to update to running status.")
                    action_literal = agent.get_experiment_action(flat_obs, experiment_plan, step_idx)
                    action_meta = {"reason": experiment_plan.description, "source": "experiment_schedule"}
                else:
                    exported_policy = load_programmatic_policy(str(session_dir / "exports" / "exported_programmatic_policy.py"))
                    action_literal = exported_policy(flat_obs, step_idx)
                    action_meta = {"reason": "control exported policy", "source": "control programmatic_policy"}

                env_action = decode_action(action_literal, env.action_space)
                next_obs, reward, terminated, truncated, info = env.step(env_action)
                next_obs = np.asarray(next_obs, dtype=float).round(4).tolist() if isinstance(next_obs, (np.ndarray, list, tuple)) else next_obs
                reward = round(float(reward), 4)
                logger.debug(f"Episode ({episode_mode}) {episode_idx} Step {step_idx}: action={env_action}, reward={reward}, terminated={terminated}, truncated={truncated}, info={info}")

                flat_next_obs = flatten_observation(next_obs, env.observation_space)
                action_json = action_to_jsonable(env_action, env.action_space)

                trajectory.append(
                    {
                        "t": step_idx,
                        "obs_flat": flat_obs,
                        "action_json": action_json,
                        "reward": reward,
                        "next_obs_flat": flat_next_obs,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                
                reward_trace.append(float(reward))
                episode_return += float(reward)
                flat_obs = flat_next_obs
                obs = next_obs
                last_action_reason = str(action_meta.get("reason", ""))

                if terminated or truncated:
                    break

            all_returns.append(float(episode_return))

            analysis = {"summary": {}, "analysis": {}}
            analysis = agent.post_episode(
                episode_index=episode_idx,
                mode=episode_mode,
                trajectory=trajectory,
                return_value=episode_return,
                terminated=terminated,
                truncated=truncated,
                experiment_plan=experiment_plan,
            )

            writer.add_scalar("episode/return", episode_return, episode_idx)
            writer.add_scalar("episode/length", len(trajectory), episode_idx)
            writer.add_scalar("episode/terminated", int(terminated), episode_idx)
            writer.add_scalar("episode/truncated", int(truncated), episode_idx)
            writer.add_scalar("episode/mean_reward", float(np.mean(reward_trace)) if reward_trace else 0.0, episode_idx)
            writer.add_scalar("episode/rolling_return_10", float(np.mean(all_returns[-10:])), episode_idx)
            writer.add_scalar("agent/llm_calls_total", agent.get_llm_call_count(), episode_idx)
            writer.add_scalar("agent/mode_experiment", int(episode_mode == "experiment"), episode_idx)
            writer.add_scalar("agent/mode_control", int(episode_mode == "control"), episode_idx)
            writer.add_scalar("agent/mode_programmatic_policy", int(episode_mode == "programmatic_policy"), episode_idx)

            add_obs_logs(writer, trajectory, episode_idx)
            add_action_logs(writer, trajectory, env.action_space, episode_idx)

            if exported_policy is None:
                top_hypotheses = agent.get_top_hypotheses(limit=8)
                constants = agent.get_constants(limit=8)
                interpretations = agent.get_interpretations(limit=8)
                writer.add_scalar("memory/num_hypotheses", len(top_hypotheses), episode_idx)
                writer.add_scalar(
                    "memory/avg_hypothesis_confidence",
                    float(np.mean([float(x.get("confidence", 0.0)) for x in top_hypotheses])) if top_hypotheses else 0.0,
                    episode_idx,
                )
                writer.add_scalar("memory/num_constants", len(constants), episode_idx)
                writer.add_scalar("memory/num_interpretations", len(interpretations), episode_idx)
                writer.add_text("agent/episode_mode_reason", episode_mode_reason[:4000], episode_idx)
                writer.add_text("agent/last_action_reason", last_action_reason[:4000], episode_idx)
                writer.add_text("agent/top_hypotheses", json.dumps(top_hypotheses, indent=2)[:12000], episode_idx)
                writer.add_text("agent/constants", json.dumps(constants, indent=2)[:12000], episode_idx)
                writer.add_text("agent/interpretations", json.dumps(interpretations, indent=2)[:12000], episode_idx)
                if experiment_plan is not None:
                    writer.add_text("agent/experiment_plan", json.dumps({
                        "name": experiment_plan.name,
                        "description": experiment_plan.description,
                        "target_question": experiment_plan.target_question,
                        "reason": experiment_plan.reason,
                        "custom_action_python_oneline_method": experiment_plan.custom_action_python_oneline_method,
                    }, indent=2), episode_idx)
                writer.add_text("agent/post_episode_analysis", json.dumps(analysis.get("analysis", {}), indent=2)[:12000], episode_idx)

            episode_artifact = session_dir / f"episode_{episode_idx:05d}_trajectory.json"
            episode_artifact.write_text(json.dumps({
                "episode_index": episode_idx,
                "mode": episode_mode,
                "reason": episode_mode_reason,
                "return": episode_return,
                "length": len(trajectory),
                "terminated": terminated,
                "truncated": truncated,
                "trajectory": make_jsonable(trajectory),
                "info_last": make_jsonable(info),
            }, indent=2))

            print(
                f"episode={episode_idx:04d} mode={episode_mode:<18} return={episode_return:9.3f} "
                f"len={len(trajectory):4d} llm_calls={agent.get_llm_call_count()}"
            )

            if exported_policy is None and ((episode_idx + 1) % max(args.export_every, 1) == 0):
                logger.info(f"Exporting programmatic policy at episode {episode_idx}...")
                last_export_path = agent.export_programmatic_policy()
                writer.add_text("agent/exported_policy_path", last_export_path, episode_idx)

    finally:
        env.close()
        writer.flush()
        writer.close()

    summary = {
        "episodes": args.episodes,
        "mean_return": float(np.mean(all_returns)) if all_returns else 0.0,
        "session_dir": str(session_dir),
        "last_export_path": last_export_path,
    }
    (session_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
