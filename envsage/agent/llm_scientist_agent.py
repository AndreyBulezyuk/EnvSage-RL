from __future__ import annotations

import ast
import json
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import traceback
import django
from django.conf import settings
import numpy as np
import gymnasium as gym
import openai
from agent.llm import LLMAgent
import importlib.util


if not settings.configured:
    django.setup()

from .models import Session, Episode, Experiment, Hypothesis, Interpretation, Constant, LLMAudit, LLMCall

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

CanonicalAction = Union[int, float, List[float], List[int]] # matches the flattened action space format for any gym environment

@dataclass
class ExperimentPlan:
    name: str
    description: str
    target_question: str
    custom_action_python_oneline_method: str
    reason: str = ""
    related_to_hypothesis: int = 1
    id: Optional[int] = None
    status: str = "planned"  # planned, active, completed, failed


class EnvSageAgent:
    """
    Environment-agnostic LLM scientist agent.

    Core design:
    - the LLM proposes hypotheses and experiments dynamically from evidence,
    - experiment plans are python lambda functions that specifically address and test the hypotheses
    - the runner owns env execution / flattening / action decoding,
    - the agent stores memory in SQLite and can export a programmatic policy.
    """

    def __init__(
        self,
        gym_env: gym.Env,
        model: str = "gpt-5.1",
        session_name: str = "default_session",
        request_timeout_s: int = 120,
        top_k_hypotheses: int = 10,
    ) -> None:
        self.model = model
        self.llm_agent = LLMAgent(model=model, session_id=session_name)
        self.session_name = session_name
        self.session = Session.objects.get_or_create(name=session_name)[0]
        self.request_timeout_s = request_timeout_s
        self.top_k_hypotheses = top_k_hypotheses
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment")
        self.session_dir = Path("runs") / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir = self.session_dir / "exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.session_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._llm_calls = 0
        self._last_control_reason = ""
        self.env_schema: Dict[str, Any] = {}
        self.gym_env = gym_env
        self.possible_actions = self.gym_env.action_space

        self._generic_bootstrap_plan() # initialize basic hypothesis & experiment

        logger.info(f"Initialized EnvSageAgent session={self.session_name} model={self.model}")
        logger.debug(f"Session dir: {self.session_dir}")



    # ------------------------------------------------------------------
    # Env schema
    # ------------------------------------------------------------------
    def set_env_schema(self, env_schema: Dict[str, Any]) -> None:
        self.env_schema = dict(env_schema)
        logger.info("Environment schema set")
        logger.debug(f"env_schema keys: {list(self.env_schema.keys())}")

    def _get_env_schema(self) -> Dict[str, Any]:
        return self.env_schema

    def _load_programmatic_policy(self):
        policy_path = str(f"runs/{self.session_name}/exports/exported_programmatic_policy.py")
        spec = importlib.util.spec_from_file_location("exported_programmatic_policy", policy_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load programmatic policy from {policy_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "act"):
            raise RuntimeError(f"policy file {policy_path} does not define act(obs)")
        return module.act

    def get_action(self, flat_obs: list[float] = [], step_idx:int = 0, episode_idx: int = 0):
        """
        Decide whether this Episode is Experiement or Programmatic policy and return 
        """
        if flat_obs == []:
            raise ValueError("flat_obs is an empty array")
        
        try:
            unresolved_experiment = Experiment.objects.filter(("status", "pending")).first() or None # get experiments with no conclusion
            logger.debug("Unresolved Experiments Django Query:")
            logger.debug(unresolved_experiment)
            if unresolved_experiment is not None:
                logger.info(f"Running pending EXPERIMENT {unresolved_experiment.id} for episode {episode_idx}")
                action_literal = self.get_experiment_action(flat_obs=flat_obs, 
                                                            experiment=unresolved_experiment, step_idx=step_idx)
                return action_literal
            else:
                logger.info(f"Decided on CONTROL mode for episode {episode_idx} with reason")
                exported_policy = None # load policy every episode to fetch the newest python file version     
                exported_policy = self._load_programmatic_policy()
                action_literal = exported_policy(flat_obs, step_idx)
                return action_literal


        except Exception as exc:
            # Generic fallback only if the LLM response fails.
            logger.error(f"Error deciding episode mode with LLM: {exc}")
            logger.error(traceback.format_exc())
            return None
        

        """
                    {{
              "name": "short free-text label",
              "description": "what this tests",
              "target_question": "Question that this experiment aims to answer",
              "custom_action_python_oneline_method": "Python lambda as action that will be eval() during steps in that experiment-episode. Only accept 'obs' and 'step_idx' as lambda param.",
            }}

            Canonical action literals:
            - Discrete: integer index.
            - Box: list of floats matching flattened action dimension.
            - MultiDiscrete: list of ints.
            - MultiBinary: list of 0/1 ints.

            Return following strict Format:
            {{
              "mode": "control" | "experiment",
              "reason": "...",
              "experiment_plan": null or {{...full plan object...}}
            }}"""


    def get_experiment_action(self, experiment: Experiment | None = None, step_idx: int = 0, flat_obs: List[float] = [] ) -> CanonicalAction:
        """
        DB column 'custom_action_python_oneline_method' is a python lambda that'll compute the actions for that experiment.
        """
        if experiment is None:
            raise ValueError("experiment is required.")
        try:
            if not experiment.custom_action_python_oneline_method or experiment.custom_action_python_oneline_method.strip() == "":
                raise ValueError("No custom action method provided in experiment plan")
            safe_globals = {"__builtins__": None}
            safe_locals = {"abs": abs, "pow": pow, "max": max, "min": min, "round": round, "math": __import__("math")}
            action = eval(experiment.custom_action_python_oneline_method, safe_globals, safe_locals)(flat_obs,step_idx)
            logger.debug(f"Evaluating custom action method for step {step_idx}: {experiment.custom_action_python_oneline_method} -> {action}")
            return self._canonicalize_action_literal(action)
        except Exception as exc:
            logger.warning(f"Error executing custom action method {experiment.custom_action_python_oneline_method}: {exc}")
            logger.error(traceback.format_exc())
            try:
                Experiment.objects.filter(session=self.session, name=experiment.name).update(status='failed')
                logger.info(f"Marked Experiment {experiment.name} as failed in DB")
            except Exception:
                logger.exception("Failed to mark experiment as failed via ORM")
            return self._canonicalize_action_literal(0)

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def act(self, flat_observation: List[float], step_idx: int) -> Tuple[CanonicalAction, Dict[str, Any]]:
        env_schema = self._get_env_schema()
        top_hypotheses = self.get_top_hypotheses(limit=self.top_k_hypotheses)
        good_outcomes = self.get_recent_good_outcome_patterns(limit=8)

        prompt = textwrap.dedent(
            f"""
            You are controlling an RL agent for ONE step in an unknown environment.
            The observation is already flattened. Do not assume semantics unless supported by evidence.

            Environment schema JSON:
            {json.dumps(env_schema, indent=2)}

            Flattened observation:
            {json.dumps([round(float(x), 6) for x in flat_observation])}

            Stored hypotheses:
            {json.dumps(top_hypotheses, indent=2)}

            Good-outcome patterns:
            {json.dumps(good_outcomes, indent=2)}

            Return strict JSON:
            {{
              "action": <canonical action literal>,
              "reason": "...",
              "referenced_hypotheses": ["..."]
            }}
            """
        )

        try:
            data = self._chat_json(prompt=prompt, purpose="step_action", episode_index=None)
            action = self._canonicalize_action_literal(data.get("action", self._neutral_action_literal()))
            meta = {
                "reason": str(data.get("reason", "")),
                "referenced_hypotheses": list(data.get("referenced_hypotheses", [])),
                "source": "llm",
            }
            self._last_control_reason = meta["reason"]
            logger.info(f"Chose action {action} with reason: {meta['reason']} and referenced hypotheses: {meta['referenced_hypotheses']}")
            return action, meta
        except Exception as exc:
            action = self._neutral_action_literal()
            meta = {
                "reason": f"neutral fallback after LLM error: {exc}",
                "referenced_hypotheses": [],
                "source": "fallback",
            }
            self._last_control_reason = meta["reason"]
            return action, meta

    # ------------------------------------------------------------------
    # Post-episode analysis
    # ------------------------------------------------------------------
    def post_episode(
        self,
        episode_index: int,
        trajectory: List[Dict[str, Any]],
        return_value: float,
        terminated: bool,
        truncated: bool,
        mode: str = "None",
        experiment_plan: Optional[ExperimentPlan] | None = None,
    ) -> Dict[str, Any]:
        logger.info(f"Post-episode processing start: episode_index={episode_index} mode={mode}")
        summary = self._summarize_trajectory(trajectory)
        ep = Episode.objects.create(
            session=self.session,
            episode_index=episode_index,
            mode=mode,
            return_value=float(return_value),
            length=len(trajectory),
            terminated=bool(terminated),
            truncated=bool(truncated),
            summary_json=json.dumps(summary),
        )
        logger.info(f"Saved Episode id={ep.id} episode_index={episode_index}")
        if experiment_plan is not None:
            exp = Experiment.objects.get_or_create(
                session=self.session,
                episode_index=episode_index,
                status="completed",
                name=experiment_plan.name,
                description=experiment_plan.description,
                target_question=experiment_plan.target_question,
                reason=experiment_plan.reason,
                custom_action_python_oneline_method=experiment_plan.custom_action_python_oneline_method,
                result_summary_json=json.dumps(summary),
            )
            logger.info(f"Saved Experiment id={exp[0].id} name={exp[0].name}")

        analysis = self._analyze_episode_with_llm(
            episode_index=episode_index,
            mode=mode,
            trajectory_summary=summary,
            experiment_plan=experiment_plan,
        )
        logger.info(f"Received analysis for episode {episode_index}, storing results")
        self._store_episode_analysis(episode_index=episode_index, analysis=analysis)
        if analysis.get("apply_current_episode_experiment_findings_to_control", False) and experiment_plan is not None:
            logger.info(f"Applying findings from episode {episode_index} experiment to programmatic python policy")
            apply_experiment_to_policy = self._apply_experiment_findings_to_policy(experiment_plan, analysis)
            Experiment.objects.filter(session=self.session, name=experiment_plan.name).update(status='implemented')
        raw_path = self.raw_dir / f"episode_{episode_index:05d}_summary.json"
        raw_path.write_text(json.dumps({"summary": summary, "analysis": analysis}, indent=2))
        logger.info(f"Wrote raw episode summary to {raw_path}")

        return {"summary": summary, "analysis": analysis}

    def _apply_experiment_findings_to_policy(self, experiment_plan: ExperimentPlan, analysis: Dict[str, Any], out_path: Optional[str] = None) -> bool:
        """"""
        env_schema = self._get_env_schema()
        top_hypotheses = self.get_top_hypotheses(limit=24)
        good_patterns = self.get_recent_good_outcome_patterns(limit=12)
        constants = self.get_constants(limit=24)
        interpretations = self.get_interpretations(limit=16)
        python_policy_text = (self.export_dir / "exported_programmatic_policy.py").read_text() if (self.export_dir / "exported_programmatic_policy.py").exists() else "# No exported policy yet"
        prompt = textwrap.dedent(
            f"""
            You are an expert RL scientist tasked with improving a programmatic policy based on new experiment findings.

            Environment schema:
            {json.dumps(env_schema, indent=2)}

            Learned hypotheses:
            {json.dumps(top_hypotheses, indent=2)}

            Learned constants:
            {json.dumps(constants, indent=2)}

            Learned interpretations:
            {json.dumps(interpretations, indent=2)}

            Good behavior summaries:
            {json.dumps(good_patterns, indent=2)}

            Experiment plan:
                {json.dumps({
                    "name": experiment_plan.name,
                    "description": experiment_plan.description,
                    "target_question": experiment_plan.target_question,
                    "custom_action_python_oneline_method": experiment_plan.custom_action_python_oneline_method,
                    "reason": experiment_plan.reason,
                }, indent=2)}

            Experiment analysis:
                {json.dumps(analysis, indent=2)}

            Most recent version of programmatic python policy:
            ```python
               {python_policy_text}
            ```
                
            - Apply the experiment findings to the programatic python policy. 
            - Only change parts of the policy that are relevant to the experiment findings.
            - Only add code that directly addresses the experiment's target question and reason.
            - Skip detailed comments and descriptions. Brief one-line truly helpful comments briefly describing what effect on the environment it'll have or what dynamics it expects to control.
            - It should be a surgical improvement to the existing policy, not a complete rewrite.
            - You can remove parts of policy code if it contradicts the experiements code and vice versa.
            - Compare the experiment's custom lambda method and insights with the existing policy code, and find a way to extend or update the existing policy with the experiment's findings. 
            - allowed import: math, numpy as np, pytorch as torch.
            - DO NOT use os, sys, subprocess, eval, exec, open, or any file/network operations.
            - Above all else execute a sanity check - if the previous experiment/episode performed badly - don't change anything and just return the previous version of the code.
            """
            )
        
        logger.info("Exporting programmatic policy via LLM")
        code = self._chat_text(prompt=prompt, purpose="export_programmatic_policy", episode_index=None)
        code = self._extract_python_code(code)
        if out_path is None:
            out_path = str(self.export_dir / "exported_programmatic_policy.py")
        Path(out_path).write_text(code)
        logger.info(f"Wrote exported policy to {out_path}")
        return True
        


    def export_programmatic_policy(self, out_path: Optional[str] = None) -> str:
        env_schema = self._get_env_schema()
        top_hypotheses = self.get_top_hypotheses(limit=24)
        good_patterns = self.get_recent_good_outcome_patterns(limit=12)
        constants = self.get_constants(limit=24)
        interpretations = self.get_interpretations(limit=16)

        prompt = textwrap.dedent(
            f"""
            Generate a deterministic Python programmatic policy from the learned session memory.

            Rules:
            - Output ONLY valid, compact Python code in a format that reduces LLM-related errors.
            - Provide: def act(obs, step_idx):
            - obs is a FLATTENED numeric observation vector.
            - return a canonical action literal matching the environment schema.
            - allowed import: math, numpy as np, pytorch as torch.
            - DO NOT use os, sys, subprocess, eval, exec, open, or any file/network operations.
            - DO NOT use assume any environment dynamics or constants based on your knowledge of known gym environments. Rely solely on the provided learned hypotheses, interpretations, constants, and good behavior patterns.

            Environment schema:
            {json.dumps(env_schema, indent=2)}

            Learned hypotheses:
            {json.dumps(top_hypotheses, indent=2)}

            Learned constants:
            {json.dumps(constants, indent=2)}

            Learned interpretations:
            {json.dumps(interpretations, indent=2)}

            Good behavior summaries:
            {json.dumps(good_patterns, indent=2)}
            """
        )
        logger.info("Exporting programmatic policy via LLM")
        code = self._chat_text(prompt=prompt, purpose="export_programmatic_policy", episode_index=None)
        code = self._extract_python_code(code)
        if out_path is None:
            out_path = str(self.export_dir / "exported_programmatic_policy.py")
        Path(out_path).write_text(code)
        logger.info(f"Wrote exported policy to {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Public getters
    # ------------------------------------------------------------------
    def get_last_control_reason(self) -> str:
        return self._last_control_reason

    def get_llm_call_count(self) -> int:
        return self._llm_calls

    def get_session_snapshot(self) -> Dict[str, Any]:
        episodes = Episode.objects.filter(session=self.session).order_by('-episode_index')[:12]
        hypotheses = Hypothesis.objects.filter(session=self.session).order_by('-confidence', '-id')[:12]
        experiments = Experiment.objects.filter(session=self.session).order_by('-id')[:8]

        recent_returns = [e.return_value for e in episodes]
        recent_experiments = [
            {
                "name": exp.name,
                "reason": exp.reason,
                "target_question": exp.target_question,
            }
            for exp in experiments
        ]
        return {
            "num_episodes": episodes.count(),
            "recent_returns": recent_returns,
            "recent_mean_return": float(np.mean(recent_returns)) if recent_returns else 0.0,
            "recent_modes": [e.mode for e in episodes],
            "top_hypotheses": [
                {
                    "name": h.name,
                    "kind": h.kind,
                    "confidence": h.confidence,
                    "status": h.status,
                    "statement": h.statement,
                }
                for h in hypotheses
            ],
            "recent_experiments": recent_experiments,
        }

    def get_top_hypotheses(self, limit: int = 10) -> List[Dict[str, Any]]:
        hypotheses = Hypothesis.objects.filter(session=self.session).order_by('-confidence', '-id')[:limit]
        return [
            {
                "name": h.name,
                "kind": h.kind,
                "statement": h.statement,
                "candidate_equation": h.candidate_equation,
                "confidence": h.confidence,
                "status": h.status,
                "evidence": h.evidence,
                "proposed_test": h.proposed_test,
            }
            for h in hypotheses
        ]

    def get_constants(self, limit: int = 16) -> List[Dict[str, Any]]:
        constants = Constant.objects.filter(session=self.session).order_by('-confidence', '-id')[:limit]
        return [
            {
                "name": c.name,
                "value": c.value,
                "confidence": c.confidence,
                "rationale": c.rationale,
            }
            for c in constants
        ]

    def get_interpretations(self, limit: int = 16) -> List[Dict[str, Any]]:
        interpretations = Interpretation.objects.filter(session=self.session).order_by('-confidence', '-id')[:limit]
        out = []
        for i in interpretations:
            out.append({
                "feature_index": i.feature_index,
                "meanings": json.loads(i.meanings_json) if i.meanings_json else [],
                "confidence": i.confidence,
                "reason": i.reason,
            })
        return out

    def get_recent_good_outcome_patterns(self, limit: int = 8) -> Dict[str, Any]:
        episodes = Episode.objects.filter(session=self.session).order_by('-return_value', '-episode_index')[:limit]
        items = []
        for e in episodes:
            try:
                s = json.loads(e.summary_json)
                s["return_value"] = float(e.return_value)
                items.append(s)
            except Exception:
                pass
        if not items:
            return {"num_good_episodes": 0, "pattern": "none yet"}

        final_obs = [x.get("final_observation_flat", []) for x in items if x.get("final_observation_flat")]
        lengths = [int(x.get("length", 0)) for x in items]
        out: Dict[str, Any] = {
            "num_good_episodes": len(items),
            "top_returns": [float(x.get("return_value", 0.0)) for x in items[: min(len(items), 8)]],
            "median_length": float(np.median(lengths)) if lengths else 0.0,
        }
        if final_obs:
            arr = np.asarray(final_obs, dtype=float)
            out["avg_final_observation_flat"] = arr.mean(axis=0).round(3).tolist()
        action_space_type = self._get_env_schema().get("action_space", {}).get("space_type")
        if action_space_type == "Discrete":
            hists = [x.get("action_histogram") for x in items if x.get("action_histogram")]
            if hists:
                out["avg_action_histogram"] = np.asarray(hists, dtype=float).mean(axis=0).round(3).tolist()
        else:
            means = [x.get("action_mean_flat") for x in items if x.get("action_mean_flat")]
            if means:
                out["avg_action_mean_flat"] = np.asarray(means, dtype=float).mean(axis=0).round(3).tolist()
        return out

    # ------------------------------------------------------------------
    # Summaries and analysis
    # ------------------------------------------------------------------
    def _summarize_trajectory(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trajectory:
            return {
                "length": 0,
                "obs_mean_flat": [],
                "obs_std_flat": [],
                "delta_mean_flat": [],
                "delta_std_flat": [],
                "final_observation_flat": [],
                "return_value": 0.0,
            }

        obs = np.asarray([step["obs_flat"] for step in trajectory], dtype=float)
        next_obs = np.asarray([step["next_obs_flat"] for step in trajectory], dtype=float)
        deltas = next_obs - obs
        rewards = np.asarray([step["reward"] for step in trajectory], dtype=float)
        action_space_type = self._get_env_schema().get("action_space", {}).get("space_type")

        out: Dict[str, Any] = {
            "length": len(trajectory),
            "reward_sum": float(rewards.sum()),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "obs_mean_flat": obs.mean(axis=0).round(3).tolist(),
            "obs_std_flat": obs.std(axis=0).round(3).tolist(),
            "delta_mean_flat": deltas.mean(axis=0).round(3).tolist(),
            "delta_std_flat": deltas.std(axis=0).round(3).tolist(),
            "first_observation_flat": obs[0].round(3).tolist(),
            "final_observation_flat": next_obs[-1].round(3).tolist(),
            "sample_trace": [
                {
                    "t": int(i),
                    "obs_flat": obs[i].round(3).tolist(),
                    "action": trajectory[int(i)]["action_json"],
                    "reward": round(float(rewards[i]), 3),
                    "next_obs_flat": next_obs[i].round(3).tolist(),
                }
                for i in np.linspace(0, len(trajectory) - 1, num=min(10, len(trajectory)), dtype=int)
            ],
        }

        if action_space_type == "Discrete":
            actions = np.asarray([int(step["action_json"]) for step in trajectory], dtype=int)
            n = int(self._get_env_schema().get("action_space", {}).get("n", max(actions.max() + 1, 1)))
            hist = np.bincount(actions, minlength=n).astype(int)
            out["action_histogram"] = hist.tolist()
            action_conditioned: Dict[str, Dict[str, Any]] = {}
            for action_idx in range(n):
                mask = actions == action_idx
                if np.any(mask):
                    action_conditioned[str(action_idx)] = {
                        "count": int(mask.sum()),
                        "delta_mean_flat": deltas[mask].mean(axis=0).round(3).tolist(),
                        "delta_std_flat": deltas[mask].std(axis=0).round(3).tolist(),
                    }
                else:
                    action_conditioned[str(action_idx)] = {
                        "count": 0,
                        "delta_mean_flat": [0.0] * obs.shape[1],
                        "delta_std_flat": [0.0] * obs.shape[1],
                    }
            out["action_conditioned_deltas"] = action_conditioned
        else:
            a = np.asarray([self._action_to_flat(step["action_json"]) for step in trajectory], dtype=float)
            out["action_mean_flat"] = a.mean(axis=0).round(3).tolist()
            out["action_std_flat"] = a.std(axis=0).round(3).tolist()
            corr = []
            if a.size > 0:
                a_center = a - a.mean(axis=0, keepdims=True)
                d_center = deltas - deltas.mean(axis=0, keepdims=True)
                denom_a = np.maximum(a_center.std(axis=0, keepdims=True), 1e-8)
                denom_d = np.maximum(d_center.std(axis=0, keepdims=True), 1e-8)
                corr_mat = (a_center / denom_a).T @ (d_center / denom_d) / max(len(a), 1)
                corr = np.asarray(corr_mat).round(3).tolist()
            out["action_delta_correlation"] = corr

        return out

    def _analyze_episode_with_llm(
        self,
        episode_index: int,
        mode: str,
        trajectory_summary: Dict[str, Any],
        experiment_plan: Optional[ExperimentPlan],
    ) -> Dict[str, Any]:
        env_schema = self._get_env_schema()
        session_snapshot = self.get_session_snapshot()
        prompt = textwrap.dedent(
            f"""
            You are an RL scientist.
            Infer environment dynamics, constants, interpretations, and follow-up tests from trajectory evidence.

            Critical rules:
            - Infer by intervention and observation only.
            - Do NOT assume that any feature index has known semantics.
            - A sustained drift in one or more flattened features under a controlled custom_action_python_oneline_method may indicate a latent force, bias, decay, accumulation, or hidden variable influence.
            - Distinguish raw observation from interpretation.
            - Propose the next experiment as a specific python lambda function to test a hypothesis, that will help understand the environment. 


            Environment schema JSON:
            {json.dumps(env_schema, indent=2)}

            Episode mode: {mode}
            Experiment plan JSON:
            {json.dumps(self._plan_to_dict(experiment_plan) if experiment_plan else None, indent=2)}

            Trajectory summary JSON:
            {json.dumps(trajectory_summary, indent=2)}

            Existing memory snapshot JSON:
            {json.dumps(session_snapshot, indent=2)}

            Only return new Entries. Only constants can be updated.
            "apply_current_episode_experiment_findings_to_control" is the most crucial part, only est this to "True" if the current episode/experiment
            improved the mean/avg/total reward compared to previous episodes. 
            Return strict JSON with keys:
            {{
              "apply_current_episode_experiment_findings_to_control": "True" or "False",
              "hypotheses": [
                {{
                  "name": "...",
                  "kind": "latent_cause|constant|dynamics|reward|interpretation|edge_case",
                  "statement": "...",
                  "candidate_equation": "optional equation template",
                  "variables": ["flat_obs[0]", "action", "..."],
                  "confidence": 0.0,
                  "status": "proposed|supported|rejected",
                  "evidence": "...",
                  "proposed_test": "..."
                }}
              ],
              "interpretations": [
                {{
                  "feature_index": 0,
                  "possible_meanings": ["...", "..."],
                  "confidence": 0.0,
                  "reason": "..."
                }}
              ],
              "constants": [
                {{
                  "name": "...",
                  "value": "...",
                  "confidence": 0.0,
                  "rationale": "..."
                }}
              ],
              "control_advice": "one short paragraph",
              "next_experiment_plan": null or {{
                "name": "...",
                "description": "...",
                "target_question": "...",
                "custom_action_python_oneline_method": "experiment_name_xxx = lambda obs, step_idx: <canonical action literal>. Accept ONLY 'obs' & 'step_idx' as lambda param."
              }}
            }}
            """
        )
        return self._chat_json(prompt=prompt, purpose="post_episode_analysis", episode_index=episode_index)

    def _store_episode_analysis(self, episode_index: int, analysis: Dict[str, Any]) -> None:
        for h in analysis.get("hypotheses", []):
            Hypothesis.objects.create(
                session=self.session,
                episode_index=episode_index,
                name=str(h.get("name", ""))[:300],
                kind=str(h.get("kind", ""))[:80],
                statement=str(h.get("statement", "")),
                candidate_equation=str(h.get("candidate_equation", "")),
                variables_json=json.dumps(h.get("variables", [])),
                confidence=float(h.get("confidence", 0.0) or 0.0),
                status=str(h.get("status", "proposed"))[:80],
                evidence=str(h.get("evidence", "")),
                proposed_test=str(h.get("proposed_test", "")),
            )
            logger.debug(f"Stored hypothesis: {h.get('name','')[:80]}")
        for item in analysis.get("interpretations", []):
            Interpretation.objects.create(
                session=self.session,
                episode_index=episode_index,
                feature_index=int(item.get("feature_index", -1)),
                meanings_json=json.dumps(item.get("possible_meanings", [])),
                confidence=float(item.get("confidence", 0.0) or 0.0),
                reason=str(item.get("reason", "")),
            )
            logger.debug(f"Stored interpretation for feature_index={item.get('feature_index')}")
        for c in analysis.get("constants", []):
            Constant.objects.create(
                session=self.session,
                episode_index=episode_index,
                name=str(c.get("name", ""))[:200],
                value=str(c.get("value", ""))[:200],
                confidence=float(c.get("confidence", 0.0) or 0.0),
                rationale=str(c.get("rationale", "")),
            )
            logger.debug(f"Stored constant: {c.get('name','')}")

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------
    def _chat_text(self, prompt: str, purpose: str, episode_index: Optional[int]) -> str:
        logger.info(f"LLM call: purpose={purpose} episode_index={episode_index}")

        try:
            response = openai.responses.create(
                model=self.model,
                instructions="You are a careful RL research assistant. Prefer compact, structured outputs.",
                input=prompt,
                temperature=0.2,
                timeout=self.request_timeout_s,
            )
            self._llm_calls += 1
            content = ""
            if getattr(response, 'output_text', None):
                content = response.output_text
            logger.debug(f"LLM response length: {len(content)}")
        except Exception as exc:
            logger.error(f"LLM request failed for purpose={purpose}: {exc}")
            logger.error(traceback.format_exc())
            content = ""
        # audit and record the full call
        self._audit_llm(episode_index=episode_index, purpose=purpose, prompt=prompt, response=content)
        return content

    def _chat_json(self, prompt: str, purpose: str, episode_index: Optional[int]) -> Dict[str, Any]:
        text = self._chat_text(prompt=prompt, purpose=purpose, episode_index=episode_index)
        try:
            return self._extract_json(text)
        except Exception:
            repair_prompt = textwrap.dedent(
                f"""
                Convert the following into strict JSON only. No markdown. No commentary.
                Content:
                {text}
                """
            )
            repaired = self._chat_text(prompt=repair_prompt, purpose=f"{purpose}_json_repair", episode_index=episode_index)
            return self._extract_json(repaired)

    def _audit_llm(self, episode_index: Optional[int], purpose: str, prompt: str, response: str) -> None:
        try:
            LLMAudit.objects.create(
                session=self.session,
                episode_index=episode_index,
                purpose=purpose,
                prompt_preview=prompt[:2000],
                response_preview=response[:2000],
            )
        except Exception:
            logger.exception("Failed to write LLMAudit record")
        try:
            LLMCall.objects.create(
                session=self.session,
                episode_index=episode_index,
                purpose=purpose,
                prompt=prompt,
                response=response,
                model=self.model,
            )
        except Exception:
            logger.exception("Failed to write LLMCall record")

    # ------------------------------------------------------------------
    # Parsing / validation helpers
    # ------------------------------------------------------------------
    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        fence = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.DOTALL)
        if fence:
            text = fence.group(1)
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
            return {"data": obj}
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def _extract_python_code(self, text: str) -> str:
        text = text.strip()
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        code = match.group(1) if match else text
        ast.parse(code)
        return code + ("\n" if not code.endswith("\n") else "")

    def _plan_to_dict(self, plan: Optional[ExperimentPlan]) -> Optional[Dict[str, Any]]:
        if plan is None:
            return None
        return {
            "name": plan.name,
            "description": plan.description,
            "target_question": plan.target_question,
            "custom_action_python_oneline_method": plan.custom_action_python_oneline_method,
            "reason": plan.reason,
        }
        
    def _canonicalize_action_literal(self, value: Any) -> CanonicalAction:
        action_spec = self._get_env_schema().get("action_space", {})
        space_type = action_spec.get("space_type")
        if space_type == "Discrete":
            n = int(action_spec.get("n", 1))
            try:
                idx = int(value)
            except Exception:
                idx = 0
            return int(min(max(idx, 0), max(n - 1, 0)))
        if space_type == "Box":
            dim = int(action_spec.get("flat_dim", 1))
            low = np.asarray(action_spec.get("low_flat", [-1.0] * dim), dtype=float)
            high = np.asarray(action_spec.get("high_flat", [1.0] * dim), dtype=float)
            arr = np.asarray(value if isinstance(value, list) else [value] * dim, dtype=float).reshape(-1)
            if arr.size < dim:
                arr = np.pad(arr, (0, dim - arr.size), mode="constant")
            arr = arr[:dim]
            arr = np.clip(arr, low, high)
            return arr.astype(float).tolist()
        if space_type == "MultiDiscrete":
            nvec = np.asarray(action_spec.get("nvec", [1]), dtype=int)
            arr = np.asarray(value if isinstance(value, list) else [value] * len(nvec), dtype=int).reshape(-1)
            if arr.size < len(nvec):
                arr = np.pad(arr, (0, len(nvec) - arr.size), mode="constant")
            arr = arr[: len(nvec)]
            arr = np.minimum(np.maximum(arr, 0), np.maximum(nvec - 1, 0))
            return arr.astype(int).tolist()
        if space_type == "MultiBinary":
            n = int(action_spec.get("n", 1))
            arr = np.asarray(value if isinstance(value, list) else [value] * n, dtype=int).reshape(-1)
            if arr.size < n:
                arr = np.pad(arr, (0, n - arr.size), mode="constant")
            arr = arr[:n]
            arr = (arr > 0).astype(int)
            return arr.tolist()
        try:
            return int(value)
        except Exception:
            return 0

    def _action_to_flat(self, action_json: CanonicalAction) -> List[float]:
        if isinstance(action_json, list):
            return [float(x) for x in action_json]
        return [float(action_json)]

    def _neutral_action_literal(self) -> CanonicalAction:
        action_spec = self._get_env_schema().get("action_space", {})
        space_type = action_spec.get("space_type")
        if space_type == "Discrete":
            return 0
        if space_type == "Box":
            low = np.asarray(action_spec.get("low_flat", [0.0]), dtype=float)
            high = np.asarray(action_spec.get("high_flat", [0.0]), dtype=float)
            finite = np.isfinite(low) & np.isfinite(high)
            mid = np.where(finite, 0.5 * (low + high), 0.0)
            return mid.astype(float).tolist()
        if space_type == "MultiDiscrete":
            nvec = np.asarray(action_spec.get("nvec", [1]), dtype=int)
            return np.zeros_like(nvec).astype(int).tolist()
        if space_type == "MultiBinary":
            n = int(action_spec.get("n", 1))
            return [0] * n
        return 0

    def _generic_bootstrap_plan(self) -> bool:
        action_spec = self._get_env_schema().get("action_space", {})
        space_type = action_spec.get("space_type")
        neutral = self._neutral_action_literal()
        Hypothesis.objects.create(
            session=self.session,
            episode_index=0,
            name="bootstrap_constant_neutral",
            kind="dynamics",
            statement="Applying a constant neutral action will reveal which features drift naturally versus which respond to intervention.",
            candidate_equation=f"lambda flat_obs,step_idx: return {neutral}",
            variables_json=json.dumps(["flat_obs[i]", "step_idx"]),
            confidence=0.5,
            status="proposed",
            evidence="none yet",
            proposed_test="Apply the same constant neutral action across the episode and observe feature drifts.",
        )

        Experiment.objects.create(
            session=self.session,
            name="bootstrap_constant_neutral",
            description="Apply a neutral, minimal or no action across the episode to identify passive drift and natural termination behavior.",
            target_question="Which flattened features drift without active intervention?",
            custom_action_python_oneline_method=f"lambda obs, step_idx: return {neutral}",
            reason="generic fallback bootstrap",
            episode_index=0
        )
        return True

        
