import os
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "REX"))
from config.paths import input_path
from torch import nn
import multiprocessing as mp
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

# 导入构建好的 RL 环境
from WindSolarExpansionEnv import DispatchBridgeEnv

# CPU 线程绑定优化
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def make_dispatch_env(province_id: int, rank: int, decision_points_excel: str, start_year: int, terminal_year: int,
                      log_dir: str):
    def _init():
        # 这里已经去掉了废弃的调度相关参数
        env = DispatchBridgeEnv(
            province_num=province_id,
            decision_points_excel=decision_points_excel,
            start_year=start_year,
            terminal_year=terminal_year,
        )
        env = Monitor(env, os.path.join(log_dir, f"proc{rank}"))
        env.reset(seed=1000 + rank)
        return env

    return _init


def linear_schedule(initial_value, final_value=0.0):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value + (1 - progress_remaining) * final_value

    return func


class CustomLoggingCallback(BaseCallback):
    def __init__(self, dump_freq: int = 8, verbose=0):
        super().__init__(verbose)
        self.dump_freq = dump_freq

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            info = infos[0]
            self.logger.record("env/year", info.get("year", np.nan))
            self.logger.record("env/year_add", info.get("year_add", np.nan))
            self.logger.record("env/remaining_budget", info.get("remaining_budget", np.nan))
            self.logger.record("reward/Reward", info.get("comp_reward", 0))
            self.logger.record("reward/r1", info.get("R1", 0))
            self.logger.record("reward/r2", info.get("R2", 0))
            self.logger.record("reward/r3", info.get("R3", 0))
            self.logger.record("reward/r4", info.get("R4", 0))

            if self.num_timesteps % self.dump_freq == 0:
                self.logger.dump(self.num_timesteps)
        return True


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, venv, save_freq: int, save_path: str):
        super().__init__()
        self.venv = venv
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.venv.save(os.path.join(self.save_path, "vecnorm-ppo-latest.pkl"))
        return True


class EntropyScheduleCallback(BaseCallback):
    """Scale the entropy coefficient with the configured learning rate."""
    def __init__(self, initial_entropy, initial_lr):
        super().__init__()
        self.initial_entropy, self.initial_lr = initial_entropy, initial_lr

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        progress = self.model._current_progress_remaining
        self.model.ent_coef = self.initial_entropy * self.model.lr_schedule(progress) / self.initial_lr


def main():
    parser = argparse.ArgumentParser(description="Train a provincial PPO agent using SI Table S23.")
    parser.add_argument("--province", type=int, choices=[44, 45, 46, 52, 53], default=44)
    parser.add_argument("--config", type=Path, default=ROOT / "REX/configs/ppo_table_S23.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--check-config", action="store_true", help="Validate configuration without training or loading weather.")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    assert cfg["n_envs"] * cfg["n_steps"] % cfg["batch_size"] == 0
    if args.check_config:
        print(json.dumps(cfg, indent=2))
        return
    log_dir = input_path("output_dir") / "training" / str(args.province)
    log_dir.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        mp.set_start_method("spawn", force=True)
    env_fns = [make_dispatch_env(args.province, i, str(input_path("decision_points")),
                                cfg["start_year"], cfg["terminal_year"], str(log_dir))
               for i in range(cfg["n_envs"])]
    venv = VecNormalize(SubprocVecEnv(env_fns, start_method="spawn"), norm_obs=True, norm_reward=True, clip_obs=10.)
    eval_raw = make_dispatch_env(args.province, 999, str(input_path("decision_points")),
                                 cfg["start_year"], cfg["terminal_year"], str(log_dir))()
    eval_env = VecNormalize(DummyVecEnv([lambda: eval_raw]), training=False,
                            norm_obs=True, norm_reward=True, clip_obs=10.)
    eval_env.obs_rms, eval_env.ret_rms = venv.obs_rms, venv.ret_rms
    try:
        model = PPO("MlpPolicy", venv, n_steps=cfg["n_steps"], batch_size=cfg["batch_size"],
                    n_epochs=cfg["n_epochs"], gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"],
                    learning_rate=linear_schedule(*cfg["learning_rate"]),
                    clip_range=linear_schedule(*cfg["clip_range"]),
                    ent_coef=cfg["ent_coef"], target_kl=cfg["target_kl"], vf_coef=cfg["vf_coef"],
                    policy_kwargs={"activation_fn": nn.ReLU,
                                   "net_arch": {"pi": cfg["policy_layers"], "vf": cfg["value_layers"]}},
                    device=args.device, verbose=1, tensorboard_log=str(log_dir))
        callbacks = [CustomLoggingCallback(verbose=1),
                     EvalCallback(eval_env, best_model_save_path=str(log_dir), log_path=str(log_dir),
                                  eval_freq=5000, n_eval_episodes=1, deterministic=True),
                     CheckpointCallback(save_freq=10000, save_path=str(log_dir), name_prefix="ppo"),
                     SaveVecNormalizeCallback(venv, save_freq=10000, save_path=str(log_dir)),
                     EntropyScheduleCallback(cfg["ent_coef"], cfg["learning_rate"][0])]
        model.learn(total_timesteps=cfg["total_timesteps"], callback=callbacks, log_interval=1)
        model.save(str(log_dir / "ppo_final"))
        venv.save(str(log_dir / "vecnormalize_final.pkl"))
    finally:
        eval_env.close()
        venv.close()


if __name__ == "__main__":
    main()
