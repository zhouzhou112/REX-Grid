import os
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


def main():
    # 文件路径 (只保留装机潜能的数据)
    decision_points_excel = r"D:\paper2\data\resourcepoint\05已安装风光数据链接\RL_STATE_2020.xlsx"
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    if os.name == "nt":  # Windows 必须 spawn
        mp.set_start_method("spawn", force=True)

    # 1) 训练用并行环境 (32 进程)
    env_fns = [
        make_dispatch_env(
            province_id=44,
            rank=i,
            decision_points_excel=decision_points_excel,
            start_year=2020,
            terminal_year=2050,
            log_dir=log_dir,
        )
        for i in range(32)
    ]
    venv = SubprocVecEnv(env_fns, start_method="spawn")

    # 注意：如果是首次训练，请使用 VecNormalize(venv, ...)，如果有预训练权重，用 .load
    try:
        venv = VecNormalize.load("logs/vecnorm-ppo-0528.pkl", venv)
        print("已加载历史 VecNormalize 模型。")
    except FileNotFoundError:
        print("未找到历史 VecNormalize，初始化全新归一化层。")
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2) 评估用串行环境
    eval_env_raw = make_dispatch_env(
        44, 999, decision_points_excel, 2020, 2050, log_dir
    )()
    eval_env = DummyVecEnv([lambda: eval_env_raw])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=10.)
    eval_env.obs_rms = venv.obs_rms
    eval_env.ret_rms = venv.ret_rms

    # 3) 回调函数配置
    custom_callback = CustomLoggingCallback(verbose=1)
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=log_dir, log_path=log_dir,
        eval_freq=5000, n_eval_episodes=1, deterministic=True
    )
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="ppo_GD")
    vec_cb = SaveVecNormalizeCallback(venv, save_freq=10000, save_path=log_dir)

    # 4) 模型加载与训练
    try:
        model = PPO.load(
            "./logs/ppo_GD_60_800000_steps.zip",
            env=venv,
            device="cuda",
            custom_objects={
                "learning_rate": linear_schedule(2e-5, 1e-6),
                "clip_range": linear_schedule(0.06, 0.02),
                "ent_coef": 0.001,
                "target_kl": 0.03,
                "n_steps": 64,  # 注意此处修改：之前参数里写了32，custom_objects里写了64，我给你统合到了这里
                "vf_coef": 0.4,
            },
            print_system_info=True,
        )
        print("已加载历史 PPO 模型进行继续训练。")
    except FileNotFoundError:
        print("未找到历史 PPO 模型，初始化全新 PPO 实例。")
        model = PPO(
            "MlpPolicy", venv, n_steps=64, batch_size=128,
            learning_rate=linear_schedule(2e-5, 1e-6), clip_range=linear_schedule(0.06, 0.02),
            ent_coef=0.001, target_kl=0.03, vf_coef=0.4, device="cuda", verbose=1
        )

    # 训练执行
    total_timesteps = 2_500_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[custom_callback, eval_callback, checkpoint_callback, vec_cb],
        log_interval=1,
        reset_num_timesteps=False,
        tb_log_name="PPO_GD-Run"
    )

    model.save("./model/ppo_dispatch_bridge_final")
    venv.save("logs/vecnorm-ppo-latest.pkl")
    print("PPO 训练圆满结束。")


if __name__ == "__main__":
    main()