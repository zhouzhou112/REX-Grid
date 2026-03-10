import pandas as pd
import numpy as np
from pathlib import Path


def read_decision_points(file_path: str | Path) -> pd.DataFrame:
    """
    从指定 Excel 或 CSV 文件中读取并返回 decision_points DataFrame
    """
    path = Path(file_path)
    if path.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # 确保 point_id 为字符串格式，防止后续关联出错
    if 'point_id' in df.columns:
        df['point_id'] = df['point_id'].astype(str)

    return df


def apply_actions_to_decision_points(df: pd.DataFrame, action_array: np.ndarray) -> pd.DataFrame:
    """
    (向量化极速版) 将强化学习动作(陆上风电、海上风电、光伏)更新到 df 中。

    参数:
      df: decision_points DataFrame，每行对应一个决策区域
      action_array: numpy数组，形状 (n_regions, 3)
                    表示 [onshore_add, offshore_add, pv_add] (MW)
    返回:
      更新后的 DataFrame (新对象，不污染原 df)
    """
    n_regions = df.shape[0]
    if action_array.shape != (n_regions, 3):
        raise ValueError(f"动作维度不匹配: df有{n_regions}行，action_array形状{action_array.shape}")

    # 复制一份，防止原地修改导致不可预期的 Bug
    new_df = df.copy()

    # 1. 提取当前安装量和潜力上限
    inst_cols = ['Installed_onshorewind', 'Installed_offshorewind', 'Installed_pv']
    pot_cols = ['Potential_onshorewind', 'Potential_offshorewind', 'Potential_pv']

    current_installed = new_df[inst_cols].values
    potential = new_df[pot_cols].values

    # 2. 计算剩余可用潜力
    remain_potential = np.clip(potential - current_installed, 0.0, None)

    # 3. 限制实际扩建量不超过剩余潜力
    real_add = np.minimum(action_array, remain_potential)

    # 4. 更新安装量
    new_installed = current_installed + real_add
    new_df[inst_cols] = new_installed

    return new_df


# ==========================================
# 简单测试入口 (开源时方便别人验证)
# ==========================================
if __name__ == '__main__':
    # 示例用法
    # 请确保路径与你 config 中的一致
    excel_file = r"D:\paper2\data\resourcepoint\05已安装风光数据链接\RL_STATE_2020.xlsx"

    try:
        df = read_decision_points(excel_file)
        n_regions = df.shape[0]

        # 模拟一个随机动作 (每个点扩建 0~100 MW)
        mock_action = np.random.uniform(0, 100, size=(n_regions, 3))

        updated_df = apply_actions_to_decision_points(df, mock_action)

        print("原始数据 (前5行):")
        print(df[['Installed_onshorewind', 'Installed_offshorewind', 'Installed_pv']].head())
        print("\n更新后数据 (前5行):")
        print(updated_df[['Installed_onshorewind', 'Installed_offshorewind', 'Installed_pv']].head())

    except FileNotFoundError:
        print(f"找不到测试文件: {excel_file}，请修改路径后再试。")