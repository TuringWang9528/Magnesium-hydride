import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型与数据
model = joblib.load('CatBoost.pkl')
df = pd.read_excel('单位产氢量.xlsx', engine='openpyxl')

# 目标列与候选特征列
target_col = 'Hydrogen production(mL/g)'
candidate_features = [c for c in df.columns if c != target_col]

# 若模型保存了训练时的特征名，优先使用以确保顺序一致
model_feature_names = getattr(model, 'feature_names_', None)
features = list(model_feature_names) if model_feature_names else candidate_features

# 将这两列作为“离散变量”处理（用 selectbox 展示数据集中出现过的取值）
discrete_cols = [c for c in ['Ball milling', 'Non ball milling'] if c in features]
numeric_cols = [c for c in features if c not in discrete_cols]

st.title("氢气生产预测平台")

inputs = {}

# 数值特征：number_input
for col in numeric_cols:
    col_min = float(df[col].min())
    col_max = float(df[col].max())
    default = float(df[col].mean())
    rng = col_max - col_min
    step = 0.001 if rng < 1 else (0.01 if rng < 10 else 0.1)

    inputs[col] = st.number_input(
        col,
        min_value=col_min,
        max_value=col_max,
        value=default,
        step=step,
        format="%.6f"
    )

# 离散特征：selectbox（选项为数据集中出现过的离散取值）
for col in discrete_cols:
    # 取去重后的离散取值，并排序；尽量保持原始类型
    options = pd.Series(df[col].dropna().unique()).tolist()
    try:
        options = sorted(options)
    except Exception:
        # 若包含混合类型，避免排序报错
        pass

    # 默认值用众数（若拿不到就用第一个）
    default_val = df[col].mode().iloc[0] if not df[col].mode().empty else options[0]
    default_idx = options.index(default_val) if default_val in options else 0

    inputs[col] = st.selectbox(col, options=options, index=default_idx)

# 组装成与训练时相同顺序的 DataFrame
X_input_df = pd.DataFrame([[inputs[c] for c in features]], columns=features)

# 基本类型保护：数值列转为数值
for c in numeric_cols:
    X_input_df[c] = pd.to_numeric(X_input_df[c], errors='coerce')

if st.button("预测"):
    # （可选）一致性校验：特征数量与模型一致
    get_cnt = getattr(model, 'get_feature_count', None)
    expected = int(get_cnt()) if callable(get_cnt) else len(features)
    provided = X_input_df.shape[1]
    if provided != expected:
        st.error(
            f"❌ 特征数量不一致：模型期望 {expected} 个，当前提供 {provided} 个。\n"
            f"请核对特征顺序/名称：{features}"
        )
    else:
        # 预测 + 保留四位小数
        y_pred = float(model.predict(X_input_df)[0])
        st.write(f"**预测的氢气生产量 (mL/g):** {y_pred:.4f}")

        # SHAP 可视化（可选）
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input_df)
            if isinstance(shap_values, list):  # 兼容分类模型返回 list 的情况
                shap_values = shap_values[0]
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                X_input_df.iloc[0],
                matplotlib=True
            )
            plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
            st.image("shap_force_plot.png")
        except Exception as e:
            st.warning(f"SHAP 可视化未成功：{e}")
