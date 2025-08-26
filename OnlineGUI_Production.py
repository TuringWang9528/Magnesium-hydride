import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('CatBoost.pkl')

data = pd.read_excel('单位产氢量.xlsx')

# 特征列和目标变量
features_columns = ['Reaction time(min)', 'Mg(g)', 'Ce(g)', 'Ca(g)', 'ln(g)', 'Fe@C(g)', 'Ball milling', 'Non ball milling', 'Hydrogen content(g)']
target_column = 'Hydrogen production(mL/g)'

# Streamlit 用户界面
st.title("Magnesium hydride hydrogen production prediction platform")

# 二分类与数值列拆分
binary_cols = ['Ball milling', 'Non ball milling']
numeric_cols = [c for c in features_columns if c not in binary_cols]

# 为每个特征创建输入字段
inputs = {}
# 数值特征：使用 number_input，并用数据集动态设定范围与默认值
for col in numeric_cols:
    col_min = float(data[col].min())
    col_max = float(data[col].max())
    default = float(data[col].mean())
    # 步长根据数值范围简单给一个合理的估计
    step = 0.001 if (col_max - col_min) < 1 else (0.01 if (col_max - col_min) < 10 else 0.1)
    inputs[col] = st.number_input(col, min_value=col_min, max_value=col_max, value=default, step=step)

# 二分类特征：使用 radio，选项固定为 0/1（并以众数作为默认值）
for col in binary_cols:
    default_bin = int(data[col].mode().iloc[0]) if not data[col].mode().empty else 0
    inputs[col] = st.radio(col, [0, 1], index=0 if default_bin == 0 else 1, horizontal=True)

# 组装输入为 DataFrame（严格按 features_columns 的列顺序）
X_input_list = [inputs[c] for c in features_columns]
X_input = np.array([X_input_list])
X_input_df = pd.DataFrame(X_input, columns=features_columns)

# 当点击“预测”按钮时
if st.button("Predict"):
    # 使用模型进行预测
    predicted_production = float(model.predict(input_values)[0])
    
    # 显示预测结果
    st.write(f"**Predicted hydrogen production volume (mL/g):** {predicted_production:.4f}")
    
    # 计算SHAP值并显示force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([inputs], columns=features_columns))
    
    # 回归模型通常返回 ndarray；若为分类模型，可能返回 list，这里做个兼容处理
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_input_df.iloc[0],
        matplotlib=True
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
