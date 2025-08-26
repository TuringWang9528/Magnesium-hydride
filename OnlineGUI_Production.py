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

# 为每个特征创建输入字段
inputs = {}
for column in features_columns:
    inputs[column] = st.number_input(column, min_value=float(data[column].min()), max_value=float(data[column].max()), value=float(data[column].mean()))

# 将输入值转换为numpy数组
input_values = np.array([list(inputs.values())])

# 当点击“预测”按钮时
if st.button("Predict"):
    # 使用模型进行预测
    predicted_production = float(model.predict(input_values)[0])
    
    # 显示预测结果
    st.write(f"**Predicted hydrogen production volume (mL/g):** {predicted_production:.4f}")
    
    # 计算SHAP值并显示force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([inputs], columns=features_columns))
    
    # 绘制SHAP值的force plot
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([inputs], columns=features_columns), matplotlib=True)
    
    # 将绘制的图保存为PNG文件，并在Streamlit中显示
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
