import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------------
# 1. 全局页面配置 (必须放在第一行)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MgH2 Prediction Platform",
    page_icon="🧪",
    layout="wide",  # 使用宽屏模式，利用屏幕空间
    initial_sidebar_state="expanded"
)

# 自定义简单的 CSS 来美化标题
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #2E86C1;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #566573;
        border-bottom: 2px solid #F2F3F4;
        padding-bottom: 10px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 缓存加载函数 (提高性能)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None
    return pd.read_excel(path, engine='openpyxl')

# -----------------------------------------------------------------------------
# 3. 页面 1: 产氢量预测 (Hydrogen Production)
# -----------------------------------------------------------------------------
def page_hydrogen_production():
    st.markdown('<p class="main-title">🧪 Hydrogen Production Prediction (mL/g)</p>', unsafe_allow_html=True)
    st.caption("Model: **CatBoost** | Target: **Hydrogen production(mL/g)**")

    # 加载文件
    model_path = 'CatBoost.pkl'
    data_path = '单位产氢量.xlsx'
    
    model = load_model(model_path)
    df = load_data(data_path)

    if model is None or df is None:
        st.error(f"❌ 缺少文件: 请确保目录下存在 `{model_path}` 和 `{data_path}`")
        return

    # 特征处理逻辑
    target_col = 'Hydrogen production(mL/g)'
    candidate_features = [c for c in df.columns if c != target_col]
    model_feature_names = getattr(model, 'feature_names_', None)
    features = list(model_feature_names) if model_feature_names else candidate_features

    discrete_cols = [c for c in ['Ball milling', 'Non ball milling'] if c in features]
    numeric_cols = [c for c in features if c not in discrete_cols]

    # --- 输入区域 (使用 Expander 和 Columns 美化) ---
    st.markdown('<p class="sub-header">1. Configure Parameters</p>', unsafe_allow_html=True)
    
    inputs = {}
    
    # 离散特征区域
    if discrete_cols:
        cols = st.columns(len(discrete_cols))
        for idx, col in enumerate(discrete_cols):
            options = pd.Series(df[col].dropna().unique()).tolist()
            try:
                options = sorted(options)
            except:
                pass
            # 默认值
            default_val = df[col].mode().iloc[0] if not df[col].mode().empty else options[0]
            default_idx = options.index(default_val) if default_val in options else 0
            
            with cols[idx]:
                inputs[col] = st.selectbox(f"Select: {col}", options=options, index=default_idx)
                if "Ball milling" in col:
                    st.caption("Ball milling=1, Non=0")

    # 数值特征区域 (每行显示 3 个)
    num_cols_per_row = 3
    cols = st.columns(num_cols_per_row)
    
    for idx, col in enumerate(numeric_cols):
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        default = float(df[col].mean())
        rng = col_max - col_min
        step = 0.001 if rng < 1 else (0.01 if rng < 10 else 0.1)
        
        with cols[idx % num_cols_per_row]:
            inputs[col] = st.number_input(
                label=col,
                min_value=col_min,
                max_value=col_max,
                value=default,
                step=step,
                format="%.6f"
            )

    # --- 预测按钮与结果 ---
    st.markdown("---")
    X_input_df = pd.DataFrame([[inputs[c] for c in features]], columns=features)
    for c in numeric_cols:
        X_input_df[c] = pd.to_numeric(X_input_df[c], errors='coerce')

    if st.button("🚀 Predict Hydrogen Production", type="primary", use_container_width=True):
        # 校验
        get_cnt = getattr(model, 'get_feature_count', None)
        expected = int(get_cnt()) if callable(get_cnt) else len(features)
        provided = X_input_df.shape[1]

        if provided != expected:
            st.error(f"Feature mismatch: Expected {expected}, got {provided}.")
        else:
            # 预测
            y_pred = float(model.predict(X_input_df)[0])
            
            # 结果展示区
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.success("Prediction Successful!")
                st.metric(label="Predicted Production", value=f"{y_pred:.4f} mL/g")
            
            with res_col2:
                # SHAP
                with st.spinner("Calculating SHAP values..."):
                    try:
                        plt.clf() # 清除旧图
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_input_df)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                        
                        shap.force_plot(
                            explainer.expected_value,
                            shap_values[0],
                            X_input_df.iloc[0],
                            matplotlib=True,
                            show=False
                        )
                        plt.savefig("shap_temp_prod.png", bbox_inches='tight', dpi=300)
                        st.image("shap_temp_prod.png", caption="SHAP Force Plot Explanation")
                    except Exception as e:
                        st.warning(f"SHAP visualization skipped: {e}")

# -----------------------------------------------------------------------------
# 4. 页面 2: 储氢密度预测 (Hydrogen Storage Density)
# -----------------------------------------------------------------------------
def page_storage_density():
    st.markdown('<p class="main-title">🔋 Hydrogen Storage Density Prediction (%)</p>', unsafe_allow_html=True)
    st.caption("Model: **NGBoost** | Target: **Hydrogen storage density(%)**")

    # 加载文件
    model_path = 'NGboost.pkl'
    data_path = '储氢密度.xlsx'
    
    model = load_model(model_path)
    df = load_data(data_path)

    if model is None or df is None:
        st.error(f"❌ 缺少文件: 请确保目录下存在 `{model_path}` 和 `{data_path}`")
        return

    # 特征处理逻辑
    target_col = 'Hydrogen storage density(%)'
    candidate_features = [c for c in df.columns if c != target_col]
    model_feature_names = getattr(model, 'feature_names_', None)
    features = list(model_feature_names) if model_feature_names else candidate_features
    features = [c for c in features if c in df.columns] # 过滤不存在的

    # 自动判定离散/数值
    auto_discrete = []
    auto_numeric = []
    for c in features:
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique(dropna=True) <= 10:
                auto_discrete.append(c)
            else:
                auto_numeric.append(c)
        else:
            auto_discrete.append(c)

    cycles_col = 'Number of cycles'
    if cycles_col in features:
        if cycles_col in auto_numeric: auto_numeric.remove(cycles_col)
        if cycles_col not in auto_discrete: auto_discrete.append(cycles_col)

    discrete_cols = auto_discrete
    numeric_cols = [c for c in features if c not in discrete_cols]

    # --- 输入区域 ---
    st.markdown('<p class="sub-header">1. Configure Parameters</p>', unsafe_allow_html=True)
    
    inputs = {}
    
    # 数值输入 (Grid Layout)
    num_cols_per_row = 3
    cols = st.columns(num_cols_per_row)
    for idx, col in enumerate(numeric_cols):
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        default = float(df[col].mean())
        rng = col_max - col_min
        step = 0.001 if rng < 1 else (0.01 if rng < 10 else 0.1)

        with cols[idx % num_cols_per_row]:
            inputs[col] = st.number_input(
                label=col,
                min_value=col_min,
                max_value=col_max,
                value=default,
                step=step,
                format="%.6f"
            )

    # 离散输入
    if discrete_cols:
        st.write("###### Discrete Variables")
        d_cols = st.columns(3)
        for idx, col in enumerate(discrete_cols):
            with d_cols[idx % 3]:
                if col == cycles_col:
                    options = [1, 2]
                    default_val = 1
                    inputs[col] = st.selectbox(col, options=options, index=0)
                    st.caption("Only 1 or 2 allowed.")
                else:
                    options = pd.Series(df[col].dropna().unique()).tolist()
                    try: options = sorted(options)
                    except: pass
                    if not options: options = [None]
                    
                    default_val = df[col].mode().iloc[0] if not df[col].mode().empty else options[0]
                    default_idx = options.index(default_val) if default_val in options else 0
                    inputs[col] = st.selectbox(col, options=options, index=default_idx)

    # --- 组装与预测 ---
    st.markdown("---")
    X_input_df = pd.DataFrame([[inputs[c] for c in features]], columns=features)
    for c in numeric_cols:
        X_input_df[c] = pd.to_numeric(X_input_df[c], errors='coerce')
    if cycles_col in X_input_df.columns:
        X_input_df[cycles_col] = pd.to_numeric(X_input_df[cycles_col], errors='coerce').astype('Int64')

    if st.button("🚀 Predict Storage Density", type="primary", use_container_width=True):
        # 校验
        get_cnt = getattr(model, 'get_feature_count', None)
        expected = int(get_cnt()) if callable(get_cnt) else len(features)
        provided = X_input_df.shape[1]

        if provided != expected:
            st.error(f"Feature mismatch: Expected {expected}, got {provided}.")
        else:
            # 预测
            y_pred = float(model.predict(X_input_df)[0])
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.success("Prediction Successful!")
                st.metric(label="Predicted Density", value=f"{y_pred:.4f} %")
            
            with res_col2:
                with st.spinner("Generating SHAP visualization..."):
                    try:
                        plt.clf() # 清除画布
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_input_df)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                        
                        shap.force_plot(
                            explainer.expected_value,
                            shap_values[0],
                            X_input_df.iloc[0],
                            matplotlib=True,
                            show=False
                        )
                        plt.savefig("shap_temp_density.png", bbox_inches='tight', dpi=300)
                        st.image("shap_temp_density.png", caption="SHAP Force Plot Explanation")
                    except Exception as e:
                        st.warning(f"SHAP visualization failed: {e}")

# -----------------------------------------------------------------------------
# 5. 主程序导航 (Sidebar Navigation)
# -----------------------------------------------------------------------------
def main():
    # 侧边栏设计
    st.sidebar.image("https://img.icons8.com/fluency/96/magnesium.png", width=60) # 可选：添加一个Logo
    st.sidebar.title("Navigation")
    
    app_mode = st.sidebar.radio(
        "Select a Prediction Model:",
        ["Hydrogen Production (mL/g)", "Storage Density (%)"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **About this App**
        
        This platform utilizes machine learning models (CatBoost & NGBoost) 
        to predict properties of Magnesium Hydride materials.
        """
    )

    # 路由逻辑
    if app_mode == "Hydrogen Production (mL/g)":
        page_hydrogen_production()
    elif app_mode == "Storage Density (%)":
        page_storage_density()

if __name__ == "__main__":
    main()
