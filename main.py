import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# EDM Dataset based on the paper
def load_edm_data():
    """Load the complete EDM experimental data from the paper"""
    data = {
        'Run': list(range(1, 31)),
        'Voltage_V': [70, 60, 60, 70, 50, 50, 60, 70, 60, 60, 70, 70, 60, 60, 60, 
                      50, 60, 50, 70, 50, 50, 60, 60, 50, 50, 60, 70, 50, 70, 70],
        'Current_A': [8, 12, 12, 8, 8, 16, 12, 16, 12, 12, 8, 16, 12, 16, 12, 
                      16, 12, 8, 16, 12, 8, 12, 12, 16, 8, 8, 12, 16, 16, 8],
        'Pulse_On_us': [10, 8, 8, 6, 6, 10, 10, 6, 8, 8, 6, 10, 8, 8, 8, 
                        6, 8, 6, 6, 8, 10, 8, 6, 6, 10, 8, 8, 10, 10, 10],
        'Pulse_Off_us': [7, 9, 9, 7, 11, 7, 9, 11, 9, 9, 11, 11, 9, 9, 9, 
                         7, 7, 7, 7, 9, 11, 11, 9, 11, 7, 9, 9, 11, 7, 11],
        'MRR_g_min': [0.23, 0.473, 0.441, 0.0789, 0.5075, 0.2574, 0.6162, 0.086, 0.4731, 0.4482,
                      0.4272, 0.616, 0.4623, 0.5707, 0.4572, 0.2193, 0.322, 0.099, 0.3206, 0.205,
                      1.051, 0.6305, 0.34, 0.0448, 0.2004, 0.7129, 0.2412, 0.525, 0.4086, 1.0208],
        'EWR_g_min': [0.003, 0.007, 0.005, 0.004, 0.0004, 0.0115, 0.007, 0.004, 0.006, 0.008,
                      0.0004, 0.0117, 0.017, 0.0101, 0.007, 0.008, 0.008, 0.0042, 0.0105, 0.0076,
                      0.0043, 0.0057, 0.0044, 0.003, 0.0039, 0.0041, 0.0085, 0.0107, 0.0133, 0.0036],
        'SR_um': [11.558, 14.717, 14.867, 7.647, 6.245, 14.514, 13.608, 10.168, 10.325, 15.851,
                  9.04, 14.514, 16.24, 11.728, 12.485, 10.008, 12.512, 6.301, 9.577, 12.629,
                  10.389, 12.196, 7.545, 6.753, 13.289, 9.149, 18.214, 16.758, 14.814, 14.322]
    }
    return pd.DataFrame(data)

# Material properties from the paper
material_properties = {
    'Ti_properties': {'mean_particle_size_um': 5, 'morphology': 'Angular', 'melting_point_C': 1670},
    'Zr_properties': {'mean_particle_size_um': 5, 'morphology': 'Angular', 'melting_point_C': 1850},
    'Nb_properties': {'mean_particle_size_um': 7, 'morphology': 'Angular', 'melting_point_C': 2468},
    'chemical_composition': {
        'Ti': {'N': 0.872, 'O': 0.349, 'C': 0.073, 'Si': 0.025, 'Fe': 0.040},
        'Nb': {'N': 0.038, 'O': 0.620, 'C': 0.020, 'Si': 0.0, 'Fe': 0.040},
        'Zr': {'N': 0.080, 'O': 0.450, 'C': 0.028, 'Si': 0.0, 'Fe': 0.030}
    }
}

# Empirical models from the paper
def predict_mrr(voltage, current, pulse_on, pulse_off):
    """Material Removal Rate prediction using empirical model from paper"""
    A, B, C, D = voltage, current, pulse_on, pulse_off
    mrr = (8.10803 + 0.29398*A - 0.13650*B - 0.12425*C + 0.10421*D + 
           7.58906E-4*A*B + 6.23438E-4*A*C - 7.52188E-4*A*D - 
           1.98516E-3*B*C - 0.018218*B*D + 0.027630*C*D - 
           2.49610E-3*A**2 + 0.010543*B**2 - 3.68991E-3*C**2 + 
           8.60088E-4*D**2)
    return max(0, mrr)

def predict_ewr(voltage, current, pulse_on, pulse_off):
    """Electrode Wear Rate prediction using empirical model from paper"""
    A, B, C, D = voltage, current, pulse_on, pulse_off
    ewr = (0.016010 - 7.80062E-4*A - 5.79770E-5*B + 2.75685E-3*C - 
           4.57785E-4*D + 1.09375E-5*A*B - 9.37500E-6*A*C - 
           9.37500E-6*A*D + 1.17187E-4*B*C - 3.90625E-5*B*D + 
           2.65625E-4*C*D + 7.10526E-6*A**2 - 1.80921E-5*B**2 - 
           3.22368E-4*C**2 - 7.23684E-5*D**2)
    return max(0, ewr)

def predict_sr(voltage, current, pulse_on, pulse_off):
    """Surface Roughness prediction using empirical model from paper"""
    A, B, C, D = voltage, current, pulse_on, pulse_off
    sr = (22.89445 - 2.99977*A + 4.27854*B + 10.92146*C + 0.37029*D - 
          8.28594E-3*A*B - 0.021459*A*C + 0.026297*A*D + 0.029445*B*C - 
          0.015008*B*D + 0.048984*C*D + 0.025885*A**2 - 0.14965*B**2 - 
          0.56411*C**2 - 0.11974*D**2)
    return max(0, sr)

# Streamlit App Framework
def main():
    st.set_page_config(page_title="EDMLearner - Intelligent Process Optimization", layout="wide")
    
    st.title("EDMLearner - Intelligent Process Optimization")
    st.markdown("### Ti-13Zr-13Nb Orthopedic Implant Manufacturing Optimization")
    
    # Display EDM image
    try:
        st.image("EDM.png", width=400, caption="Electrical Discharge Machining Process")
    except:
        st.info("Place EDM.png image file in the same directory as the app")
    
    # Sidebar for navigation
    app_mode = st.sidebar.selectbox("Choose Application Mode",
        ["Data Explorer", "Process Optimizer", "ML Predictions", 
         "3D Surface Analysis", "Digital Twin Simulator"])
    
    # Load data
    df = load_edm_data()
    
    if app_mode == "Data Explorer":
        data_explorer(df)
    elif app_mode == "Process Optimizer":
        process_optimizer(df)
    elif app_mode == "ML Predictions":
        ml_predictions(df)
    elif app_mode == "3D Surface Analysis":
        surface_analysis(df)
    elif app_mode == "Digital Twin Simulator":
        digital_twin_simulator(df)

def data_explorer(df):
    st.header("Experimental Data Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Interactive correlation heatmap
        st.subheader("Parameter Correlation Matrix")
        corr_vars = ['Voltage_V', 'Current_A', 'Pulse_On_us', 'Pulse_Off_us', 'MRR_g_min', 'EWR_g_min', 'SR_um']
        corr_matrix = df[corr_vars].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       title="Process Parameters Correlation",
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Dataset Overview")
        st.write(f"**Total Experiments:** {len(df)}")
        st.write(f"**Design Type:** Face-Centered CCD")
        st.write(f"**Material:** Ti-13Zr-13Nb")
        st.write(f"**Application:** Orthopedic Implants")
        
        # Summary statistics
        st.subheader("Response Variables Summary")
        responses = df[['MRR_g_min', 'EWR_g_min', 'SR_um']].describe()
        st.dataframe(responses)

def process_optimizer(df):
    st.header("Multi-Objective Process Optimization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Optimization Goals")
        mrr_weight = st.slider("MRR Importance", 0.0, 1.0, 0.4)
        ewr_weight = st.slider("EWR Importance (minimize)", 0.0, 1.0, 0.3)
        sr_weight = st.slider("SR Importance (minimize)", 0.0, 1.0, 0.3)
        
        st.subheader("Process Constraints")
        voltage_range = st.slider("Voltage Range (V)", 50, 70, (50, 70))
        current_range = st.slider("Current Range (A)", 8, 16, (8, 16))
        pulse_on_range = st.slider("Pulse ON Range (μs)", 6, 10, (6, 10))
        pulse_off_range = st.slider("Pulse OFF Range (μs)", 7, 11, (7, 11))
    
    with col2:
        # Pareto frontier analysis
        st.subheader("Pareto Frontier Analysis")
        
        # Create objective function combinations
        objectives_data = []
        for _, row in df.iterrows():
            mrr_norm = row['MRR_g_min'] / df['MRR_g_min'].max()
            ewr_norm = 1 - (row['EWR_g_min'] / df['EWR_g_min'].max())  # Minimize
            sr_norm = 1 - (row['SR_um'] / df['SR_um'].max())  # Minimize
            
            composite_score = mrr_weight * mrr_norm + ewr_weight * ewr_norm + sr_weight * sr_norm
            
            objectives_data.append({
                'Run': row['Run'],
                'MRR_Score': mrr_norm,
                'EWR_Score': ewr_norm,
                'SR_Score': sr_norm,
                'Composite_Score': composite_score,
                'Voltage': row['Voltage_V'],
                'Current': row['Current_A']
            })
        
        obj_df = pd.DataFrame(objectives_data)
        
        # 3D scatter plot of objectives
        fig = go.Figure(data=go.Scatter3d(
            x=obj_df['MRR_Score'],
            y=obj_df['EWR_Score'], 
            z=obj_df['SR_Score'],
            mode='markers+text',
            marker=dict(
                size=8,
                color=obj_df['Composite_Score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Composite Score")
            ),
            text=obj_df['Run'],
            hovertemplate='Run: %{text}<br>MRR Score: %{x:.3f}<br>EWR Score: %{y:.3f}<br>SR Score: %{z:.3f}'
        ))
        
        fig.update_layout(
            title="Multi-Objective Optimization Space",
            scene=dict(
                xaxis_title="MRR Score (Higher Better)",
                yaxis_title="EWR Score (Higher Better)", 
                zaxis_title="SR Score (Higher Better)"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Best solutions table
        st.subheader("Top 5 Optimized Solutions")
        best_solutions = obj_df.nlargest(5, 'Composite_Score')
        for idx, row in best_solutions.iterrows():
            original_row = df[df['Run'] == row['Run']].iloc[0]
            st.write(f"**Run {row['Run']}** - Score: {row['Composite_Score']:.3f}")
            st.write(f"Parameters: V={original_row['Voltage_V']}V, I={original_row['Current_A']}A, "
                    f"T_on={original_row['Pulse_On_us']}μs, T_off={original_row['Pulse_Off_us']}μs")
            st.write(f"Results: MRR={original_row['MRR_g_min']:.4f}, EWR={original_row['EWR_g_min']:.4f}, SR={original_row['SR_um']:.2f}")
            st.write("---")

def ml_predictions(df):
    st.header("Machine Learning Predictions")
    
    # Prepare data
    features = ['Voltage_V', 'Current_A', 'Pulse_On_us', 'Pulse_Off_us']
    X = df[features]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        target_var = st.selectbox("Select Target Variable", 
                                 ['MRR_g_min', 'EWR_g_min', 'SR_um'])
        model_type = st.selectbox("Select Model", 
                                 ['Random Forest', 'Gradient Boosting', 'Empirical Model'])
        
        st.subheader("Prediction Parameters")
        voltage = st.slider("Voltage (V)", 50, 70, 60)
        current = st.slider("Current (A)", 8, 16, 12)
        pulse_on = st.slider("Pulse ON (μs)", 6, 10, 8)
        pulse_off = st.slider("Pulse OFF (μs)", 7, 11, 9)
        
        # Make prediction
        if model_type == "Empirical Model":
            if target_var == 'MRR_g_min':
                prediction = predict_mrr(voltage, current, pulse_on, pulse_off)
            elif target_var == 'EWR_g_min':
                prediction = predict_ewr(voltage, current, pulse_on, pulse_off)
            else:
                prediction = predict_sr(voltage, current, pulse_on, pulse_off)
        else:
            # Train ML model
            y = df[target_var]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if model_type == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = GradientBoostingRegressor(random_state=42)
            
            model.fit(X_train, y_train)
            prediction = model.predict([[voltage, current, pulse_on, pulse_off]])[0]
            
            # Model performance
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            st.subheader("Model Performance")
            st.write(f"R² Score: {r2:.3f}")
            st.write(f"RMSE: {rmse:.4f}")
        
        st.subheader("Prediction Result")
        st.metric(f"Predicted {target_var}", f"{prediction:.4f}")
    
    with col2:
        st.subheader("Feature Importance Analysis")
        
        if model_type != "Empirical Model":
            # Feature importance plot
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(feature_importance_df, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title=f"Feature Importance for {target_var}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Sensitivity analysis
        st.subheader("Parameter Sensitivity Analysis")
        sensitivity_data = []
        
        base_params = [voltage, current, pulse_on, pulse_off]
        param_names = ['Voltage', 'Current', 'Pulse_On', 'Pulse_Off']
        param_ranges = [(50, 70), (8, 16), (6, 10), (7, 11)]
        
        for i, (param_name, (min_val, max_val)) in enumerate(zip(param_names, param_ranges)):
            param_values = np.linspace(min_val, max_val, 20)
            predictions = []
            
            for val in param_values:
                test_params = base_params.copy()
                test_params[i] = val
                
                if model_type == "Empirical Model":
                    if target_var == 'MRR_g_min':
                        pred = predict_mrr(*test_params)
                    elif target_var == 'EWR_g_min':
                        pred = predict_ewr(*test_params)
                    else:
                        pred = predict_sr(*test_params)
                else:
                    pred = model.predict([test_params])[0]
                
                predictions.append(pred)
                sensitivity_data.append({
                    'Parameter': param_name,
                    'Value': val,
                    'Prediction': pred
                })
        
        sens_df = pd.DataFrame(sensitivity_data)
        fig = px.line(sens_df, x='Value', y='Prediction', color='Parameter',
                     title=f"Parameter Sensitivity for {target_var}")
        st.plotly_chart(fig, use_container_width=True)

def surface_analysis(df):
    st.header("3D Response Surface Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Surface Configuration")
        response_var = st.selectbox("Response Variable", 
                                   ['MRR_g_min', 'EWR_g_min', 'SR_um'])
        
        param1 = st.selectbox("X-Axis Parameter", 
                             ['Voltage_V', 'Current_A', 'Pulse_On_us', 'Pulse_Off_us'])
        param2 = st.selectbox("Y-Axis Parameter", 
                             ['Current_A', 'Voltage_V', 'Pulse_On_us', 'Pulse_Off_us'])
        
        # Fixed parameters
        st.subheader("Fixed Parameters")
        fixed_params = {}
        for param in ['Voltage_V', 'Current_A', 'Pulse_On_us', 'Pulse_Off_us']:
            if param not in [param1, param2]:
                if param == 'Voltage_V':
                    fixed_params[param] = st.slider(f"{param}", 50, 70, 60)
                elif param == 'Current_A':
                    fixed_params[param] = st.slider(f"{param}", 8, 16, 12)
                elif param == 'Pulse_On_us':
                    fixed_params[param] = st.slider(f"{param}", 6, 10, 8)
                else:
                    fixed_params[param] = st.slider(f"{param}", 7, 11, 9)
    
    with col2:
        # Generate surface data
        if param1 == 'Voltage_V':
            x_vals = np.linspace(50, 70, 25)
        elif param1 == 'Current_A':
            x_vals = np.linspace(8, 16, 25)
        elif param1 == 'Pulse_On_us':
            x_vals = np.linspace(6, 10, 25)
        else:
            x_vals = np.linspace(7, 11, 25)
        
        if param2 == 'Voltage_V':
            y_vals = np.linspace(50, 70, 25)
        elif param2 == 'Current_A':
            y_vals = np.linspace(8, 16, 25)
        elif param2 == 'Pulse_On_us':
            y_vals = np.linspace(6, 10, 25)
        else:
            y_vals = np.linspace(7, 11, 25)
        
        X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
        Z_mesh = np.zeros_like(X_mesh)
        
        for i in range(len(y_vals)):
            for j in range(len(x_vals)):
                params = fixed_params.copy()
                params[param1] = X_mesh[i, j]
                params[param2] = Y_mesh[i, j]
                
                voltage = params['Voltage_V']
                current = params['Current_A'] 
                pulse_on = params['Pulse_On_us']
                pulse_off = params['Pulse_Off_us']
                
                if response_var == 'MRR_g_min':
                    Z_mesh[i, j] = predict_mrr(voltage, current, pulse_on, pulse_off)
                elif response_var == 'EWR_g_min':
                    Z_mesh[i, j] = predict_ewr(voltage, current, pulse_on, pulse_off)
                else:
                    Z_mesh[i, j] = predict_sr(voltage, current, pulse_on, pulse_off)
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(x=X_mesh, y=Y_mesh, z=Z_mesh,
                                        colorscale='Viridis')])
        
        fig.update_layout(
            title=f'3D Response Surface: {response_var}',
            scene=dict(
                xaxis_title=param1,
                yaxis_title=param2,
                zaxis_title=response_var
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Contour plot
        fig_contour = go.Figure(data=go.Contour(x=x_vals, y=y_vals, z=Z_mesh,
                                               colorscale='Viridis'))
        fig_contour.update_layout(
            title=f'Contour Plot: {response_var}',
            xaxis_title=param1,
            yaxis_title=param2
        )
        st.plotly_chart(fig_contour, use_container_width=True)

def digital_twin_simulator(df):
    st.header("Digital Twin Process Simulator")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Process Parameters")
        voltage = st.slider("Voltage (V)", 50, 70, 60, key="dt_voltage")
        current = st.slider("Current (A)", 8, 16, 12, key="dt_current")
        pulse_on = st.slider("Pulse ON (μs)", 6, 10, 8, key="dt_pulse_on")
        pulse_off = st.slider("Pulse OFF (μs)", 7, 11, 9, key="dt_pulse_off")
        
        machining_time = st.slider("Machining Time (min)", 5, 60, 20)
        
    with col2:
        st.subheader("Real-time Predictions")
        
        mrr_pred = predict_mrr(voltage, current, pulse_on, pulse_off)
        ewr_pred = predict_ewr(voltage, current, pulse_on, pulse_off)
        sr_pred = predict_sr(voltage, current, pulse_on, pulse_off)
        
        st.metric("Material Removal Rate", f"{mrr_pred:.4f} g/min", 
                 delta=f"{(mrr_pred - df['MRR_g_min'].mean()):.4f}")
        st.metric("Electrode Wear Rate", f"{ewr_pred:.4f} g/min",
                 delta=f"{(ewr_pred - df['EWR_g_min'].mean()):.4f}")
        st.metric("Surface Roughness", f"{sr_pred:.2f} μm",
                 delta=f"{(sr_pred - df['SR_um'].mean()):.2f}")
        
        # Process efficiency metrics
        st.subheader("Process Efficiency")
        material_removed = mrr_pred * machining_time
        electrode_consumed = ewr_pred * machining_time
        wear_ratio = (electrode_consumed / material_removed * 100) if material_removed > 0 else 0
        
        st.metric("Material Removed", f"{material_removed:.3f} g")
        st.metric("Electrode Consumed", f"{electrode_consumed:.4f} g") 
        st.metric("Wear Ratio", f"{wear_ratio:.2f}%")
        
    with col3:
        st.subheader("Process Status")
        
        # Process health indicators
        mrr_health = "Optimal" if 0.3 < mrr_pred < 0.8 else "Acceptable" if 0.1 < mrr_pred < 1.2 else "Critical"
        ewr_health = "Optimal" if ewr_pred < 0.008 else "Acceptable" if ewr_pred < 0.015 else "Critical"
        sr_health = "Optimal" if sr_pred < 10 else "Acceptable" if sr_pred < 15 else "Critical"
        
        st.write(f"**MRR Status:** {mrr_health}")
        st.write(f"**EWR Status:** {ewr_health}")
        st.write(f"**SR Status:** {sr_health}")
        
        # Process recommendations
        st.subheader("Recommendations")
        recommendations = []
        
        if mrr_pred < 0.2:
            recommendations.append("Increase current or pulse ON time for higher MRR")
        if ewr_pred > 0.012:
            recommendations.append("Reduce current to minimize electrode wear")
        if sr_pred > 15:
            recommendations.append("Reduce pulse ON time for better surface finish")
        if not recommendations:
            recommendations.append("Current parameters are well optimized!")
            
        for rec in recommendations:
            st.write(f"• {rec}")
    
    # Advanced analytics
    st.subheader("Advanced Process Analytics")
    
    # Process parameter radar chart
    col1, col2 = st.columns(2)
    
    with col1:
        # Normalize parameters for radar chart
        param_scores = {
            'MRR Score': min(mrr_pred / df['MRR_g_min'].max(), 1.0),
            'EWR Score': min(1.0 - ewr_pred / df['EWR_g_min'].max(), 1.0),
            'SR Score': min(1.0 - sr_pred / df['SR_um'].max(), 1.0),
            'Efficiency': min(material_removed / (material_removed + electrode_consumed), 1.0) if (material_removed + electrode_consumed) > 0 else 0
        }
        
        # Create radar chart
        categories = list(param_scores.keys())
        values = list(param_scores.values())
        values += values[:1]  # Complete the circle
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name='Current Process'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Process Performance Radar"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Process timeline simulation
        st.subheader("Process Timeline")
        time_steps = np.linspace(0, machining_time, 100)
        
        # Simulate gradual wear and material removal
        material_curve = time_steps * mrr_pred
        electrode_curve = time_steps * ewr_pred
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_steps, y=material_curve, name='Material Removed (g)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=time_steps, y=electrode_curve, name='Electrode Wear (g)', line=dict(color='red')))
        
        fig.update_layout(
            title="Process Timeline Simulation",
            xaxis_title="Time (minutes)",
            yaxis_title="Mass (g)"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
