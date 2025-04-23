import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import torch
import torch.nn as nn
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import TSNE
from scipy.interpolate import griddata

# -------------------- CONFIG --------------------
st.set_page_config(page_title="EEG Psychiatric App", layout="wide")
sns.set(style="whitegrid")
warnings.filterwarnings("ignore")

# -------------------- SIDEBAR NAVIGATION --------------------
page = st.sidebar.radio("Navigation", ["📊 Visualization Dashboard", "🧠 Predict Disorder"])

# -------------------- EEG MODEL --------------------
class EEGClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, neurons, dropout=0.10):
        super(EEGClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, neurons[0])
        self.layer2 = nn.Linear(neurons[0], neurons[1])
        self.layer3 = nn.Linear(neurons[1], neurons[2])
        self.layer4 = nn.Linear(neurons[2], neurons[3])
        self.layer5 = nn.Linear(neurons[3], neurons[4])
        self.output_layer = nn.Linear(neurons[4], output_dim)

        self.bn1 = nn.BatchNorm1d(neurons[0])
        self.bn2 = nn.BatchNorm1d(neurons[1])
        self.bn3 = nn.BatchNorm1d(neurons[2])
        self.bn4 = nn.BatchNorm1d(neurons[3])
        self.bn5 = nn.BatchNorm1d(neurons[4])

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.gelu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.gelu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.gelu(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        x = self.gelu(self.bn4(self.layer4(x)))
        x = self.dropout(x)
        x = self.gelu(self.bn5(self.layer5(x)))
        x = self.output_layer(x)
        return x

# -------------------- VISUALIZATION DASHBOARD --------------------
if page == "📊 Visualization Dashboard":
    st.title("📊 EEG Psychiatric Disorders Dashboard")
    st.markdown("##### Visualize EEG features, then slide your 6 band‑mean values to predict your disorder.")

    @st.cache_data
    def load_and_preprocess(path="EEG.machinelearing_data_BRMH.csv"):
        df = pd.read_csv(path)
        df = (df.dropna(axis=1, how='all')
                .dropna(axis=0)
                .drop_duplicates()
                .drop(columns=['NoiseChannel1', 'NoiseChannel2'], errors='ignore'))
        bands = ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']
        for b in bands:
            col = f"{b}_mean"
            if col not in df:
                cols = [c for c in df.columns if c.startswith("AB.") and f".{b}." in c]
                df[col] = df[cols].mean(axis=1)
        le = LabelEncoder()
        df['target'] = le.fit_transform(df['main.disorder'])
        raw_feats = [c for c in df.columns if c.startswith("AB.") or c.startswith("COH.")]
        selector = VarianceThreshold(threshold=1e-3)
        selector.fit(df[raw_feats])
        feature_names = [f for f, keep in zip(raw_feats, selector.get_support()) if keep]
        scaler = StandardScaler().fit(df[feature_names].values.astype(np.float32))
        return df, bands, feature_names, scaler, le

    df, bands, feature_names, scaler, label_enc = load_and_preprocess()

    # 2.1 Main Disorder & Gender
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Main Disorder Distribution")
        fig, ax = plt.subplots()
        df['main.disorder'].value_counts().plot.bar(color='skyblue', ax=ax)
        st.pyplot(fig)
    with c2:
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots()
        if 'sex' in df:
            df['sex'].value_counts().plot.bar(color='lightgreen', ax=ax)
        st.pyplot(fig)

    st.markdown("---")

    # 2.2 Age & IQ
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['age'], kde=True, ax=ax, color='purple')
        st.pyplot(fig)
    with c4:
        st.subheader("IQ by Disorder")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='main.disorder', y='IQ', palette="Set2", ax=ax)
        plt.setp(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

    st.markdown("---")

    # 2.3 EEG Band Power Histograms
    st.subheader("EEG Band Power Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()
    palette = sns.color_palette("Set2", len(bands))
    for i, b in enumerate(bands):
        sns.histplot(df[f"{b}_mean"], bins=40, ax=axes[i], color=palette[i], edgecolor='black')
        axes[i].set_title(f"{b.capitalize()} Mean Power")
    for j in range(len(bands), 6):
        fig.delaxes(axes[j])
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # 2.4 Band Power by Disorder
    st.subheader("Band Power by Main Disorder")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, b in enumerate(bands):
        sns.boxplot(x='main.disorder', y=f"{b}_mean", data=df,
                    palette=[palette[i]]*df['main.disorder'].nunique(), ax=axes[i])
        axes[i].tick_params(axis='x', rotation=45)
    for j in range(len(bands), 6):
        fig.delaxes(axes[j])
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # 2.5 Mean PSD Heatmap
    st.subheader("Mean PSD per Electrode by Disorder")
    electrodes = ['FP1','FP2','F7','F3','Fz','F4','F8','C3','Cz','C4','P3','Pz','P4','O1','O2']
    records = []
    for d in df['main.disorder'].unique():
        row = {'disorder': d}
        for e in electrodes:
            cols = [c for c in df if c.startswith('AB.') and e in c]
            row[e] = df.loc[df['main.disorder']==d, cols].mean().mean()
        records.append(row)
    heat_df = pd.DataFrame(records).set_index('disorder')
    fig = px.imshow(heat_df, labels=dict(x="Electrode", y="Disorder", color="Mean PSD"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 2.6 Alpha vs Beta
    st.subheader("Alpha vs Beta by Specific Disorder")
    fig = px.scatter(df, x='alpha_mean', y='beta_mean', color='specific.disorder')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 2.7 PairGrid
    st.subheader("Pairwise Alpha PSD")
    sample_feats = [c for c in df if c.startswith('AB.') and '.alpha.' in c][:3]
    g = sns.PairGrid(df, vars=sample_feats, hue='main.disorder', palette="Set2", corner=False)
    g.map_upper(sns.scatterplot, alpha=0.6, edgecolor='w', s=30)
    g.map_lower(sns.scatterplot, alpha=0.6, edgecolor='w', s=30)
    g.map_diag(sns.histplot, element='step', fill=False)
    g.add_legend(bbox_to_anchor=(1.05,1))
    plt.suptitle("Pairwise Alpha PSD", y=1.02)
    st.pyplot(g.fig)
    plt.close(g.fig)

    st.markdown("---")

    # 2.8 t-SNE of Delta Coherence
    st.subheader("t-SNE of Delta Coherence")
    delta_feats = [c for c in df if c.startswith('COH.') and ".delta." in c]
    if delta_feats:
        X2 = TSNE(n_components=2, random_state=0).fit_transform(df[delta_feats].fillna(0))
        fig = px.scatter(x=X2[:,0], y=X2[:,1], color=df['main.disorder'])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 2.9 Mean Coherence per Band
    st.subheader("Mean Coherence per Frequency Band")
    mean_coh = []
    for b in bands:
        cols = [c for c in df if c.startswith('COH.') and f".{b}." in c]
        mean_coh.append({'band': b.capitalize(), 'coherence': df[cols].mean().mean() if cols else 0})
    mc_df = pd.DataFrame(mean_coh)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x='band', y='coherence', data=mc_df, palette="Set2", edgecolor='black', ax=ax)
    for p in ax.patches:
        ax.text(p.get_x()+p.get_width()/2, p.get_height()+0.01, f"{p.get_height():.2f}", ha='center', va='bottom')
    st.pyplot(fig)

    st.markdown("---")

    # 2.10 Radar Chart
    st.subheader("Mean Band-Power Profile per Disorder")
    radar_df = df.groupby('main.disorder')[[f"{b}_mean" for b in bands]].mean()
    angles = np.linspace(0, 2*np.pi, len(bands), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    colors = sns.color_palette("Set2", len(radar_df))
    for i, (d, row) in enumerate(radar_df.iterrows()):
        vals = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, vals, color=colors[i], linewidth=2, label=d)
        ax.fill(angles, vals, color=colors[i], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_yticklabels([])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
    st.pyplot(fig)

    st.markdown("---")

    # 2.11 Topographic Map

    st.subheader("Simulated EEG Topographic Map for Each Band")

    # Dropdown to select EEG Band
    selected_band = st.selectbox("Select EEG Band for Topographic Map", bands)

    # Filter columns for selected band
    band_cols = [col for col in df.columns if f'.{selected_band}.' in col and col.startswith('AB.')]
    band_means = df[band_cols].mean()

    # Map electrode labels to positions (10-20 system approximation)
    electrode_pos = {
        'FP1': (-0.5, 1.0), 'FP2': (0.5, 1.0), 'F7': (-1.0, 0.5), 'F3': (-0.5, 0.5), 'Fz': (0.0, 0.5),
        'F4': (0.5, 0.5), 'F8': (1.0, 0.5), 'T3': (-1.0, 0.0), 'C3': (-0.5, 0.0), 'Cz': (0.0, 0.0),
        'C4': (0.5, 0.0), 'T4': (1.0, 0.0), 'T5': (-1.0, -0.5), 'P3': (-0.5, -0.5), 'Pz': (0.0, -0.5),
        'P4': (0.5, -0.5), 'T6': (1.0, -0.5), 'O1': (-0.5, -1.0), 'O2': (0.5, -1.0)
    }

    # Extract electrode labels from column names
    electrode_labels = [col.split('.')[-1] for col in band_means.index]
    positions = np.array([electrode_pos[elec] for elec in electrode_labels])
    values = band_means.values

    # Interpolation Grid
    grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]
    grid_z = griddata(positions, values, (grid_x, grid_y), method='cubic')

    # Plot Topographic Map
    fig, ax = plt.subplots(figsize=(6, 6))
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=100, cmap='Spectral')
    ax.scatter(positions[:,0], positions[:,1], c='black', s=50, label='Electrodes')
    for i, txt in enumerate(electrode_labels):
        ax.annotate(txt, (positions[i,0], positions[i,1]), fontsize=8, ha='center')
    plt.colorbar(contour)
    plt.title(f"Topographic Map - {selected_band.capitalize()} Band")
    plt.axis('off')
    st.pyplot(fig)

    st.markdown("---")

    #---------------------------- Topographic Maps Across Disorders and Bands --------------------

    st.subheader("Topographic Maps Across Disorders and Bands")

    # Prepare electrode positions (as before)
    electrode_pos = {
        'FP1': (-0.5, 1.0), 'FP2': (0.5, 1.0), 'F7': (-1.0, 0.5), 'F3': (-0.5, 0.5), 'Fz': (0.0, 0.5),
        'F4': (0.5, 0.5), 'F8': (1.0, 0.5), 'T3': (-1.0, 0.0), 'C3': (-0.5, 0.0), 'Cz': (0.0, 0.0),
        'C4': (0.5, 0.0), 'T4': (1.0, 0.0), 'T5': (-1.0, -0.5), 'P3': (-0.5, -0.5), 'Pz': (0.0, -0.5),
        'P4': (0.5, -0.5), 'T6': (1.0, -0.5), 'O1': (-0.5, -1.0), 'O2': (0.5, -1.0)
    }

    # Unique disorders
    disorders = df['main.disorder'].unique() 

    # Grid plot setup
    fig, axes = plt.subplots(len(disorders), len(bands), figsize=(4 * len(bands), 4 * len(disorders)))

    # Loop through disorders and bands
    for i, disorder in enumerate(disorders):
        df_subset = df[df['main.disorder'] == disorder]
        for j, band in enumerate(bands):
            ax = axes[i, j] if len(disorders) > 1 else axes[j]

            # Calculate mean for current disorder and band
            band_cols = [col for col in df.columns if f'.{band}.' in col and col.startswith('AB.')]
            band_means = df_subset[band_cols].mean()
            electrode_labels = [col.split('.')[-1] for col in band_means.index]
            positions = np.array([electrode_pos[elec] for elec in electrode_labels])
            values = band_means.values

            grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]
            grid_z = griddata(positions, values, (grid_x, grid_y), method='cubic')

            contour = ax.contourf(grid_x, grid_y, grid_z, levels=100, cmap='Spectral')
            ax.scatter(positions[:,0], positions[:,1], c='black', s=30)
            ax.set_title(f"{disorder} - {band.capitalize()}", fontsize=10)
            ax.axis('off')

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    #---------------------------- 3D t-SNE Visualization --------------------

    st.subheader("3D t-SNE Visualization of EEG Features")
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Choose top 100 features to reduce complexity
    feature_subset = df[[col for col in df.columns if col.startswith("AB.") or col.startswith("COH.")]].fillna(0).iloc[:, :100]

    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(feature_subset)

    tsne_df = pd.DataFrame(tsne_results, columns=["TSNE1", "TSNE2", "TSNE3"])
    tsne_df["Disorder"] = df["main.disorder"]

    fig = px.scatter_3d(tsne_df, x="TSNE1", y="TSNE2", z="TSNE3", color="Disorder", opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

#---------------------------- Band Power Violin Plots --------------------
    st.markdown("---")
    st.subheader("Violin Plots of Band Power by Disorder")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, b in enumerate(bands):
        sns.violinplot(x='main.disorder', y=f"{b}_mean", data=df, ax=axes[i], palette="Set2", inner="quartile")
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_title(f"{b.capitalize()} Band")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

#---------------------------- Interactive Electrode Map --------------------
    import plotly.graph_objects as go

    st.subheader("Interactive Electrode Map with Band Power")

    # Dropdown to select EEG Band for the interactive map
    selected_band = st.selectbox("Select EEG Band for Interactive Electrode Map", bands, key="interactive_band")

    # Calculate mean power for selected band
    band_cols = [col for col in df.columns if f'.{selected_band}.' in col and col.startswith('AB.')]
    band_means = df[band_cols].mean()

    # Electrode positions (10-20 system approx)
    electrode_pos = {
        'FP1': (-0.5, 1.0), 'FP2': (0.5, 1.0), 'F7': (-1.0, 0.5), 'F3': (-0.5, 0.5), 'Fz': (0.0, 0.5),
        'F4': (0.5, 0.5), 'F8': (1.0, 0.5), 'T3': (-1.0, 0.0), 'C3': (-0.5, 0.0), 'Cz': (0.0, 0.0),
        'C4': (0.5, 0.0), 'T4': (1.0, 0.0), 'T5': (-1.0, -0.5), 'P3': (-0.5, -0.5), 'Pz': (0.0, -0.5),
        'P4': (0.5, -0.5), 'T6': (1.0, -0.5), 'O1': (-0.5, -1.0), 'O2': (0.5, -1.0)
    }

    electrode_labels = [col.split('.')[-1] for col in band_means.index]
    positions = np.array([electrode_pos[elec] for elec in electrode_labels])
    values = band_means.values

    # Plotly Interactive Scatter Map
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode='markers+text',
        text=electrode_labels,
        textposition="top center",
        marker=dict(
            size=15,
            color=values,
            colorscale='Spectral',
            colorbar=dict(title='Power'),
            line=dict(width=2, color='DarkSlateGrey')
        ),
        hoverinfo='text',
        hovertext=[f"{label}: {val:.2f}" for label, val in zip(electrode_labels, values)]
    ))

    fig.update_layout(
        title=f'Interactive Electrode Map - {selected_band.capitalize()} Band',
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        showlegend=False,
        width=600,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


# -------------------- PREDICTION PAGE --------------------
if page == "🧠 Predict Disorder":
    @st.cache_data
    def load_and_preprocess(path="EEG.machinelearing_data_BRMH.csv"):
            df = pd.read_csv(path)
            df = (df.dropna(axis=1, how='all')
                    .dropna(axis=0)
                    .drop_duplicates()
                    .drop(columns=['NoiseChannel1', 'NoiseChannel2'], errors='ignore'))
            bands = ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']
            for b in bands:
                col = f"{b}_mean"
                if col not in df:
                    cols = [c for c in df.columns if c.startswith("AB.") and f".{b}." in c]
                    df[col] = df[cols].mean(axis=1)
            le = LabelEncoder()
            df['target'] = le.fit_transform(df['main.disorder'])
            raw_feats = [c for c in df.columns if c.startswith("AB.") or c.startswith("COH.")]
            selector = VarianceThreshold(threshold=1e-3)
            selector.fit(df[raw_feats])
            feature_names = [f for f, keep in zip(raw_feats, selector.get_support()) if keep]
            scaler = StandardScaler().fit(df[feature_names].values.astype(np.float32))
            return df, bands, feature_names, scaler, le

    df, bands, feature_names, scaler, label_enc = load_and_preprocess()
    st.title("🧠 EEG Psychiatric Disorder Predictor")
    st.markdown("Enter your 226 features either by **uploading a CSV** or **pasting values below**. Optional demographic inputs help personalize the interface but are not used for prediction.")

    col1, col2 = st.columns(2)
    with col1:
        st.slider("Age", 10, 80, 30)
        st.radio("Sex", ["M", "F"])
    with col2:
        st.text_input("IQ Score", value="110")

    input_type = st.radio("Choose input method:", ["Upload CSV", "Paste 226 values"])
    user_input = None

    if input_type == "Upload CSV":
        st.download_button("📥 Download Template", data=",".join(["feature_"+str(i) for i in range(226)]), file_name="eeg_template.csv")
        uploaded_file = st.file_uploader("Upload your single-row .csv file:", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if df.shape[1] == 226:
                user_input = df.values
            else:
                st.error("Uploaded file must have exactly 226 columns.")

    elif input_type == "Paste 226 values":
        pasted = st.text_area("Paste 226 comma-separated numbers:")
        if pasted:
            try:
                values = [float(v.strip()) for v in pasted.split(",")]
                if len(values) != 226:
                    st.error(f"❌ You provided {len(values)} values. Exactly 226 required.")
                else:
                    user_input = np.array(values).reshape(1, -1)
            except:
                st.error("❌ Invalid numeric format. Ensure all values are numbers separated by commas.")

    if user_input is not None and st.button("🎯 Predict Disorder"):
        with torch.no_grad():
            scaler = StandardScaler()
            scaler.mean_ = np.zeros(226)
            scaler.scale_ = np.ones(226)
            user_scaled = scaler.transform(user_input)

            input_dim = 226
            output_dim = 7
            neurons = [1024, 512, 256, 128, 64]
            model = EEGClassifier(input_dim, output_dim, neurons)
            model.load_state_dict(torch.load("best_model.pt", map_location=torch.device("cpu")))
            model.eval()

            tensor_input = torch.tensor(user_scaled, dtype=torch.float32)
            output = model(tensor_input)
            predicted_class = torch.argmax(output, dim=1).item()

            disorder_map = {
                0: "Addictive disorder",
                1: "Trauma and stress related disorder",
                2: "Mood disorder",
                3: "Healthy control",
                4: "Obsessive compulsive disorder",
                5: "Schizophrenia",
                6: "Anxiety disorder"
            }

            st.success(f"🎯 Predicted Disorder: **{disorder_map.get(predicted_class, 'Unknown')}**")

    with st.container():  # OR keep inside the Visual Dashboard block
        st.subheader("🎯 Quick Predict: Based on Band Mean Values")
        input_band_values = {}
        for b in bands:
            input_band_values[b] = st.slider(f"{b.capitalize()} Mean", float(df[f"{b}_mean"].min()), float(df[f"{b}_mean"].max()), float(df[f"{b}_mean"].mean()))

        if st.button("Predict Disorder Based on Band Means"):
            # rest of prediction code...

            # Create dummy feature vector for 226 features (simplified with band mean mapping)
            dummy_input = np.zeros(226)
            band_to_index_map = {  # Example mapping of band indices (You can replace with correct ones)
                'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3, 'highbeta': 4, 'gamma': 5
            }
            for band, val in input_band_values.items():
                dummy_input[band_to_index_map[band]] = val

            with torch.no_grad():
                scaler = StandardScaler()
                scaler.mean_ = np.zeros(226)
                scaler.scale_ = np.ones(226)
                user_scaled = scaler.transform(dummy_input.reshape(1, -1))

                input_dim = 226
                output_dim = 7
                neurons = [1024, 512, 256, 128, 64]
                model = EEGClassifier(input_dim, output_dim, neurons)
                model.load_state_dict(torch.load("best_model.pt", map_location=torch.device("cpu")))
                model.eval()

                tensor_input = torch.tensor(user_scaled, dtype=torch.float32)
                output = model(tensor_input)
                predicted_class = torch.argmax(output, dim=1).item()

                disorder_map = {
                    0: "Addictive disorder",
                    1: "Trauma and stress related disorder",
                    2: "Mood disorder",
                    3: "Healthy control",
                    4: "Obsessive compulsive disorder",
                    5: "Schizophrenia",
                    6: "Anxiety disorder"
                }

                st.success(f"🎯 Predicted Disorder: **{disorder_map.get(predicted_class, 'Unknown')}**")

# --- Project Resources Section ---
st.markdown("---")
st.subheader("🔗 Project Resources")

st.markdown("""
- 📂 **Dataset**: [EEG Psychiatric Disorders Dataset](https://www.kaggle.com/datasets/shashwatwork/eeg-psychiatric-disorders-dataset?select=EEG.machinelearing_data_BRMH.csv)
- 📄 **Final Paper**: [Download Final Paper](https://github.com/revanth-15/Data-VizProject/blob/main/Decoding%20Psychiatric%20Disorders%20with%20EEG%20Signals%20-%20%20Research%20Paper%20-%20Group%208.pdf)
- 📚 **References**: [Download References](https://github.com/Shreyasrs23/Group-8-data-Viz/tree/main/Researh%20Papers)
""")

with st.expander("📚 View References"):
    st.markdown("""
    1. Identification of Major Psychiatric Disorders From Resting-State Electroencephalography Using a Machine Learning Approach. By *Su Mi Park, Boram Jeong, Da Young Oh, Chi-Hyun Choi, Hee Yeon Jung, Jun-Young Lee, Donghwan Lee and Jung-Seok Choi* [Link](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2021.707581/pdf)

    2. EEG Frequency Bands in Psychiatric Disorders: A Review of Resting State Studies Jennifer J. Newson and Tara C. Thiagarajan*. [Link](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2018.00521/pdf)

    3. EEG in psychiatric practice: to do or not to do? Published online by Cambridge University Press:  02 January 2018. [Link](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/F671BE17183176E364A2C2578B2DDD61/S1355514600007434a.pdf/eeg-in-psychiatric-practice-to-do-or-not-to-do.pdf)
    """)
