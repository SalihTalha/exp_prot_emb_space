import streamlit as st
import numpy as np
import plotly.express as px
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from visualizations.utils import load_labels, load_data, common_labels

labels = load_labels().drop(["Unnamed: 0"], axis=1)

os.environ["STREAMLIT_SERVER_MAXMESSAGESIZE"] = "300"  # Max message size in MB
os.environ["STREAMLIT_SERVER_RUNONSAVE"] = "true"     # Enable run on save

st.set_page_config(
    page_title="Heatmap Inspector",
    page_icon="🏙",
    layout="wide",  # Set the layout to wide mode
    initial_sidebar_state="expanded"  # Expand sidebar by default

)

@st.cache_data
def load_dataset(dataset_name: str, label_name: str, label_value: str, reverse_filter: bool) -> np.ndarray:
    """
    Returns a matrix (2D NumPy array) for the given dataset name.
    Replace these examples with your actual data-loading logic.
    """

    if dataset_name == "ANKH":
        tensor = torch.load(f"./final_embeddings/ankh_merged_tensor.pt", map_location=torch.device('cpu'))
        indexes = list(labels[labels[label_name] == label_value].index)
        if reverse_filter:
            mask = torch.ones(tensor.size(0), dtype=torch.bool)  # Initialize all True
            mask[indexes] = False  # Set False where you want to remove
            # Select only the rows you want to keep
            return tensor[mask][:1000]
        return tensor[indexes][:1000]
    elif dataset_name == "ProtGPT2":
        tensor = torch.load(f"./final_embeddings/protgpt2_merged_tensor.pt", map_location=torch.device('cpu'))
        indexes = list(labels[labels[label_name] == label_value].index)
        if reverse_filter:
            mask = torch.ones(tensor.size(0), dtype=torch.bool)  # Initialize all True
            mask[indexes] = False  # Set False where you want to remove
            # Select only the rows you want to keep
            return tensor[mask][:1000]
        return tensor[indexes][:1000]
    elif dataset_name == "ESM2":
        tensor = torch.load(f"./final_embeddings/esm2_merged_tensor.pt", map_location=torch.device('cpu'))
        indexes = list(labels[labels[label_name] == label_value].index)
        if reverse_filter:
            mask = torch.ones(tensor.size(0), dtype=torch.bool)  # Initialize all True
            mask[indexes] = False  # Set False where you want to remove
            # Select only the rows you want to keep
            return tensor[mask][:1000]
        return tensor[indexes][:1000]
    else:
        # Default case or raise an error
        raise ValueError(f"Unknown dataset name: {dataset_name}")


# --- Streamlit App ---

st.title("Grandmother Cell Inspection")

st.write("""
---
**How to Use**  
1. Select a dataset from the dropdown.  
2. Adjust `vmin` and `vmax` to set the color scale range.  
3. Click the **Plot Heatmap** button.  
4. Zoom or pan using the Plotly toolbar in the top-right corner.  
""")

# 1. Dropdown to choose the dataset
DATASET_OPTIONS = [
    "ANKH",
    "ProtGPT2",
    "ESM2"
]

layer_counts = {
    "ANKH": 48,
    "ProtGPT2": 36,
    "ESM2": 48
}

dataset_name = st.selectbox("Select a dataset:", DATASET_OPTIONS)

LABEL_OPTIONS = {i: labels[i].unique() for i in labels.keys()}
label_column = st.selectbox("Select label column:", list(LABEL_OPTIONS.keys()))

# Dynamically populate label_value based on the chosen label_column
possible_values_for_column = LABEL_OPTIONS[label_column]
label_value = st.selectbox("Select label value:", possible_values_for_column)


layer = st.slider(
    "Layer",
    min_value=0,  # a rough lower limit
    max_value= layer_counts[dataset_name],  # a rough upper limit
    value=0,
    step=1
)

data_limit = st.slider(
    "Data Limit",
    min_value=100,  # a rough lower limit
    max_value=1000,  # a rough upper limit
    value=100,
    step=1
)

reverse_filter = st.checkbox("Exclude Filter", value=False)
with st.expander("About exclude filter"):
    st.info("Selecting this checkbox allows you to inspect the exclusion of the selected label. "
            "This allows you to examine activations outside the selected label. This may take some time.")

if "data" not in st.session_state:
    st.session_state.data = None
if "layer" not in st.session_state:
    st.session_state.layer = 0

# 5. Button to plot the heatmap
if st.button("Plot Heatmap") or st.session_state.data is not None:
    st.session_state.data = load_dataset(dataset_name, label_column, label_value, reverse_filter)[:data_limit, :, :]

    # Access stored data
    data = st.session_state.data

    # 3. Determine min/max of the data, for slider guidance
    data_min, data_max = np.percentile(data[:, layer, :], 0.5), np.percentile(data[:, layer, :], 99.5) # 5000, 8000
    data_minm, data_maxm = np.percentile(data[:, layer, :], 0), np.percentile(data[:, layer, :], 100)

    st.write(f"**Selected dataset**: {dataset_name} — shape: {data.shape}")
    st.write(f"**Label column**: {label_column}, **Label value**: {label_value}")
    st.write("Choose color scale limits (vmin, vmax):")

    vmin = st.number_input(
        f"vmin (lower bound) - 0.5 percentile = {str(data_min) if abs(data_min) < 100 else int(data_min)} - min: {str(data_minm) if abs(data_minm) < 100 else int(data_minm)}",
        value=float(data_min),  # Default value
        step=0.1  # Increment step for float values
    )

    vmax = st.number_input(
        f"vmax (upper bound) 99.5 percentile = {str(data_max) if abs(data_max) < 100 else int(data_max)} - max: {str(data_maxm) if data_maxm < 100 else int(data_maxm)}",
        value=float(data_max),  # Default value
        step=0.1  # Increment step for float values
    )

    data = data * ((data >= data_max) | (data <= data_min)).float()

    if vmin > vmax:
        st.warning("vmin is greater than vmax. Please adjust the sliders.")
    else:
        fig = px.imshow(
            data[:, layer, :],
            zmin=vmin,
            zmax=vmax,
            color_continuous_scale="RdBu_r",
            aspect="auto"  # Keeps cells from getting too stretched
        )

        # Configure Plotly layout
        fig.update_layout(
            dragmode="zoom",
            margin=dict(l=40, r=40, t=40, b=40),
            xaxis_title=f"Neurons",
            yaxis_title=f"Data Points of {label_column}:{label_value}",
            title=f"Heatmap of {dataset_name} Activations on {layer}. Layer" # & Exclude: {reverse_filter}
        )

        # fig.update_xaxes(range=[100, 150])

        # Display the figure
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Download High-Resolution PNG"):
            # Convert to high-res PNG
            img_bytes = fig.to_image(format="png", scale=4.5)  # Adjust scale for higher resolution
            st.download_button(
                label="Download Image",
                data=img_bytes,
                file_name="heatmap.png",
                mime="image/png"
            )
