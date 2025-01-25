import pandas as pd
import streamlit as st
import plotly.express as px
import os
import numpy as np
import torch

from visualizations.utils import load_labels, load_data, common_labels, get_seq_lens, get_label_name

labels = load_labels().drop(["Unnamed: 0"], axis=1)
seq_lengths = np.array(get_seq_lens())

os.environ["STREAMLIT_SERVER_MAXMESSAGESIZE"] = "300"  # Max message size in MB
os.environ["STREAMLIT_SERVER_RUNONSAVE"] = "true"     # Enable run on save

st.set_page_config(
    page_title="Heatmap Inspector",
    page_icon="🏙",
    layout="wide",  # Set the layout to wide mode
    initial_sidebar_state="expanded"  # Expand sidebar by default

)


def run_label_analysis():
    # 1) Load the entire dataset (no filtering by label_name/label_value)
    #    but slice out the chosen neuron (neuron_index).
    dataset = load_dataset(
        dataset_name=dataset_name,  # or "ANKH"/"ProtGPT2"/ etc.
        label_name=None,
        label_value=None,
        reverse_filter=False,
        layer=layer,
        size_limit=40000,  # pick a large enough number or remove if you don't want a limit
        neuron=neuron_index
    )
    # dataset shape is now [N, seq_len] if 'neuron' was given.

    # 2) For each sample i, compute fraction of positions > min_value
    dataset_np = dataset.numpy()  # easier to handle in numpy
    results = []

    for label_col in labels.columns:
        for unique_label in labels[label_col].unique():
            idx = labels[labels[label_col] == unique_label].index

            if len(idx) < 10:
                # No samples have this label => skip
                continue

            # 4) Extract the activations for these indices
            group_data = dataset_np[idx]  # shape => [g, seq_len], where g is # of samples

            count_above = np.sum(group_data > max_value) + np.sum(group_data < min_value)
            total_positions = group_data.size
            fraction = count_above / total_positions
            percentage = fraction * 100

            # 5) Compare to your percentage_threshold
            group_label = "OUTLIER" if percentage >= percentage_threshold else "NON-OUTLIER"

            # Save results
            results.append({
                "label_column": label_col,
                "label_value": unique_label,
                "label_name": get_label_name(unique_label),
                "count_samples_in_group": len(idx),
                "percentage_above_min_value": percentage,
                "mean_sequence_lengths": np.mean(seq_lengths[idx]),
                "mean_activation": np.mean(group_data),
                "group": group_label,
            })


    results_df = pd.DataFrame(results)
    results_df["label_value"] = results_df["label_value"].astype(str)

    # Sort by percentage descending
    results_df.sort_values(by="percentage_above_min_value", ascending=False, inplace=True)

    # Show the summary
    st.subheader("Label Grouping Summary")

    # Example: number of HIGH vs LOW across *all* columns
    high_count = sum(results_df["group"] == "OUTLIER")
    low_count = sum(results_df["group"] == "NON-OUTLIER")
    st.write(f"Number of OUTLIER labels (all columns combined): {high_count}")
    st.write(f"Number of NON-OUTLIER labels (all columns combined): {low_count}")

    # Show top 5 and bottom 5
    st.write("### OUTLIERS Percentage Dataframe")
    st.dataframe(results_df[results_df["group"] == "OUTLIER"])

    st.write("### NON-OUTLIERS Percentage Dataframe")
    st.dataframe(results_df[results_df["group"] == "NON-OUTLIER"])

    # results_df.to_csv(f"results_{datetime.datetime.now()}.csv")



@st.cache_data
def load_dataset(dataset_name: str, label_name: str, label_value: str, reverse_filter: bool, size_limit=1000, neuron=None, layer=0) -> np.ndarray:
    """
    Returns a matrix (2D NumPy array) for the given dataset name.
    Replace these examples with your actual data-loading logic.
    """

    if dataset_name == "ANKH":
        tensor = torch.load(f"./final_embeddings/ankh_merged_tensor.pt", map_location=torch.device('cpu'))
        tensor = tensor[:, layer, :]
        if neuron:
            tensor = tensor[:, neuron]
        if label_name and label_value:
            indexes = list(labels[labels[label_name] == label_value].index)
        else:
            indexes = range(len(tensor))
        if reverse_filter:
            mask = torch.ones(tensor.size(0), dtype=torch.bool)  # Initialize all True
            mask[indexes] = False  # Set False where you want to remove
            # Select only the rows you want to keep
            return tensor[mask][:size_limit]
        return tensor[indexes][:size_limit]
    elif dataset_name == "ProtGPT2":
        tensor = torch.load(f"./final_embeddings/protgpt2_merged_tensor.pt", map_location=torch.device('cpu'))
        tensor = tensor[:, layer, :]
        if neuron:
            tensor = tensor[:, neuron]
        if label_name and label_value:
            indexes = list(labels[labels[label_name] == label_value].index)
        else:
            indexes = range(len(tensor))
        if reverse_filter:
            mask = torch.ones(tensor.size(0), dtype=torch.bool)  # Initialize all True
            mask[indexes] = False  # Set False where you want to remove
            # Select only the rows you want to keep
            return tensor[mask][:size_limit]
        return tensor[indexes][:size_limit]
    elif dataset_name == "ESM2":
        tensor = torch.load(f"./final_embeddings/esm2_merged_tensor.pt", map_location=torch.device('cpu'))
        tensor = tensor[:, layer, :]
        print(tensor.shape)
        if neuron:
            tensor = tensor[:, neuron]
            print(tensor.shape)
        if label_name and label_value:
            indexes = list(labels[labels[label_name] == label_value].index)
        else:
            indexes = range(len(tensor))
        if reverse_filter:
            mask = torch.ones(tensor.size(0), dtype=torch.bool)  # Initialize all True
            mask[indexes] = False  # Set False where you want to remove
            # Select only the rows you want to keep
            return tensor[mask][:size_limit]
        return tensor[indexes][:size_limit]
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
    max_value=layer_counts[dataset_name],  # a rough upper limit
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
    # In column 1: generate and show a plot
    st.session_state.data = load_dataset(dataset_name, label_column, label_value, reverse_filter, layer=layer)[:data_limit]

    # Access stored data
    data = st.session_state.data

    # 3. Determine min/max of the data, for slider guidance
    data_min, data_max = np.percentile(data, 0.5), np.percentile(data, 99.5)  # 5000, 8000
    data_minm, data_maxm = np.percentile(data, 0), np.percentile(data, 100)

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

    # data = data * ((data >= data_max) | (data <= data_min)).float()

    # Create two columns: col1 is wide, col2 is narrow (just an example ratio)
    col1, col2 = st.columns([3, 1])

    with col1:
        if vmin > vmax:
            st.warning("vmin is greater than vmax. Please adjust the sliders.")
        else:
            fig = px.imshow(
                data,
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

    with col2:
        neuron_index = st.number_input(
            label="Neuron Index",
            min_value=0,
            value=0,
            step=1,
            help="Neuron index you want to analyze"
        )

        # 2) Numeric input for the activation threshold (min_value)
        min_value = st.number_input(
            label="Min Activation Value",
            value=0.0,
            step=0.1,
            help="Count positions below this activation"
        )

        max_value = st.number_input(
            label="Max Activation Value",
            value=0.0,
            step=0.1,
            help="Count positions above this activation"
        )

        # 3) Percentage threshold
        percentage_threshold = st.slider(
            label="Percentage threshold (0-100)",
            min_value=0,
            max_value=100,
            value=50,
            help="Labels with fraction of positions above min_value exceeding this threshold are 'HIGH'"
        )

        # 4) Button
        run_button = st.button("Run Analysis")

    if run_button:
        run_label_analysis()