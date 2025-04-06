import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
def plot_line(
    data,
    labels=None,
    xlabel="x",
    ylabel="y",
    title=None,
    x_tick=None,
    width=1200,
    height=600,
    hlines=None,  # New parameter for horizontal lines
    hline_labels=None  # Labels for the horizontal lines
):
    # Convert tensor→numpy
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    n_points = data.shape[-1]

    # Default x = integer indices
    if x_tick is None:
        x_tick = np.arange(n_points)

    fig = go.Figure()
    if data.ndim == 2:
        for i, row in enumerate(data):
            name = labels[i] if labels else f"Line {i}"
            fig.add_trace(go.Scatter(x=x_tick, y=row, mode="lines", name=name))
    else:
        fig.add_trace(go.Scatter(x=x_tick, y=data, mode="lines"))

    if hlines is not None:
        for idx, hline in enumerate(hlines):
            line_label = hline_labels[idx] if hline_labels else f"hline {idx}"
            fig.add_trace(go.Scatter(
                x=[x_tick[0], x_tick[-1]],
                y=[hline, hline],
                mode="lines",
                line=dict(dash="dash", width=2),
                name=line_label
            ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height
    )

    if x_tick is not None:
        if isinstance(x_tick[0], str):
            fig.update_xaxes(type="category", tickvals=x_tick)
        else:
            fig.update_xaxes(tickmode="array", tickvals=x_tick)

    fig.show()


def plot_bar(data, x_tick=None, xlabel="x", ylabel="y", title=None, figsize=(14, 6), tick_angle=45):
    # Convert tensor→numpy if needed
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    
    # If no x_tick provided, use default indices (as strings)
    if x_tick is None:
        x_tick = [str(i) for i in range(len(data))]
    
    # Ensure x_tick and data have the same length
    if len(x_tick) != len(data):
        raise ValueError(f"Length mismatch: x_tick has {len(x_tick)} elements, data has {len(data)} elements")
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the bar plot
    ax.bar(range(len(data)), data)
    
    # Set the x-tick positions and labels
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(x_tick, rotation=tick_angle, ha='right')
    
    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Adjust bottom margin to accommodate long tick labels
    plt.subplots_adjust(bottom=0.25)
    
    # Add gridlines for better readability
    # ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_heatmap(data,x_tick,y_tick,xlabel,ylabel,title='None'):
    x_tick_vals = list(range(data.shape[1]))
    y_tick_vals = list(range(data.shape[0]))
    fig = go.Figure(data=go.Heatmap(
            z=data,
            x=x_tick_vals,  # assign tick values along the x-axis
            y=y_tick_vals,  # assign tick values along the y-axis
            colorscale='Viridis'  # choose a colorscale
        ))
    fig.update_layout(
    title=title,
    xaxis=dict(
        title=xlabel,
        tickmode='array',
        tickvals=x_tick_vals,
        ticktext=x_tick
    ),
    yaxis=dict(
        title=ylabel,
        tickmode='array',
        tickvals=y_tick_vals,
        ticktext=y_tick
        ),
    )

    fig.show()


def plot_tensor_scatter(data, x_tick, y_tick, xlabel,ylabel,labels=None, title='None',figsize = (8,6),markers = None,sharey=True): # if sharey, ytick is a single list else xtick is
    # If a single tensor is provided, wrap it in a list.
    if not isinstance(data, list):
        data = [data]
    # Default labels if not provided.
    if labels is None:
        labels = [f"Tensor {i+1}" for i in range(len(data))]
    
    fig, ax = plt.subplots(figsize=figsize)
    sc = None  # This will hold the last scatter for the colorbar.

    if not markers:
        markers = ['o','^','s','v','p']
        if len(data) > len(markers):
            raise ValueError("Not enough markers provided for the number of data, include your own markers")
    
    vmax = max([np.max(tensor) for tensor in data])
    vmin = min([np.min(tensor) for tensor in data])
    # Loop through each tensor and add it as a scatter trace.
    for sample_pos,(tensor, label) in enumerate(zip(data, labels)):
        ny, nx = tensor.shape
        sample_xtick = x_tick[sample_pos] if sharey else x_tick
        sameple_ytick = y_tick if sharey else y_tick[sample_pos]
        xs, ys, colors = [], [], []
        for i in range(ny):
            for j in range(nx):
                xs.append(sample_xtick[j])
                ys.append(i if isinstance(sameple_ytick[0], str) else sameple_ytick[i])
                colors.append(tensor[i, j])
        
        sc = ax.scatter(xs, ys, c=colors, cmap='viridis', label=label, s=75, edgecolors='k',marker = markers[sample_pos],vmax=vmax,vmin=vmin,alpha=0.7)
    
    # Set the x-axis limits to a free range (e.g., 0 to 100).
    # ax.set_xlim(0, 100)
    
    # Configure y-axis:
    # If y_tick are strings, use indices for positions and set the tick labels accordingly.
    if isinstance(y_tick[0], str):
        ax.set_yticks(range(len(y_tick)))
        ax.set_yticklabels(y_tick)
    else:
        ax.set_yticks(y_tick)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add a colorbar for the marker colors.
    plt.colorbar(sc, ax=ax,)
    
    # Place legend on top of the plot.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(data))
    
    plt.tight_layout()
    plt.show()