import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import utils


def getColorDistribution(image):
    blue, green, red = utils.splitChannels(image * 255)
    figure = plt.figure(figsize=(15, 15), dpi=80)
    plot = figure.add_subplot(projection='3d')

    plot.set_title("RGB")
    plot.set_xlabel('Red')
    plot.set_ylabel('Green')
    plot.set_zlabel('Blue')

    plot.scatter(
        xs=red,
        ys=green,
        zs=blue,
        marker=".",
    )

    plot.plot([0, 255], [0, 255], [0, 255], color="black", linestyle='-', linewidth=2)

    plt.show()
    return figure

def getPlaneScatters(flatData, idx_inliers, finalPoints):
    inliers = flatData[idx_inliers]

    mask = np.ones(len(flatData), dtype=bool)
    mask[idx_inliers] = False
    outliers = flatData[mask]

    inliers_marker_data = go.Scatter3d(
        x=inliers[:, 0],
        y=inliers[:, 1],
        z=inliers[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color='blue',  # set color to an array/list of desired values
            opacity=0.9
        )
    )

    fig = go.Figure(data=inliers_marker_data)
    fig.update_scenes(xaxis_title="Blue", yaxis_title="Green", zaxis_title="Red", xaxis_range=[0, 1], yaxis_range=[0, 1], zaxis_range=[0, 1])
    fig.add_scatter3d(
        x=outliers[:, 0],
        y=outliers[:, 1],
        z=outliers[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color='coral',  # set color to an array/list of desired values
            opacity=0.5
        )
    )
    fig.add_scatter3d(x=[finalPoints[0][0], finalPoints[1][0]], y=[finalPoints[0][1], finalPoints[1][1]], z=[finalPoints[0][2], finalPoints[1][2]], mode='lines',
        marker=dict(size=1, color='black', opacity=0.8)
    )
    fig.add_scatter3d(x=[finalPoints[0][0], finalPoints[2][0]], y=[finalPoints[0][1], finalPoints[2][1]], z=[finalPoints[0][2], finalPoints[2][2]], mode='lines',
        marker=dict(size=1, color='black', opacity=0.8)
    )
    fig.add_scatter3d(x=[finalPoints[1][0], finalPoints[2][0]], y=[finalPoints[1][1], finalPoints[2][1]], z=[finalPoints[1][2], finalPoints[2][2]], mode='lines',
        marker=dict(size=1, color='black', opacity=0.8)
    )
    fig.add_scatter3d(
        x=[0, 1],
        y=[0, 1],
        z=[0, 1],
        mode='lines',
        marker=dict(
            size=1,
            color='gray',  # set color to an array/list of desired values
            opacity=0.5
        )
    )

    return fig

def getProjectionPlaneScatter(data, idx):
    scatter = go.Scatter(
        x=data[:, 0],
        y=data[:, 1],
        mode='markers',
        marker=dict(
            size=1,
            color='green',  # set color to an array/list of desired values
            opacity=0.9
        )
    )
    fig = go.Figure(data=scatter)

    fig.add_scatter(
        x=data[idx][:, 0],
        y=data[idx][:, 1],
        mode='markers',
        marker=dict(
            size=1,
            color='blue',
            opacity=0.9
        )
    )
    return fig
