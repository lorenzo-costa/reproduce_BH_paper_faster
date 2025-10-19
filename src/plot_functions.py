import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import logging


# Suppress matplotlib category warning for boxplots
logging.getLogger("matplotlib.category").setLevel(logging.ERROR)

def aggregate_results(
    results, y_axis, x_axis, factors=None, log_x_axis=True, log_y_axis=False
):
    """Aggregate

    Parameters
    ----------
    results : _type_
        _description_
    y_axis : _type_
        _description_
    x_axis : _type_
        _description_
    factors : _type_, optional
        _description_, by default None
    log_x_axis : bool, optional
        _description_, by default True
    log_y_axis : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    if factors is None:
        factors = []

    grouping = [x_axis] + factors

    grouped_stats = (
        results.groupby(grouping).agg({y_axis: ["mean", "sem"]}).reset_index()
    )
    grouped_stats.columns = grouping + [
        f"{y_axis}_mean",
        f"{y_axis}_sem",
    ]

    if log_y_axis is True:
        grouped_stats[f"{y_axis}_mean"] = np.log10(grouped_stats[f"{y_axis}_mean"])
        grouped_stats[f"{y_axis}_sem"] = (
            grouped_stats[f"{y_axis}_sem"] / grouped_stats[f"{y_axis}_mean"]
        )

    if log_x_axis is True:
        grouped_stats[x_axis] = np.log10(grouped_stats[x_axis])
    return grouped_stats


def plot_with_bands(x_axis, y_axis, **kwargs):
    """Plot lines with confidence/error bands for each method.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to plot.
    x_axis : str
        The name of the column to be used for the x-axis.
    y_axis : str
        The name of the column to be used for the y-axis.
    factors : list, optional
        A list of column names to be used as additional factors for grouping,
        by default None
    plot_bands : str, optional
        Name of the column containing the standard error for the y-axis values,
        if None no bands are drawn, by default None
    """
    data = kwargs.pop("data")
    factors = kwargs.pop("factors", None)
    plot_bands = kwargs.pop("plot_bands", None)
    colors = kwargs.pop("colors", None)
    linestyles = kwargs.pop("linestyles", None)

    ax = plt.gca()
    hue_variable = factors[0] if factors is not None and len(factors) >= 1 else None

    if hue_variable is not None:
        for hue_var in data[hue_variable].unique():
            subset = data[data[hue_variable] == hue_var].sort_values(x_axis)
            line = ax.plot(
                subset[x_axis],
                subset[y_axis],
                marker="o",
                linestyle=linestyles[hue_var] if linestyles is not None else "-",
                color=colors[hue_var] if colors is not None else None,
                label=hue_var,
            )
            color = line[0].get_color()

            if plot_bands is not None:
                ax.fill_between(
                    subset[x_axis],
                    subset[y_axis] - subset[plot_bands],
                    subset[y_axis] + subset[plot_bands],
                    alpha=0.2,
                    color=color,
                )
    else:
        # assume single line
        subset = data.sort_values(x_axis)
        line = ax.plot(
            subset[x_axis], subset[y_axis], marker="o", linestyle="-", label=None
        )
        color = line[0].get_color()

        if plot_bands is not None:
            ax.fill_between(
                subset[x_axis],
                subset[y_axis] - subset[plot_bands],
                subset[y_axis] + subset[plot_bands],
                alpha=0.2,
                color=color,
            )


def plot_individual(
    results,
    y_axis,
    x_axis,
    factors=None,
    colors=None,
    linestyles=None,
    save_path=None,
    log_y_axis=True,
    log_x_axis=False,
    se_bands=True,
    height=4,
    aspect=1.3,
    name_conversion=None,
):
    """
    Plot individual lineplots for each combination of aggregate_x and aggregate_y.
    """

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if factors is None:
        factors = []

    hue_variable = factors[0] if len(factors) >= 1 else None
    grouping_vars = factors[1:] if len(factors) >= 2 else []

    grouped_stats = aggregate_results(
        results,
        y_axis=y_axis,
        x_axis=x_axis,
        factors=factors,
        log_y_axis=log_y_axis,
        log_x_axis=log_x_axis,
    )

    if len(grouping_vars) > 0:
        for group_values, data_subset in grouped_stats.groupby(grouping_vars):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)

            fig, ax = plt.subplots(figsize=(height * aspect, height))

            plot_with_bands(
                data=data_subset,
                x_axis=x_axis,
                y_axis=y_axis + "_mean",
                factors=factors,
                plot_bands=y_axis + "_sem" if se_bands else None,
                ax=ax,
                colors=colors,
                linestyles=linestyles,
            )

            if len(grouping_vars) == 2:
                title = (
                    f"{name_conversion.get(grouping_vars[0], grouping_vars[0]).replace('_', ' ').title()}: {group_values[0]}, "
                    f"{name_conversion.get(grouping_vars[1], grouping_vars[1]).replace('_', ' ').title()}: {group_values[1]}"
                )
            elif len(grouping_vars) == 1:
                title = (
                    f"{name_conversion.get(grouping_vars[0], grouping_vars[0]).replace('_', ' ').title()}: {group_values[0]}"
                )
            else:
                title = (
                    f"{name_conversion.get(x_axis, x_axis).replace('_', ' ').title()} vs "
                    f"{name_conversion.get(y_axis, y_axis).replace('_', ' ').title()}"
                )

            ax.set_title(title)
            xlabel = (
                name_conversion.get(x_axis, x_axis).replace("_", " ").title()
            )
            ylabel = (
                name_conversion.get(y_axis, y_axis).replace("_", " ").title()
            )
            ax.set_xlabel("Log " + xlabel if log_x_axis else xlabel)
            ax.set_ylabel("Log " + ylabel if log_y_axis else ylabel)
            # plt.tight_layout()
            plt.legend(title="Method")

            if save_path is not None:
                path = f"{save_path}" + "".join(f"_{val}" for val in group_values)
                plt.savefig(
                    f"{path}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.savefig(
                    f"{path}.pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
            else:
                plt.show()
            plt.close()
    else:
        raise NotImplementedError


def plot_grid(
    results,
    x_axis,
    y_axis,
    factors,
    colors=None,
    linestyles=None,
    save_path=None,
    log_x_axis=True,
    log_y_axis=False,
    se_bands=True,
    height=1.3,
    aspect=1.3,
    name_conversion=None,
):
    """
    Plot a grid of RMSE lineplots faceted by x_axis and y_axis.
    """
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    grouped_stats = aggregate_results(
        results,
        x_axis=x_axis,
        y_axis=y_axis,
        factors=factors,
        log_x_axis=log_x_axis,
        log_y_axis=log_y_axis,
    )
    hue_variable = factors[0] if len(factors) >= 3 else None
    aggregate_x = factors[1] if len(factors) >= 3 else factors[0]
    aggregate_y = factors[2] if len(factors) >= 3 else factors[1]

    # Sort aggregate_y values in descending order so largest appears on top
    grouped_stats[aggregate_y] = pd.Categorical(
        grouped_stats[aggregate_y],
        categories=sorted(grouped_stats[aggregate_y].unique(), reverse=True),
        ordered=True,
    )

    g = sns.FacetGrid(
        grouped_stats,
        row=aggregate_y,
        col=aggregate_x,
        margin_titles=True,
        sharey=True,
        sharex=True,
        height=height,
        aspect=aspect,
    )
    g.map_dataframe(
        plot_with_bands,
        x_axis=x_axis,
        y_axis=y_axis + "_mean",
        factors=factors,
        plot_bands=y_axis + "_sem" if se_bands else None,
        colors=colors,
        linestyles=linestyles,
    )
    # Remove default x/y axis labels and tick labels from all subplots
    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        
    # Set x and y axis labels only in central places
    x_label = (
        "Log " + name_conversion.get(x_axis, x_axis).replace("_", " ").title()
        if log_x_axis
        else name_conversion.get(x_axis, x_axis).replace("_", " ").title()
    )
    g.axes[-1, g.axes.shape[1] // 2].set_xlabel(x_label)
    
    y_label = (
        "Log " + name_conversion.get(y_axis, y_axis).replace("_", " ").title()
        if log_y_axis
        else name_conversion.get(y_axis, y_axis).replace("_", " ").title()
    )
    
    g.axes[g.axes.shape[0] // 2, 0].set_ylabel(y_label)
   
   
    # Set column facet titles
    for ax in range(g.axes.shape[1]):
        title = (
            f"{name_conversion.get(aggregate_x, aggregate_x).replace('_', ' ').title()}: {g.col_names[ax]}"
        )
        g.axes[0, ax].set_title(title)
    # Set custom row facet labels 
    for ax in range(g.axes.shape[0]):
        text = f"{int(g.row_names[ax]*100)}\% {name_conversion.get(aggregate_y, aggregate_y).replace('_', ' ').title()}"
        # text = f"{name_conversion.get(aggregate_y, aggregate_y).replace('_', ' ').title()}: {g.row_names[ax]}"
        g.axes[ax, -1].texts[0].set_text(text)
    # Set figure title at the top
    if log_y_axis is True:
        g.figure.suptitle(
            "Log " + name_conversion.get(x_axis, x_axis) + " vs Log " + name_conversion.get(y_axis, y_axis)
            if log_x_axis
            else name_conversion.get(x_axis, x_axis) + " vs Log " + name_conversion.get(y_axis, y_axis),
            y=1.02,
        )
    else:
        g.figure.suptitle(
            "Log " + name_conversion.get(x_axis, x_axis) + " vs " + name_conversion.get(y_axis, y_axis)
            if log_x_axis
            else name_conversion.get(x_axis, x_axis) + " vs " + name_conversion.get(y_axis, y_axis),
            y=1.02,
        )
    g.add_legend()
    if save_path is not None:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    return g


def plot_boxplot(
    results,
    colors=None,
    save_path=None,
    log_rmse=True,
    log_x=False,
    height=1.3,
    aspect=1.3,
    x="degrees_of_freedom",
    y="rmse",
    n_boxplots=3,
):
    """
    Plot RMSE boxplot versus the given x variable. Optionally log-transform RMSE or x,
    and limit number of boxplots.
    """
    plt.rcParams.update(custom_rcparams)
    if colors is None:
        colors = defaults_colors
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    temp = results.copy()

    if log_x is True:
        temp[x] = np.log10(temp[x])
    if log_rmse is True:
        temp["rmse"] = np.log10(temp["rmse"])

    if x == "SNR":
        xname = f"{x.replace('_', ' ')}"
    else:
        xname = f"{x.replace('_', ' ').title()}"

    if n_boxplots is not None:
        if n_boxplots < len(temp[x].unique()):
            # Select n_boxplots evenly spaced along x
            df_values = sorted(temp[x].unique())
            selected_dfs = np.linspace(0, len(df_values) - 1, n_boxplots, dtype=int)
            selected_dfs = [df_values[i] for i in selected_dfs]
            temp = temp[temp[x].isin(selected_dfs)]

    fig, ax = plt.subplots(figsize=(height * aspect, height))
    sns.boxplot(data=temp, x=x, y=y, hue="name", ax=ax)

    plt.xlabel(f"Log {xname}" if log_x else f"{xname}")
    plt.ylabel("Log RMSE" if log_rmse else "RMSE")
    plt.tight_layout()
    plt.legend(title="Method")

    plt.title(
        "Boxplot of "
        + ("Log RMSE" if log_rmse else "RMSE")
        + " vs "
        + (f"Log {xname}" if log_x else f"{xname}")
    )

    if save_path is not None:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
