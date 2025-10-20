import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import logging
import re


# Suppress weird matplotlib category warning for boxplots
logging.getLogger("matplotlib.category").setLevel(logging.ERROR)


def aggregate_results(
    results, 
    y_axis, 
    x_axis, 
    factors=None, 
    log_x_axis=True, 
    log_y_axis=False
):
    """Compute dataset with mean and standard error for each group.

    Parameters
    ----------
    results : pd.DataFrame
        Input DataFrame to group and aggregate.
    y_axis : str
        The name of the column to be used for the y-axis.
    x_axis : str
        The name of the column to be used for the x-axis.
    factors : list, optional
        A list of column names to be used as additional factors for grouping,
        by default None
    log_x_axis : bool, optional
        Whether to use a logarithmic scale for the x-axis, by default True
    log_y_axis : bool, optional
        Whether to use a logarithmic scale for the y-axis, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame containing the aggregated results with mean and standard error for each group
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


def plot_with_bands(
    x_axis, 
    y_axis, 
    **kwargs
):
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
    colors : dict, optional
        A dictionary mapping factor values to colors, by default None
    linestyles : dict, optional
        A dictionary mapping factor values to linestyles, by default None
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

def plot_grid(
    results,
    x_axis,
    y_axis,
    factors,
    plotting_function=None,
    **kwargs
):
    """Plot a grid of plots using the specified plotting function.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing the data to plot.
    x_axis : str
        The name of the column to be used for the x-axis.
    y_axis : str
        The name of the column to be used for the y-axis.
    factors : list
        A list of column names to be used as additional factors for grouping.
    plotting_function : callable, optional
        A function to use for plotting, by default None
    height : float, optional
        Height of each facet in inches, by default 1.3
    aspect : float, optional
        Aspect ratio of each facet, by default 1.3
    group_variables : bool, optional
        Whether to aggregate results by computing mean and standard error
        for each combination of factors, by default False
    se_bands : bool, optional
        Whether to plot standard error bands, by default False
    log_y_axis : bool, optional
        Whether to use a logarithmic scale for the y-axis, by default False
    log_x_axis : bool, optional
        Whether to use a logarithmic scale for the x-axis, by default False
    name_conversion : dict, optional
        A dictionary mapping variable names to more descriptive names for
        axis labels and titles, by default {}
    add_legend : bool, optional
        Whether to add a legend to the plot, by default True
    save_path : str, optional
        Path to save the plot, by default None
        

    Returns
    -------
    sns.FacetGrid
        The FacetGrid object containing the plots.
    """
    if plotting_function is None:
        raise ValueError("plotting_function must be provided.")
    
    height = kwargs.get("height", 1.3)
    save_path = kwargs.get("save_path", None)
    group_variables = kwargs.get("group_variables", False)
    se_bands = kwargs.get("se_bands", False)
    log_y_axis = kwargs.get("log_y_axis", False)
    log_x_axis = kwargs.get("log_x_axis", False)
    aspect = kwargs.get("aspect", 1.3)
    name_conversion = kwargs.get("name_conversion", {})
    add_legend = kwargs.get("add_legend", True)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if group_variables is True:
        grouped_stats = aggregate_results(
            results,
            x_axis=x_axis,
            y_axis=y_axis,
            factors=factors,
            log_x_axis=log_x_axis,
            log_y_axis=log_y_axis,
        )
    else:
        # for consistency, for boxplot we don't aggregate
        grouped_stats = results.copy()
    
    if len(factors) < 2:
        # for consistency this forces FaceGrid to plot a single cell
        grouped_stats = grouped_stats.copy()
        grouped_stats['_single_facet'] = ''

        g = sns.FacetGrid(
            grouped_stats,
            col='_single_facet',
            height=height,
            aspect=aspect,
            sharey=True,
            sharex=True,
        )
    else:
        hue_variable = factors[0] if len(factors) >= 2 else None
        aggregate_x = factors[1] if len(factors) >= 2 else factors[0]
        aggregate_y = factors[2] if len(factors) >= 3 else None

        if aggregate_y:
            # for consistency, if only one aggregating variable plot a row
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
    
    new_y = y_axis + "_mean" if group_variables is True else y_axis
    new_bands = y_axis + "_sem" if (se_bands is True and group_variables is True) else None
    
    g.map_dataframe(
        plotting_function,
        x_axis=x_axis,
        y_axis=new_y,
        factors=factors,
        plot_bands=new_bands,
        **kwargs
    )
    # remove default x/y axis labels and tick labels from all subplots
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

    if len(factors) >= 2:
        # column facet titles
        for ax in range(g.axes.shape[1]):
            # put percentage sign for fraction variables
            
            if re.search(r'(?<![a-z])(?:percentage|fraction|prop)(?![a-z])', aggregate_x, re.IGNORECASE):
                title = (
                    f"{int(g.col_names[ax]*100)}% {name_conversion.get(aggregate_x, aggregate_x).replace('_', ' ').title()}"
                )
            else:
                title = f"{name_conversion.get(aggregate_x, aggregate_x).replace('_', ' ').title()}: {g.col_names[ax]}"
            g.axes[0, ax].set_title(title)
            
        # custom row facet labels 
        if aggregate_y is not None:
            for ax in range(g.axes.shape[0]):
                # put percentage sign for fraction variables
                if re.search(r'(?<![a-z])(?:percentage|fraction|prop)(?![a-z])', aggregate_y, re.IGNORECASE):
                    text = f"{int(g.row_names[ax]*100)}\\% {name_conversion.get(aggregate_y, aggregate_y).replace('_', ' ').title()}"
                else:
                    text = f"{name_conversion.get(aggregate_y, aggregate_y).replace('_', ' ').title()}: {g.row_names[ax]}"
                g.axes[ax, -1].texts[0].set_text(text)
    
    if add_legend is True:
        g.add_legend()
    
    # code to make the title centered above the grid not the legend
    plot_center_x = (g.axes[0, 0].get_position().x0 + g.axes[0, -1].get_position().x1) / 2
    if log_y_axis is True:
        g.figure.suptitle(
            "Log " + name_conversion.get(x_axis, x_axis) + " vs Log " + name_conversion.get(y_axis, y_axis)
            if log_x_axis
            else name_conversion.get(x_axis, x_axis) + " vs Log " + name_conversion.get(y_axis, y_axis),
            y=1.02,
            x=plot_center_x,
        )
    else:
        g.figure.suptitle(
            "Log " + name_conversion.get(x_axis, x_axis) + " vs " + name_conversion.get(y_axis, y_axis)
            if log_x_axis
            else name_conversion.get(x_axis, x_axis) + " vs " + name_conversion.get(y_axis, y_axis),
            y=1.02,
            x=plot_center_x,
        )
    
    if save_path is not None:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    return g


def plot_boxplot(
    x_axis, 
    y_axis, 
    **kwargs
):
    """
    Plot RMSE boxplot versus the given x variable. Optionally log-transform RMSE or x,
    and limit number of boxplots.
    """
    data = kwargs.pop("data")
    factors = kwargs.pop("factors", None)
    save_path = kwargs.pop("save_path", None)
    log_y_axis = kwargs.pop("log_y_axis", False)
    log_x_axis = kwargs.pop("log_x_axis", False)
    height = kwargs.pop("height", 4)
    aspect = kwargs.pop("aspect", 1.3)
    n_boxplots = kwargs.pop("n_boxplots", 5)
    colors = kwargs.pop("colors", None)
    
    ax = plt.gca()

    temp = data.copy()

    if log_x_axis is True:
        temp[x_axis] = np.log10(temp[x_axis])
    if log_y_axis is True:
        temp[y_axis] = np.log10(temp[y_axis])


    if n_boxplots < len(temp[x_axis].unique()):
        # Select n_boxplots evenly spaced along x
        df_values = sorted(temp[x_axis].unique())
        selected_dfs = np.linspace(0, len(df_values) - 1, n_boxplots, dtype=int)
        selected_dfs = [df_values[i] for i in selected_dfs]
        temp = temp[temp[x_axis].isin(selected_dfs)]
    
    hue_variable = None
    if factors is not None and len(factors) >= 1:
        hue_variable = factors[0]
    
    if hue_variable is not None:
        # Create palette from colors dict if provided
        palette = None
        if colors is not None:
            # Get unique hue values in the data
            hue_order = sorted(temp[hue_variable].unique())
            palette = [colors.get(hue_val, None) for hue_val in hue_order]
        
        sns.boxplot(data=temp, x=x_axis, y=y_axis, hue=hue_variable, palette=palette, ax=ax)
    else:
        sns.boxplot(data=temp, x=x_axis, y=y_axis, ax=ax)


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