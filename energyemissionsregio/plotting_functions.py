import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import SymLogNorm
from cmcrameri import cm


def plot_proxy_data_single_country(data_df, gdf, proxy_var_unit, save_path):
    merged_df = pd.merge(
        gdf, data_df, left_on="code", right_on="region_code", how="left"
    )

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(1, 1, wspace=0.1, hspace=0)

    vmin, vmax = min(merged_df["value"]), max(merged_df["value"])
    norm = SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax)

    ax1 = plt.subplot(gs[:, :])

    merged_df.plot(
        column="value",
        cmap=cm.batlow_r,
        linewidth=0.8,
        ax=ax1,
        edgecolor="none",
        norm=norm,
    )
    ax1.axis("off")

    # legend
    sm = plt.cm.ScalarMappable(cmap=cm.batlow_r, norm=norm)
    sm._A = []
    axins = inset_axes(ax1, width="90%", height="5%", loc="lower center", borderpad=-2)
    clb = fig.colorbar(sm, cax=axins, orientation="horizontal", shrink=0.8)
    clb.ax.tick_params(labelsize=15)

    clb.ax.set_title(f"({proxy_var_unit})", fontsize=15)

    # plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize=15)

    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=200)


def plot_proxy_data_both_countries(data_df, gdf, proxy_var_unit, save_path):
    merged_df = pd.merge(
        gdf, data_df, left_on="code", right_on="region_code", how="left"
    )

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(1, 2, wspace=0.1, hspace=0)

    vmin, vmax = min(merged_df["value"]), max(merged_df["value"])
    norm = SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax)

    # Germany --------
    ax1 = plt.subplot(gs[:, :1])

    merged_df_de = merged_df[merged_df["code"].str.startswith("DE")]
    merged_df_de.plot(
        column="value",
        cmap=cm.batlow_r,
        linewidth=0.8,
        ax=ax1,
        edgecolor="none",
        norm=norm,
    )
    ax1.axis("off")

    # Spain --------
    ax2 = plt.subplot(gs[:, 1:])

    merged_df_es = merged_df[merged_df["code"].str.startswith("ES")]
    merged_df_es.plot(
        column="value",
        cmap=cm.batlow_r,
        linewidth=0.8,
        ax=ax2,
        edgecolor="none",
        norm=norm,
    )
    ax2.axis("off")

    # legend
    sm = plt.cm.ScalarMappable(cmap=cm.batlow_r, norm=norm)
    sm._A = []
    axins = inset_axes(ax2, width="90%", height="5%", loc="lower center", borderpad=-2)
    clb = fig.colorbar(sm, cax=axins, orientation="horizontal", shrink=0.8)
    clb.ax.tick_params(labelsize=15)

    clb.ax.set_title(f"({proxy_var_unit})", fontsize=15)

    # plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, fontsize=15)

    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=200)


def plot_solved_proxy_data_single_country(data_df, gdf, save_path):
    merged_df = pd.merge(
        gdf, data_df, left_on="code", right_on="region_code", how="left"
    )

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(1, 1, wspace=0.1, hspace=0)

    # Germany --------
    ax1 = plt.subplot(gs[:, :])

    merged_df.plot(
        column="value", cmap=cm.batlow_r, linewidth=0.8, ax=ax1, edgecolor="none"
    )
    ax1.axis("off")

    # legend
    sm = plt.cm.ScalarMappable(cmap=cm.batlow_r)
    sm._A = []
    axins = inset_axes(ax1, width="90%", height="5%", loc="lower center", borderpad=-2)
    clb = fig.colorbar(sm, cax=axins, orientation="horizontal", shrink=0.8)
    clb.ax.tick_params(labelsize=15)

    clb.ax.set_title(f"", fontsize=15)

    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=200)


def plot_solved_proxy_data_both_countries(data_df, gdf, save_path):
    merged_df = pd.merge(
        gdf, data_df, left_on="code", right_on="region_code", how="left"
    )

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(1, 2, wspace=0.1, hspace=0)

    # Germany --------
    ax1 = plt.subplot(gs[:, :1])

    merged_df_de = merged_df[merged_df["code"].str.startswith("DE")]
    merged_df_de.plot(
        column="value", cmap=cm.batlow_r, linewidth=0.8, ax=ax1, edgecolor="none"
    )
    ax1.axis("off")

    # Spain --------
    ax2 = plt.subplot(gs[:, 1:])

    merged_df_es = merged_df[merged_df["code"].str.startswith("ES")]
    merged_df_es.plot(
        column="value", cmap=cm.batlow_r, linewidth=0.8, ax=ax2, edgecolor="none"
    )
    ax2.axis("off")

    # legend
    sm = plt.cm.ScalarMappable(cmap=cm.batlow_r)
    sm._A = []
    axins = inset_axes(ax2, width="90%", height="5%", loc="lower center", borderpad=-2)
    clb = fig.colorbar(sm, cax=axins, orientation="horizontal", shrink=0.8)
    clb.ax.tick_params(labelsize=15)

    clb.ax.set_title(f"", fontsize=15)

    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=200)


def plot_target_data_single_country(
    data_df_true, data_df_disagg, gdf_true, gdf_disagg, target_var_unit, save_path
):
    merged_df_true = pd.merge(
        gdf_true, data_df_true, left_on="code", right_on="region_code", how="left"
    )

    merged_df_disagg = pd.merge(
        gdf_disagg, data_df_disagg, left_on="code", right_on="region_code", how="left"
    )

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(1, 2, wspace=0.1, hspace=0)

    vmin, vmax = min(merged_df_disagg["value"]), max(merged_df_true["value"])
    norm = SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax)

    # True --------
    ax1 = plt.subplot(gs[:, :1])

    merged_df_true.plot(
        column="value",
        cmap=cm.batlow_r,
        linewidth=0.8,
        ax=ax1,
        edgecolor="none",
        norm=norm,
    )
    ax1.axis("off")

    #  Disagg --------
    ax2 = plt.subplot(gs[:, 1:])

    merged_df_disagg.plot(
        column="value",
        cmap=cm.batlow_r,
        linewidth=0.8,
        ax=ax2,
        edgecolor="none",
        norm=norm,
    )
    ax2.axis("off")

    # legend
    sm = plt.cm.ScalarMappable(cmap=cm.batlow_r, norm=norm)
    sm._A = []
    axins = inset_axes(ax2, width="90%", height="5%", loc="lower center", borderpad=-2)
    clb = fig.colorbar(sm, cax=axins, orientation="horizontal", shrink=0.8)
    clb.ax.tick_params(labelsize=15)

    clb.ax.set_title(f"[{target_var_unit}]", fontsize=15)

    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=200)


def plot_target_data_both_countries(
    data_df_true, data_df_disagg, gdf_true, gdf_disagg, target_var_unit, save_path
):
    merged_df_true = pd.merge(
        gdf_true, data_df_true, left_on="code", right_on="region_code", how="left"
    )

    merged_df_disagg = pd.merge(
        gdf_disagg, data_df_disagg, left_on="code", right_on="region_code", how="left"
    )

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(2, 2, wspace=0.1, hspace=0)

    vmin, vmax = min(merged_df_disagg["value"]), max(merged_df_true["value"])
    norm = SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax)

    # Germany - True --------
    ax1 = plt.subplot(gs[:1, :1])

    merged_df_de = merged_df_true[merged_df_true["code"].str.startswith("DE")]
    merged_df_de.plot(
        column="value",
        cmap=cm.batlow_r,
        linewidth=0.8,
        ax=ax1,
        edgecolor="none",
        norm=norm,
    )
    ax1.axis("off")

    # Spain - True --------
    ax2 = plt.subplot(gs[:1, 1:])

    merged_df_es = merged_df_true[merged_df_true["code"].str.startswith("ES")]
    merged_df_es.plot(
        column="value",
        cmap=cm.batlow_r,
        linewidth=0.8,
        ax=ax2,
        edgecolor="none",
        norm=norm,
    )
    ax2.axis("off")

    # Germany - Disagg --------
    ax3 = plt.subplot(gs[1:, :1])

    merged_df_de = merged_df_disagg[merged_df_disagg["code"].str.startswith("DE")]
    merged_df_de.plot(
        column="value",
        cmap=cm.batlow_r,
        linewidth=0.8,
        ax=ax3,
        edgecolor="none",
        norm=norm,
    )
    ax3.axis("off")

    # Spain - Disagg --------
    ax4 = plt.subplot(gs[1:, 1:])

    merged_df_es = merged_df_disagg[merged_df_disagg["code"].str.startswith("ES")]
    merged_df_es.plot(
        column="value",
        cmap=cm.batlow_r,
        linewidth=0.8,
        ax=ax4,
        edgecolor="none",
        norm=norm,
    )
    ax4.axis("off")

    # legend
    sm = plt.cm.ScalarMappable(cmap=cm.batlow_r, norm=norm)
    sm._A = []
    axins = inset_axes(ax4, width="90%", height="5%", loc="lower center", borderpad=-2)
    clb = fig.colorbar(sm, cax=axins, orientation="horizontal", shrink=0.8)
    clb.ax.tick_params(labelsize=15)

    clb.ax.set_title(f"({target_var_unit})", fontsize=15)

    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=200)


def plot_nuts0_data(
    de_true_value, es_true_value, data_df_disagg, gdf_disagg, target_var_unit, save_path
):

    merged_df_disagg = pd.merge(
        gdf_disagg, data_df_disagg, left_on="code", right_on="region_code", how="left"
    )

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(1, 2, wspace=0.1, hspace=0)

    vmin, vmax = min(merged_df_disagg["value"]), max(merged_df_disagg["value"])
    norm = SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax)

    # Germany - Disagg --------
    ax1 = plt.subplot(gs[:, :1])

    merged_df_de = merged_df_disagg[merged_df_disagg["code"].str.startswith("DE")]
    merged_df_de.plot(
        column="value",
        cmap=cm.batlow_r,
        linewidth=0.8,
        ax=ax1,
        edgecolor="none",
        norm=norm,
    )
    ax1.axis("off")
    ax1.set_title(f"Country total: {de_true_value} {target_var_unit}", fontsize=15)

    # Spain - Disagg --------
    ax2 = plt.subplot(gs[:, 1:])

    merged_df_es = merged_df_disagg[merged_df_disagg["code"].str.startswith("ES")]
    merged_df_es.plot(
        column="value",
        cmap=cm.batlow_r,
        linewidth=0.8,
        ax=ax2,
        edgecolor="none",
        norm=norm,
    )
    ax2.axis("off")
    ax2.set_title(f"Country total: {es_true_value} {target_var_unit}", fontsize=15)

    # legend
    sm = plt.cm.ScalarMappable(cmap=cm.batlow_r, norm=norm)

    sm._A = []
    axins = inset_axes(ax2, width="90%", height="5%", loc="lower center", borderpad=-2)
    clb = fig.colorbar(sm, cax=axins, orientation="horizontal", shrink=0.8)
    clb.ax.tick_params(labelsize=15)

    clb.ax.set_title(f"({target_var_unit})", fontsize=15)

    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=200)


def plot_validation_data(
    true_df,
    disagg_df,
    gdf,
    de_total,
    es_total,
    mae_de,
    mae_es,
    target_var_unit,
    true_data_source,
    save_path,
):

    fig = plt.figure(figsize=(13, 13))
    gs = fig.add_gridspec(2, 2, wspace=0.1, hspace=0)

    merged_df_true = pd.merge(
        gdf, true_df, left_on="code", right_on="region_code", how="left"
    )

    merged_df_disagg = pd.merge(
        gdf, disagg_df, left_on="code", right_on="region_code", how="left"
    )

    vmin = np.minimum(true_df["value"].min(), disagg_df["value"].min())
    vmax = np.maximum(true_df["value"].max(), disagg_df["value"].max())

    norm = SymLogNorm(linthresh=0.1, vmin=vmin, vmax=vmax)

    # Germany - True --------
    ax1 = plt.subplot(gs[:1, :1])

    merged_df_de = merged_df_true[merged_df_true["code"].str.startswith("DE")]

    merged_df_de.plot(
        column="value",
        cmap=cm.batlow_r,
        linewidth=0.8,
        ax=ax1,
        edgecolor="none",
        norm=norm,
    )
    # Hide spines (the borders of the plot)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)

    # Hide ticks and tick labels
    ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax1.set_ylabel(f"{true_data_source} values", fontsize=15)
    ax1.set_title(f"Germany\nCountry total = {de_total} {target_var_unit}", fontsize=15)

    # Spain - True --------
    ax2 = plt.subplot(gs[:1, 1:])

    merged_df_es = merged_df_true[merged_df_true["code"].str.startswith("ES")]

    merged_df_es.plot(
        column="value",
        cmap=cm.batlow_r,
        linewidth=0.8,
        ax=ax2,
        edgecolor="none",
        norm=norm,
    )

    ax2.set_title(f"Spain\nCountry total = {es_total} {target_var_unit}", fontsize=15)
    ax2.axis("off")

    # Germany - Disagg --------
    ax3 = plt.subplot(gs[1:, :1])

    merged_df_de = merged_df_disagg[merged_df_disagg["code"].str.startswith("DE")]
    merged_df_de.plot(
        column="value",
        cmap=cm.batlow_r,
        linewidth=0.8,
        ax=ax3,
        edgecolor="none",
        norm=norm,
    )
    # Hide spines (the borders of the plot)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)

    # Hide ticks and tick labels
    ax3.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax3.set_ylabel("Disaggregated values", fontsize=15)
    ax3.set_xlabel(f"RMSE = {mae_de} {target_var_unit}", fontsize=15)

    # Spain - Disagg --------
    ax4 = plt.subplot(gs[1:, 1:])

    merged_df_es = merged_df_disagg[merged_df_disagg["code"].str.startswith("ES")]
    merged_df_es.plot(
        column="value",
        cmap=cm.batlow_r,
        linewidth=0.8,
        ax=ax4,
        edgecolor="none",
        norm=norm,
    )

    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.spines["left"].set_visible(False)
    ax4.spines["bottom"].set_visible(False)
    ax4.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax4.set_xlabel(f"RMSE = {mae_es} {target_var_unit}", fontsize=15)

    # legend
    # Add colorbar legend outside the plot
    sm = plt.cm.ScalarMappable(cmap=cm.batlow_r, norm=norm)
    sm._A = []
    axins = inset_axes(
        ax4,
        width="60%",
        height="20%",
        loc="lower center",
        bbox_to_anchor=(0.1, 0.05, 0.9, 0.05),  # Adjust for position
        bbox_transform=fig.transFigure,
        borderpad=0,
    )
    clb = fig.colorbar(sm, cax=axins, orientation="horizontal", shrink=0.8)
    clb.ax.tick_params(labelsize=15)
    clb.ax.set_title(f"({target_var_unit})", fontsize=15)

    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=200)
