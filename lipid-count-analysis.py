#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# User inputs
# ============================================================
lipid_types = [
    "POPC", "CHOL"
]

# Bulk concentrations
bulk_conc_u = [611, 283, 21, 0, 0, 0, 0, 0, 87]
bulk_conc_l = [354, 230, 194, 35, 35, 35, 35, 35, 0]

upper_regions = [f"{i}" for i in range(1, 8)]
lower_regions = [f"{i}" for i in range(1, 8)]

input_csv = "output.csv"
output_csv = "output_analysis.csv"
plot_dir = "plots"

os.makedirs(plot_dir, exist_ok=True)

# ============================================================
# Functions
# ============================================================
def compute_cbulk(lipid_types, bulk_conc):
    """Compute bulk lipid fractions"""
    total_bulk = sum(bulk_conc)
    return {
        lipid: (conc / total_bulk) if conc > 0 else 0.0
        for lipid, conc in zip(lipid_types, bulk_conc)
    }


def compute_clocal(df_region):
    """
    Compute local lipid fractions.
    Explicitly force numeric conversion to avoid string division errors.
    """
    df_region = df_region.copy()

    # 🔑 CRITICAL FIX: force numeric dtype
    df_region["Average_Contacts"] = pd.to_numeric(
        df_region["Average_Contacts"],
        errors="coerce"
    )

    total_contacts = df_region["Average_Contacts"].sum()

    if total_contacts == 0 or np.isnan(total_contacts):
        df_region["Clocal"] = 0.0
    else:
        df_region["Clocal"] = df_region["Average_Contacts"] / total_contacts

    return df_region


def compute_deindex(df_region, cbulk_dict):
    """Compute de-index with zero-bulk handling"""
    df_region = df_region.copy()
    df_region["Cbulk"] = df_region["Lipid_Type"].map(cbulk_dict)

    df_region["DeIndex"] = np.where(
        df_region["Cbulk"] > 0,
        df_region["Clocal"] / df_region["Cbulk"],
        0.0
    )
    return df_region


def get_bulk_dict_for_region(region):
    """Return leaflet-specific bulk concentrations"""
    if region in upper_regions:
        return compute_cbulk(lipid_types, bulk_conc_u)
    elif region in lower_regions:
        return compute_cbulk(lipid_types, bulk_conc_l)
    else:
        raise ValueError(f"Unknown protein region: {region}")


def plot_barplots(df):
    """Bar plots of de-index per protein region"""
    for region in df["Protein_Region"].unique():
        sub = df[df["Protein_Region"] == region]

        plt.figure(figsize=(8, 4))
        sns.barplot(data=sub, x="Lipid_Type", y="DeIndex")
        plt.title(f"De-index – {region}")
        plt.ylabel("De-index")
        plt.xlabel("Lipid Type")
        plt.xticks()
        plt.tight_layout()
        plt.grid()
        plt.savefig(f"{plot_dir}/deindex_bar_{region}.png", dpi=300)
        plt.close()


def plot_heatmap(df):
    """Heatmap of de-index values"""
    pivot = df.pivot(
        index="Protein_Region",
        columns="Lipid_Type",
        values="DeIndex"
    )

    plt.figure(figsize=(11, 6))
    sns.heatmap(
        pivot,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("Leaflet-Specific De-index Heatmap")
    plt.ylabel("Protein Region")
    plt.xlabel("Lipid Type")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/deindex_heatmap.png", dpi=300)
    plt.close()


# ============================================================
# Main workflow
# ============================================================
def main():
    # Read CSV
    df = pd.read_csv(input_csv)

    # Clean whitespace (extra safety)
    df["Protein_Region"] = df["Protein_Region"].str.strip()
    df["Lipid_Type"] = df["Lipid_Type"].str.strip()

    results = []

    for region in df["Protein_Region"].unique():
        df_region = df[df["Protein_Region"] == region]

        # Compute Clocal (safe numeric handling)
        df_region = compute_clocal(df_region)

        # Leaflet-specific bulk
        cbulk_dict = get_bulk_dict_for_region(region)

        # Compute de-index
        df_region = compute_deindex(df_region, cbulk_dict)

        results.append(df_region)

    df_final = pd.concat(results, ignore_index=True)

    # Save CSV
    df_final.to_csv(output_csv, index=False)

    # Plots
    plot_barplots(df_final)
    plot_heatmap(df_final)

    print("Leaflet-specific de-index calculation complete")
    print(f"Results saved to: {output_csv}")
    print(f"Plots saved in: {plot_dir}/")


if __name__ == "__main__":
    main()




