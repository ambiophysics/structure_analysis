import MDAnalysis as mda
from MDAnalysis.analysis import distances
import numpy as np
import csv

results = [("Protein_Region", "Lipid_Type", "Average_Contacts")]

def analyze_lipid_contact(tpr, xtc, region_name, protein_sel, lipid_names, lipid_group_name=None, cutoff=5.5):
    u = mda.Universe(tpr, xtc)
    protein = u.select_atoms(protein_sel)

    # Build lipid selection string
    if isinstance(lipid_names, list):
        lipid_sel = "resname " + " ".join(lipid_names)
    else:
        lipid_sel = f"resname {lipid_names}"

    lipid_atoms = u.select_atoms(lipid_sel)
    near_lipid_per_frame = []

    for ts in u.trajectory[10000:35000]:
        dist_matrix = distances.distance_array(protein.positions, lipid_atoms.positions, box=u.dimensions)
        close_atom_indices = np.where(dist_matrix < cutoff)[1]
        close_resids = set(lipid_atoms[close_atom_indices].resids)
        near_lipid_per_frame.append(close_resids)

        lipid_label = lipid_group_name if lipid_group_name else (lipid_names if isinstance(lipid_names, str) else ", ".join(lipid_names))
        #print(f"Frame {ts.frame} | {region_name} | {lipid_label}: {len(close_resids)} residues near protein")

    avg = np.mean([len(r) for r in near_lipid_per_frame])
    print(f"Average number of {lipid_group_name or lipid_names} residues near protein region {region_name}: {avg:.2f}\n")
    results.append((region_name, lipid_group_name or lipid_names, avg))


# === Input parameters ===
TPR = "trajs/inp.tpr"
XTC = "trajs/inp.xtc"
CUTOFF = 5.5  # Å

# === Protein regions ===
protein_regions = {
    "A": "resid x-y",
    "B": "resid e-f",
    "C": "resid p-q",
}

# === Lipid types to analyze ===
lipid_sets = [
    ("POPC", "POPC"),
    ("CHOL", "CHOL"),
]

# === Run analysis for each protein region and lipid type ===
for region_name, protein_sel in protein_regions.items():
    for lipid_names, lipid_group_name in lipid_sets:
        analyze_lipid_contact(TPR, XTC, region_name, protein_sel, lipid_names, lipid_group_name, cutoff=CUTOFF)


import csv

output_file = "output.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Protein_Region", "Lipid_Type", "Average_Contacts"])
    for row in results:
        writer.writerow([row[0], row[1], row[2]])

print(f"CSV written to: {output_file}")
