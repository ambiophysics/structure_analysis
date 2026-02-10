#!/usr/bin/env python
# coding: utf-8

# In[1]:


import MDAnalysis as mda
import numpy as np
import csv
import os
from collections import defaultdict
from MDAnalysis.lib.distances import distance_array


# In[2]:


def load_universe(topology, trajectory):
    return mda.Universe(topology, trajectory)


# In[3]:


def define_regions(u):
    beta1AR_elements = {
        "I": u.select_atoms("bynum 738-1181"),

    }

    gas_elements = {
        "H5": u.select_atoms("bynum 13185-13622"),

    }

    return beta1AR_elements, gas_elements


# In[4]:


def build_residue_map(atomgroup):
    """
    Maps atom indices → residue indices
    """
    residues = atomgroup.residues
    atom_indices = atomgroup.atoms.indices

    atom_to_res = np.empty(len(atom_indices), dtype=np.int32)
    res_info = []

    for i, res in enumerate(residues):
        mask = np.isin(atom_indices, res.atoms.indices)
        atom_to_res[mask] = i
        res_info.append((res.resid, res.resname))

    return atom_to_res, res_info


# In[5]:


def compute_contacts_fast(
    u,
    beta_regions,
    gas_regions,
    cutoff_nm=0.7,
):
    cutoff = cutoff_nm * 10.0  # nm → Å
    contact_counter = defaultdict(int)
    total_frames = 0

    # Precompute residue maps ONCE
    beta_maps = {
        k: build_residue_map(v) for k, v in beta_regions.items()
    }
    gas_maps = {
        k: build_residue_map(v) for k, v in gas_regions.items()
    }

    for ts in u.trajectory:
        total_frames += 1

        for b_name, b_atoms in beta_regions.items():
            b_map, b_resinfo = beta_maps[b_name]

            for g_name, g_atoms in gas_regions.items():
                g_map, g_resinfo = gas_maps[g_name]

                # Single distance matrix per region pair
                dmat = distance_array(
                    b_atoms.positions,
                    g_atoms.positions,
                    box=u.dimensions,
                )

                contact_atoms = np.where(dmat <= cutoff)
                if contact_atoms[0].size == 0:
                    continue

                # Reduce atom contacts → residue contacts
                b_res_ids = b_map[contact_atoms[0]]
                g_res_ids = g_map[contact_atoms[1]]

                unique_pairs = set(zip(b_res_ids, g_res_ids))

                for bi, gi in unique_pairs:
                    b_resid, b_resname = b_resinfo[bi]
                    g_resid, g_resname = g_resinfo[gi]

                    key = (
                        b_name,
                        g_name,
                        b_resid,
                        b_resname,
                        g_resid,
                        g_resname,
                    )
                    contact_counter[key] += 1

    return contact_counter, total_frames


# In[6]:


def filter_contacts(counter, total_frames, threshold=0.9):
    results = []

    for key, count in counter.items():
        occupancy = count / total_frames
        if occupancy >= threshold:
            results.append((*key, count, occupancy))

    return results


# In[7]:


def write_csv(output_csv, contacts):
    header = [
        "beta_region",
        "gas_region",
        "beta_resid",
        "beta_resname",
        "gas_resid",
        "gas_resname",
        "contact_frames",
        "occupancy",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(contacts)


# In[8]:


def main(topology, trajectory, output_csv):
    u = load_universe(topology, trajectory)
    beta_regions, gas_regions = define_regions(u)

    counter, nframes = compute_contacts_fast(
        u,
        beta_regions,
        gas_regions,
        cutoff_nm=0.7,
    )

    persistent_contacts = filter_contacts(counter, nframes)
    write_csv(output_csv, persistent_contacts)

    print(f"Frames analysed: {nframes}")
    print(f"Persistent contacts: {len(persistent_contacts)}")


# In[9]:


if __name__ == "__main__":
    main(
        topology="structure.gro",
        trajectory="structure.xtc",
        output_csv="output.csv",
    )


# In[ ]:



