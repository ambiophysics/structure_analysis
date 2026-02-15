#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis import distances


# In[2]:


u = mda.Universe("structure.gro", "structure.xtc")


# In[3]:


groups = {
    "K347_6x58": "index 5325:5346 and name CA",
    "F359_7x35": "index 5537:5556 and name CA",
}


# In[4]:


atomgroups = {name: u.select_atoms(sel) for name, sel in groups.items()}


# In[5]:


group_pairs = [
    ("K347_6x58", "F359_7x35"),
]


# In[6]:


results = {}

for g1_name, g2_name in group_pairs:
    g1 = atomgroups[g1_name]
    g2 = atomgroups[g2_name]

    dist_list = []

    for ts in u.trajectory[::10]:
        dmat = distances.distance_array(
            g1.positions,
            g2.positions,
            box=u.dimensions
        )
        dist_list.append(dmat)

    dist_arr = np.array(dist_list)       # (n_frames, n1, n2)
    dist_arr = np.round(dist_arr, 3) / 10.0  # Å → nm

    results[(g1_name, g2_name)] = dist_arr


# In[10]:


for (g1, g2), dist_arr in results.items():

    dist_ts = dist_arr.min(axis=(1, 2))  # or mean

    fname = f"{g1}-{g2}-dist.txt"
    np.savetxt(fname, dist_ts)

    plt.figure(figsize=(13, 6))
    plt.margins(x=0)
    plt.ylim(0.5, 2.2)
    plt.xlabel("Time (ns)")
    plt.ylabel("Distance (nm)")
    plt.title(f"{g1}–{g2} Distance (nm)")
    plt.plot(dist_ts)
    plt.tight_layout()
    plt.savefig(f"{g1}-{g2}-dist.png", dpi=300)
    plt.close()


# In[12]:


all_dist_ts = {}   # store time series for combined plots

for (g1, g2), dist_arr in results.items():

    dist_ts = dist_arr.min(axis=(1, 2))  # or mean
    label = f"{g1}-{g2}"
    all_dist_ts[label] = dist_ts

    # ---- Save distance time series ----
    np.savetxt(f"{label}-dist.txt", dist_ts)

    # ---- Time series plot ----
    plt.figure(figsize=(13, 2))
    plt.margins(x=0)
    plt.ylim(1.4, 2.3)
    plt.xlabel("Time (ns)")
    plt.ylabel("Distance (nm)")
    plt.title(f"{label} Distance (nm)")
    plt.plot(dist_ts)
    plt.tight_layout()
    plt.savefig(f"{label}-timeseries.png", dpi=300)
    plt.close()

    # ---- KDE plot ----
    plt.figure(figsize=(4, 3))
    sns.kdeplot(dist_ts, fill=True, bw_adjust=0.8)
    plt.xlabel("Distance (nm)")
    plt.ylabel("Density")
    plt.title(f"{label} KDE")
    plt.tight_layout()
    plt.savefig(f"{label}-kde.png", dpi=300)
    plt.close()


# In[19]:


plt.figure(figsize=(18, 6))

for label, dist_ts in all_dist_ts.items():
    plt.plot(dist_ts, label=label, linewidth=1.5)

plt.margins(x=0)
plt.xlabel("Time (ns)")
plt.ylabel("Distance (nm)")
plt.ylim(0.5,2.5)
plt.title("Minimum Distance Time Series (All Group Pairs)")
plt.legend(frameon=False, fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig("All-group_pairs-timeseries.png", dpi=300)
plt.close()


# In[15]:


plt.figure(figsize=(5, 4))

for label, dist_ts in all_dist_ts.items():
    sns.kdeplot(dist_ts, label=label, bw_adjust=0.8)

plt.xlabel("Distance (nm)")
plt.ylabel("Density")
plt.title("Distance KDE (All Group Pairs)")
plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig("All-group_pairs-kde.png", dpi=300)
plt.close()


# In[ ]:



