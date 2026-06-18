import matplotlib.pyplot as plt

# ---------------------------
# SYSTEM NAMES
# ---------------------------
systems = ["Portal", "BM25", "Metadata", "Fuzzy HCIR"]

# ---------------------------
# METRICS FROM YOUR RESULTS
# ---------------------------
map_scores = [0.7328, 0.5926, 0.7263, 0.5988]
ndcg_scores = [0.6678, 0.6033, 0.6368, 0.5661]
mrr_scores = [0.8556, 0.8444, 0.8667, 0.8778]

# ---------------------------
# PLOT SETUP
# ---------------------------
x = range(len(systems))

plt.figure(figsize=(10,6))

plt.plot(x, map_scores, marker='o', label='MAP')
plt.plot(x, ndcg_scores, marker='o', label='nDCG@10')
plt.plot(x, mrr_scores, marker='o', label='MRR')

plt.xticks(x, systems)

plt.title("Figure 4: Retrieval Performance Comparison")
plt.xlabel("Retrieval Methods")
plt.ylabel("Score")

plt.legend()
plt.grid(True)

plt.tight_layout()

# ---------------------------
# SAVE FIGURE
# ---------------------------
plt.savefig("evaluation/figure4.png", dpi=300)

plt.show()