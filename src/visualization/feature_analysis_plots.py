
import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT = "results/features/training_candidate_features.csv"
OUTPUT = "results/features/analysis"

os.makedirs(OUTPUT, exist_ok=True)

df = pd.read_csv(INPUT)

# 1. Distance distribution
plt.figure(figsize=(9, 5))
plt.hist(
    df["distance_to_arc_m"],
    bins=100,
    edgecolor="black"
)
plt.xlabel("Distance to road segment (metres)")
plt.ylabel("Frequency")
plt.title("Distribution of GPS-to-Road Distance")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT, "distance_distribution.png"),
    dpi=200
)
plt.close()

# 2. Distance distribution, zoomed to 200 metres
plt.figure(figsize=(9, 5))
plt.hist(
    df.loc[df["distance_to_arc_m"] <= 200, "distance_to_arc_m"],
    bins=80,
    edgecolor="black"
)
plt.xlabel("Distance to road segment (metres)")
plt.ylabel("Frequency")
plt.title("GPS-to-Road Distance (0–200 metres)")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT, "distance_distribution_zoomed.png"),
    dpi=200
)
plt.close()

# 3. Heading difference distribution
plt.figure(figsize=(9, 5))
plt.hist(
    df["heading_difference_deg"].dropna(),
    bins=72,
    edgecolor="black"
)
plt.xlabel("Heading difference (degrees)")
plt.ylabel("Frequency")
plt.title("Distribution of Heading Difference")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT, "heading_distribution.png"),
    dpi=200
)
plt.close()

# 4. Distance by candidate rank
plt.figure(figsize=(9, 5))
df.boxplot(
    column="distance_to_arc_m",
    by="candidate_rank",
    showfliers=False
)
plt.xlabel("Candidate rank")
plt.ylabel("Distance (metres)")
plt.title("Distance by Candidate Rank")
plt.suptitle("")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT, "distance_by_rank.png"),
    dpi=200
)
plt.close()

# 5. Feature correlation
numeric_features = [
    "distance_to_arc_m",
    "gps_heading_deg",
    "road_heading_deg",
    "heading_difference_deg",
    "candidate_rank"
]

correlation = df[numeric_features].corr()

plt.figure(figsize=(8, 6))
plt.imshow(
    correlation,
    vmin=-1,
    vmax=1,
    aspect="auto"
)
plt.colorbar(label="Correlation")
plt.xticks(
    range(len(numeric_features)),
    numeric_features,
    rotation=45,
    ha="right"
)
plt.yticks(
    range(len(numeric_features)),
    numeric_features
)
plt.title("Candidate Feature Correlation")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT, "feature_correlation.png"),
    dpi=200
)
plt.close()

print("FEATURE ANALYSIS PLOTS GENERATED")
print("Output folder:", OUTPUT)
print("Files:")
for filename in sorted(os.listdir(OUTPUT)):
    print("-", filename)
