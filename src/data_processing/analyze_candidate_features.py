
import pandas as pd
import numpy as np
import os

PATH = "results/features/training_candidate_features.csv"

df = pd.read_csv(PATH)

print("\n========== DATASET OVERVIEW ==========")
print("Rows:", len(df))
print("Trips:", df["trip_id"].nunique())
print("Unique observations:", df[
    ["trip_id", "observation_id"]
].drop_duplicates().shape[0])

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE CHECK ==========")
print(
    "Duplicate trip-observation-candidate rows:",
    df.duplicated(
        ["trip_id", "observation_id", "candidate_arc_id"]
    ).sum()
)

print("\n========== CANDIDATE COUNT ==========")
counts = df.groupby(
    ["trip_id", "observation_id"]
).size()

print(counts.value_counts().sort_index())

print("\n========== DISTANCE STATISTICS (METRES) ==========")
print(df["distance_to_arc_m"].describe(
    percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
))

print("\n========== HEADING DIFFERENCE ==========")
print(df["heading_difference_deg"].describe())

print("\n========== REFERENCE ROUTE MEMBERSHIP ==========")
print(df["is_in_reference_route"].value_counts())

print("\n========== DISTANCE BY CANDIDATE RANK ==========")
print(
    df.groupby("candidate_rank")["distance_to_arc_m"]
    .agg(["mean", "median", "min", "max"])
)

print("\n========== OBSERVATION COVERAGE ==========")
print("Observations with 5 candidates:", (counts == 5).sum())
print("Observations with fewer than 5:", (counts < 5).sum())
print("Maximum candidates:", counts.max())
print("Minimum candidates:", counts.min())

print("\nANALYSIS COMPLETED")
