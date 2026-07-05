import pandas as pd
import numpy as np

data = pd.read_csv("dataset/PTSD_research.csv")

data["PTSD"] = (data["pcl_total"] >= 33).astype(int)

print("Before:")
print(data["PTSD"].value_counts())

# ================= PTSD CASES =================
ptsd_cases = []

for _ in range(200):
    row = {}

    for col in data.columns:
        if col == "PTSD":
            continue

        if col in [
            "recent_accident","family_accident",
            "witnessed_serious_injury","witnessed_corpse",
            "traumatic_scene","psychiatric_history"
        ]:
            row[col] = np.random.choice([0,1], p=[0.3, 0.7])  # mostly yes

        elif col == "psychological_resilience":
            row[col] = np.random.randint(10, 50)

        else:
            row[col] = data[col].sample().values[0]

    row["PTSD"] = 1
    ptsd_cases.append(row)

ptsd_df = pd.DataFrame(ptsd_cases)

# ================= NON-PTSD CASES WITH TRAUMA (VERY IMPORTANT) =================
safe_cases = []

for _ in range(300):
    row = {}

    for col in data.columns:
        if col == "PTSD":
            continue

        if col in [
            "recent_accident","family_accident",
            "witnessed_serious_injury","witnessed_corpse",
            "traumatic_scene"
        ]:
            row[col] = np.random.choice([0,1], p=[0.5, 0.5])  # MIXED

        elif col == "psychological_resilience":
            row[col] = np.random.randint(60, 100)  # HIGH resilience

        elif col == "psychiatric_history":
            row[col] = 0  # mostly no

        else:
            row[col] = data[col].sample().values[0]

    row["PTSD"] = 0
    safe_cases.append(row)

safe_df = pd.DataFrame(safe_cases)

# ================= FINAL =================
final = pd.concat([data, ptsd_df, safe_df], ignore_index=True)

print("\nAfter:")
print(final["PTSD"].value_counts())

final.to_csv("dataset/PTSD_research.csv", index=False)

print("\n✅ FIXED DATASET CREATED")