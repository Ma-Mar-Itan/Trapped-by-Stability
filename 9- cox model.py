import pandas as pd

from lifelines import CoxPHFitter


# HR+/HER2+ SUBGROUP


subtype_col = "Mol Subtype"

hrher2_labels = [
    "HR+/HER2+",
    "HRpos_HER2pos",
    "ER/PR pos, HER2 pos",
    "ER/PR pos HER2 pos",
    1
]

hrher2 = df[df[subtype_col].isin(hrher2_labels)].copy()

print("HR+/HER2+ size:", hrher2.shape)

# REBUILD SURVIVAL TIME VARIABLES


event_col = "Recurrence event(s)"

local_recur_col = "Days to local recurrence (from the date of diagnosis) "
distant_recur_col = "Days to distant recurrence(from the date of diagnosis) "

last_local_col = "Days to last local recurrence free assessment (from the date of diagnosis) "
last_distant_col = "Days to last distant recurrence free assemssment(from the date of diagnosis) "

# numeric conversion
for col in [
    local_recur_col,
    distant_recur_col,
    last_local_col,
    last_distant_col,
    event_col
]:
    hrher2[col] = pd.to_numeric(
        hrher2[col],
        errors="coerce"
    )

# recurrence time
hrher2["recurrence_time_days"] = hrher2[
    [local_recur_col, distant_recur_col]
].min(axis=1)

# censoring/follow-up time
hrher2["followup_time_days"] = hrher2[
    [last_local_col, last_distant_col]
].max(axis=1)

# final RFS time
hrher2["rfs_time_days"] = np.where(
    hrher2[event_col] == 1,
    hrher2["recurrence_time_days"],
    hrher2["followup_time_days"]
)

# convert to years
hrher2["rfs_time_years"] = (
    hrher2["rfs_time_days"] / 365.25
)

# remove invalid rows
hrher2 = hrher2[
    hrher2["rfs_time_years"].notna()
]

hrher2 = hrher2[
    hrher2["rfs_time_years"] > 0
]

print(hrher2.shape)

# KEEP NEEDED VARIABLES


cox_df = hrher2[
    [
        "rfs_time_years",
        "Recurrence event(s)",
        "Cluster",
        "Manufacturer_clean"
    ]
].copy()


# CLEAN VARIABLES


cox_df = cox_df.dropna()

cox_df["Cluster"] = cox_df["Cluster"].astype(int)


# ONE-HOT ENCODE MANUFACTURER


cox_df = pd.get_dummies(
    cox_df,
    columns=["Manufacturer_clean"],
    drop_first=True
)

print("\nCox dataframe columns:")
print(cox_df.columns)

print("\nFinal shape:")
print(cox_df.shape)


# FIT COX MODEL


cph = CoxPHFitter(
    penalizer=0.01
)

cph.fit(
    cox_df,
    duration_col="rfs_time_years",
    event_col="Recurrence event(s)"
)


# RESULTS


summary = cph.summary[
    [
        "coef",
        "exp(coef)",
        "p",
        "exp(coef) lower 95%",
        "exp(coef) upper 95%"
    ]
]

print("\nManufacturer-adjusted Cox model:")
print(summary)
