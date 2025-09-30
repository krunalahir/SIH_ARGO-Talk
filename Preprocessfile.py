import xarray as xr
import pandas as pd
from pathlib import Path

# Directory where all floats and master file are stored
DATA_DIR = Path(__file__).resolve().parent / "data"
master_file = DATA_DIR / "argo_real_sample.csv"

def preprocess_single_float(file_path: Path, float_counter: int) -> pd.DataFrame:
    """Preprocess a single ARGO float NetCDF file into a clean dataframe."""
    ds = xr.open_dataset(file_path, engine="netcdf4")

    pres_var = "PRES_ADJUSTED" if "PRES_ADJUSTED" in ds else "PRES"
    temp_var = "TEMP_ADJUSTED" if "TEMP_ADJUSTED" in ds else "TEMP"
    psal_var = "PSAL_ADJUSTED" if "PSAL_ADJUSTED" in ds else "PSAL"

    df = ds[[pres_var, temp_var, psal_var, "LATITUDE", "LONGITUDE", "JULD"]].to_dataframe().reset_index()
    df = df.dropna(subset=[pres_var, temp_var, psal_var])

    # Rename columns
    df = df.rename(columns={
        pres_var: "pressure_mean",
        temp_var: "temperature_mean",
        psal_var: "salinity_mean",
        "LATITUDE": "lat",
        "LONGITUDE": "lon",
        "JULD": "date"
    })

    # Convert numeric columns
    for col in ["pressure_mean", "temperature_mean", "salinity_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Assign numeric float_id
    df["float_id"] = float_counter

    # Summary
    df["summary"] = df.apply(
        lambda row: (f"Profile at ({row['lat']:.2f}, {row['lon']:.2f}) on {row['date'].date()} - "
                     f"P={row['pressure_mean']} dbar, T={row['temperature_mean']} °C, "
                     f"S={row['salinity_mean']} PSU"), axis=1
    )

    return df

def update_master():
    """Detect all .nc files in data/ and update master CSV."""
    nc_files = list(DATA_DIR.glob("*.nc"))
    if not nc_files:
        print("❌ No .nc files found in 'data/' folder.")
        return

    all_dfs = []
    float_counter = 1

    for nc_file in nc_files:
        try:
            print(f"📂 Processing {nc_file.name} ...")
            df = preprocess_single_float(nc_file, float_counter)
            all_dfs.append(df)
            float_counter += 1
        except Exception as e:
            print(f"⚠️ Skipped {nc_file.name}: {e}")

    new_df = pd.concat(all_dfs, ignore_index=True)

    # Merge with master CSV
    if master_file.exists():
        master_df = pd.read_csv(master_file)
        # Ensure numeric columns and date parsing
        for col in ["pressure_mean", "temperature_mean", "salinity_mean"]:
            master_df[col] = pd.to_numeric(master_df[col], errors="coerce")
        master_df["date"] = pd.to_datetime(master_df["date"])
        master_df["float_id"] = pd.to_numeric(master_df["float_id"], errors="coerce")
        combined = pd.concat([master_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["float_id", "date", "pressure_mean"], keep="last")
    else:
        combined = new_df

    combined.to_csv(master_file, index=False)
    print(f"✅ Master file updated: {master_file} ({len(combined)} records)")
