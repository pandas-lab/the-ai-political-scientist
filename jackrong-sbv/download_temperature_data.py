"""
Download temperature anomaly data from Berkeley Earth
and merge with the UCDP GED data.
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO

def download_berkeley_earth_data():
    """Download global temperature anomaly data from Berkeley Earth."""

    # Berkeley Earth provides monthly land temperature anomalies
    # We'll use the global monthly data
    url = "https://berkeleyearth.org/data/BERIS_LandOnly_Trend_175001-202312.csv.gz"

    print("Downloading Berkeley Earth temperature data...")
    response = requests.get(url)

    # Parse the gzipped CSV
    import gzip
    with gzip.GzipFile(fileobj=StringIO(response.text)) as f:
        data = f.read().decode('utf-8')

    df = pd.read_csv(StringIO(data))
    return df

def download_nasa_gistemp():
    """Download NASA GISTEMP monthly temperature anomalies."""

    # NASA GISTEMP global monthly data - CSV format
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"

    print("Downloading NASA GISTEMP data...")
    response = requests.get(url)

    # Parse the CSV format (Year,Jan,Feb,...,Dec,...)
    lines = response.text.strip().split('\n')

    # Skip header and title lines
    data_lines = [l for l in lines if l and not l.startswith('Land-Ocean')]

    data = []
    for line in data_lines:
        parts = line.split(',')
        if len(parts) >= 13:
            try:
                year = int(parts[0])
                # Monthly anomalies (Jan-Dec are columns 1-12)
                for i in range(12):
                    val = parts[i + 1].strip()
                    if val and val not in ['---', '..', '***']:
                        month = i + 1
                        anomaly = float(val)
                        data.append({'year': year, 'month': month, 'anomaly': anomaly})
            except ValueError:
                continue

    return pd.DataFrame(data)

def merge_with_ged_data(temp_df, ged_path='cm.csv'):
    """Merge temperature data with GED data."""

    print("Loading GED data...")
    ged_df = pd.read_csv(ged_path)

    # The GED data already has year and month columns
    print(f"GED data year range: {ged_df['year'].min()} - {ged_df['year'].max()}")
    print(f"Temp data year range: {temp_df['year'].min()} - {temp_df['year'].max()}")

    # Merge on year and month - use suffixes to avoid column conflicts
    merged = ged_df.merge(temp_df, on=['year', 'month'], how='left', suffixes=('', '_temp'))

    return merged

if __name__ == "__main__":
    # Try NASA GISTEMP first (more reliable format)
    try:
        temp_df = download_nasa_gistemp()
        print(f"Downloaded {len(temp_df)} months of temperature data")
        print(temp_df.describe())

        # Merge and save
        merged = merge_with_ged_data(temp_df)
        merged.to_csv('cm_with_temp.csv', index=False)
        print(f"\nMerged data saved: {len(merged)} observations")

    except Exception as e:
        print(f"NASA download failed: {e}")
        print("Trying Berkeley Earth...")
        try:
            temp_df = download_berkeley_earth_data()
            print(f"Downloaded {len(temp_df)} observations")
        except Exception as e2:
            print(f"Berkeley Earth also failed: {e2}")
            print("Will use alternative approach...")
