"""
BLS CES Average Hourly Earnings Data Fetcher
Pulls monthly average hourly earnings (2022-2025) for Total Private 
and key NAICS supersectors from the BLS API.
"""

import requests
import pandas as pd
from datetime import datetime


# BLS API endpoint (v2)
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# CES Series IDs for Average Hourly Earnings (data type code 03)
# Format: CES + supersector code + 0000000 + data type (03)
SERIES_IDS = {
    "CES0500000003": "Total Private",
    "CES5000000003": "Information",
    "CES6000000003": "Professional & Business Services",
    "CES5500000003": "Financial Activities",
    "CES6500000003": "Education & Health Services",
    "CES7000000003": "Leisure & Hospitality",
    "CES4200000003": "Retail Trade",
    "CES3000000003": "Manufacturing",
    "CES2000000003": "Construction",
    "CES1000000003": "Mining & Logging",
}


def fetch_bls_data(
    series_ids: list[str],
    start_year: int = 2022,
    end_year: int = 2025,
    api_key: str | None = None,
) -> dict:
    """
    Fetch data from the BLS API for the given series IDs.
    
    Parameters
    ----------
    series_ids : list[str]
        List of BLS series IDs to fetch
    start_year : int
        Start year for data retrieval
    end_year : int
        End year for data retrieval
    api_key : str, optional
        BLS API registration key (allows more requests per day)
        
    Returns
    -------
    dict
        JSON response from the BLS API
    """
    headers = {"Content-type": "application/json"}
    
    payload = {
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    
    # Include API key if provided (increases daily request limit)
    if api_key:
        payload["registrationkey"] = api_key
    
    response = requests.post(BLS_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    
    return response.json()


def parse_bls_response(response: dict, series_mapping: dict[str, str]) -> pd.DataFrame:
    """
    Parse BLS API JSON response into a tidy pandas DataFrame.
    
    Parameters
    ----------
    response : dict
        JSON response from the BLS API
    series_mapping : dict
        Mapping of series IDs to industry names
        
    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns: date, industry, avg_hourly_earnings
    """
    if response.get("status") != "REQUEST_SUCCEEDED":
        raise ValueError(f"BLS API request failed: {response.get('message', 'Unknown error')}")
    
    records = []
    
    for series in response.get("Results", {}).get("series", []):
        series_id = series.get("seriesID")
        industry_name = series_mapping.get(series_id, series_id)
        
        for observation in series.get("data", []):
            year = int(observation["year"])
            month = int(observation["period"].replace("M", ""))  # "M01" -> 1
            
            # Skip annual averages (M13)
            if month > 12:
                continue
                
            date = datetime(year, month, 1)
            value = float(observation["value"])
            
            records.append({
                "date": date,
                "industry": industry_name,
                "avg_hourly_earnings": value,
            })
    
    df = pd.DataFrame(records)
    
    # Sort by date and industry for cleaner output
    df = df.sort_values(["date", "industry"]).reset_index(drop=True)
    
    return df


def get_ces_hourly_earnings(
    start_year: int = 2022,
    end_year: int = 2025,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Main function to fetch and parse CES average hourly earnings data.
    
    Parameters
    ----------
    start_year : int
        Start year for data retrieval (default: 2022)
    end_year : int
        End year for data retrieval (default: 2025)
    api_key : str, optional
        BLS API registration key for higher rate limits
        
    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns:
        - date: datetime, first of each month
        - industry: str, industry name
        - avg_hourly_earnings: float, average hourly earnings in dollars
    """
    series_ids = list(SERIES_IDS.keys())
    
    print(f"Fetching CES average hourly earnings data ({start_year}-{end_year})...")
    print(f"Industries: {', '.join(SERIES_IDS.values())}")
    
    # Fetch data from BLS API
    response = fetch_bls_data(
        series_ids=series_ids,
        start_year=start_year,
        end_year=end_year,
        api_key=api_key,
    )
    
    # Parse response into DataFrame
    df = parse_bls_response(response, SERIES_IDS)
    
    print(f"Retrieved {len(df)} observations across {df['industry'].nunique()} industries")
    
    return df


if __name__ == "__main__":
    # Fetch the data
    # Note: For higher rate limits, register at https://data.bls.gov/registrationEngine/
    # and pass your API key: get_ces_hourly_earnings(api_key="your_key_here")
    df = get_ces_hourly_earnings(start_year=2022, end_year=2025)
    
    # Display sample of the data
    print("\n" + "=" * 60)
    print("Sample of retrieved data:")
    print("=" * 60)
    print(df.head(20).to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary by Industry (latest available month):")
    print("=" * 60)
    latest_date = df["date"].max()
    latest_data = df[df["date"] == latest_date].sort_values("avg_hourly_earnings", ascending=False)
    print(f"\nLatest data: {latest_date.strftime('%B %Y')}\n")
    print(latest_data[["industry", "avg_hourly_earnings"]].to_string(index=False))
    
    # Save to CSV
    output_file = "ces_hourly_earnings_2022_2025.csv"
    df.to_csv(output_file, index=False)
    print(f"\nData saved to: {output_file}")
