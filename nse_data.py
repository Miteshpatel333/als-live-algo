import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta


NSE_BASE = "https://www.nseindia.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 12; Mobile) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Mobile Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


def create_nse_session():
    """
    Create an NSE session with browser-like headers.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        session.get(
            NSE_BASE,
            timeout=15,
        )
    except Exception:
        pass

    return session


def fetch_nse_option_history(
    symbol="NIFTY",
    expiry_date=None,
    option_type="CE",
    strike_price=None,
    from_date=None,
    to_date=None,
):
    """
    Fetch historical NSE index-option contract data.

    Example:
        symbol="NIFTY"
        expiry_date="22-Sep-2026"
        option_type="CE"
        strike_price=23350
    """

    if expiry_date is None:
        raise ValueError("expiry_date is required")

    if from_date is None:
        from_date = (
            datetime.now() - timedelta(days=90)
        ).strftime("%d-%m-%Y")

    if to_date is None:
        to_date = datetime.now().strftime("%d-%m-%Y")

    session = create_nse_session()

    url = f"{NSE_BASE}/api/historical/foCPV"

    params = {
        "from": from_date,
        "to": to_date,
        "instrumentType": "OPTIDX",
        "symbol": symbol,
        "expiryDate": expiry_date,
        "optionType": option_type,
    }

    if strike_price is not None:
        params["strikePrice"] = strike_price

    response = session.get(
        url,
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"NSE request failed: HTTP {response.status_code}"
        )

    try:
        data = response.json()
    except Exception:
        # Sometimes NSE may return HTML/CSV instead of JSON.
        text = response.text.strip()

        if not text:
            raise RuntimeError("NSE returned empty response")

        try:
            return pd.read_csv(StringIO(text))
        except Exception:
            raise RuntimeError(
                "NSE returned an unexpected response format"
            )

    if isinstance(data, dict):
        records = data.get("data", data)
    else:
        records = data

    if not isinstance(records, list):
        raise RuntimeError("Unexpected NSE response structure")

    if len(records) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    return normalize_nse_option_data(df)


def normalize_nse_option_data(df):
    """
    Normalize NSE option historical data into a consistent format.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Standardize column names
    rename_map = {}

    for col in out.columns:
        clean = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        rename_map[col] = clean

    out = out.rename(columns=rename_map)

    # Common NSE column aliases
