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
    aliases = {
        "date": ["date", "trade_date"],
        "expiry": ["expiry", "expiry_date"],
        "strike": ["strike_price", "strike"],
        "option_type": ["option_type", "opt_type"],
        "open": ["open", "open_price"],
        "high": ["high", "high_price"],
        "low": ["low", "low_price"],
        "close": ["close", "close_price"],
        "ltp": ["ltp", "last_price"],
        "volume": ["volume", "no_of_contracts"],
        "oi": ["open_interest", "oi"],
    }

    for standard_name, candidates in aliases.items():
        for candidate in candidates:
            if candidate in out.columns:
                if standard_name not in out.columns:
                    out[standard_name] = out[candidate]
                break

    if "date" in out.columns:
        out["date"] = pd.to_datetime(
            out["date"],
            errors="coerce",
            dayfirst=True,
        )

    numeric_columns = [
        "strike",
        "open",
        "high",
        "low",
        "close",
        "ltp",
        "volume",
        "oi",
    ]

    for col in numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col],
                errors="coerce",
            )

    if "date" in out.columns:
        out = out.dropna(subset=["date"])
        out = out.sort_values("date")

    return out.reset_index(drop=True)


def get_available_expiries(
    symbol="NIFTY",
    from_date=None,
    to_date=None,
):
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
    }

    response = session.get(
        url,
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"NSE request failed: HTTP {response.status_code}"
        )

    data = response.json()
    records = data.get("data", [])

    if not records:
        return []

    df = pd.DataFrame(records)

    if "FH_EXPIRY_DT" in df.columns:
        values = df["FH_EXPIRY_DT"]
    elif "expiryDate" in df.columns:
        values = df["expiryDate"]
    elif "expiry" in df.columns:
        values = df["expiry"]
    else:
        return []

    expiries = (
        pd.to_datetime(
            values,
            errors="coerce",
            dayfirst=True,
        )
        .dropna()
        .dt.strftime("%d-%b-%Y")
        .unique()
        .tolist()
    )

    return sorted(expiries)


def download_option_pair(
    symbol,
    expiry_date,
    strike_price,
    from_date,
    to_date,
):
    ce = fetch_nse_option_history(
        symbol=symbol,
        expiry_date=expiry_date,
        option_type="CE",
        strike_price=strike_price,
        from_date=from_date,
        to_date=to_date,
    )

    pe = fetch_nse_option_history(
        symbol=symbol,
        expiry_date=expiry_date,
        option_type="PE",
        strike_price=strike_price,
        from_date=from_date,
        to_date=to_date,
    )

    return {
        "CE": ce,
        "PE": pe,
    }


def prepare_straddle_dataframe(ce_df, pe_df):
    if ce_df is None or pe_df is None:
        return pd.DataFrame()

    if ce_df.empty or pe_df.empty:
        return pd.DataFrame()

    if "date" not in ce_df.columns:
        return pd.DataFrame()

    if "date" not in pe_df.columns:
        return pd.DataFrame()

    ce_price_col = (
        "close"
        if "close" in ce_df.columns
        else "ltp"
        if "ltp" in ce_df.columns
        else None
    )

    pe_price_col = (
        "close"
        if "close" in pe_df.columns
        else "ltp"
        if "ltp" in pe_df.columns
        else None
    )

    if ce_price_col is None or pe_price_col is None:
        return pd.DataFrame()

    ce = ce_df[
        ["date", ce_price_col]
    ].rename(
        columns={ce_price_col: "CE_PRICE"}
    )

    pe = pe_df[
        ["date", pe_price_col]
    ].rename(
        columns={pe_price_col: "PE_PRICE"}
    )

    merged = pd.merge(
        ce,
        pe,
        on="date",
        how="inner",
    )

    if merged.empty:
        return merged

    merged["STRADDLE_PRICE"] = (
        merged["CE_PRICE"] +
        merged["PE_PRICE"]
    )

    return merged.sort_values("date").reset_index(drop=True)
