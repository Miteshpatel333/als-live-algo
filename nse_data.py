import requests
import pandas as pd
from datetime import datetime, timedelta


NSE_BASE = "https://www.nseindia.com"
NSE_API = f"{NSE_BASE}/api/historicalOR/foCPV"
NSE_REPORT = f"{NSE_BASE}/report-detail/fo_eq_security"


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 12; Mobile) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Mobile Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_REPORT,
    "Connection": "keep-alive",
}


def create_nse_session():
    """
    Create NSE session and obtain cookies from the official
    historical contract-wise report page.
    """

    session = requests.Session()
    session.headers.update(HEADERS)

    try:
        response = session.get(
            NSE_REPORT,
            timeout=20,
        )

        if response.status_code not in (200, 403):
            raise RuntimeError(
                f"NSE report page returned HTTP {response.status_code}"
            )

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Unable to connect to NSE: {exc}"
        )

    return session


def _date_string(value):
    """
    Convert date-like input into DD-MM-YYYY.
    """

    if value is None:
        return None

    if isinstance(value, datetime):
        return value.strftime("%d-%m-%Y")

    try:
        return pd.to_datetime(value).strftime("%d-%m-%Y")
    except Exception:
        return str(value)


def _expiry_string(value):
    """
    Convert expiry into DD-MMM-YYYY.
    Example: 2026-09-22 -> 22-Sep-2026
    """

    if value is None:
        return None

    if isinstance(value, datetime):
        return value.strftime("%d-%b-%Y")

    try:
        return pd.to_datetime(value).strftime("%d-%b-%Y")
    except Exception:
        return str(value)


def _year_from_expiry(expiry_date):
    """
    Extract expiry year.
    """

    try:
        return str(pd.to_datetime(expiry_date).year)
    except Exception:
        text = str(expiry_date)

        if text[-4:].isdigit():
            return text[-4:]

        return str(datetime.now().year)


def _request_json(session, params):
    """
    Request NSE historical F&O data.

    NSE currently uses:
        /api/historicalOR/foCPV

    The year parameter is required.
    """

    response = session.get(
        NSE_API,
        params=params,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"NSE request failed: HTTP {response.status_code}"
        )

    content_type = response.headers.get(
        "content-type",
        ""
    ).lower()

    text = response.text.strip()

    if not text:
        raise RuntimeError(
            "NSE returned an empty response"
        )

    # Normal JSON response
    try:
        return response.json()

    except Exception:
        pass

    # Helpful diagnostic for HTML / anti-bot response
    if (
        "text/html" in content_type
        or text.startswith("<")
        or "<html" in text.lower()
    ):
        raise RuntimeError(
            "NSE returned an HTML/anti-bot response. "
            "Please retry after a few seconds."
        )

    raise RuntimeError(
        "NSE returned an unexpected response format"
    )


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
        raise ValueError(
            "expiry_date is required"
        )

    if option_type not in ("CE", "PE"):
        raise ValueError(
            "option_type must be CE or PE"
        )

    if strike_price is None:
        raise ValueError(
            "strike_price is required"
        )

    if from_date is None:
        from_date = (
            datetime.now() - timedelta(days=90)
        )

    if to_date is None:
        to_date = datetime.now()

    from_date = _date_string(from_date)
    to_date = _date_string(to_date)
    expiry_date = _expiry_string(expiry_date)

    year = _year_from_expiry(expiry_date)

    session = create_nse_session()

    params = {
        "from": from_date,
        "to": to_date,
        "instrumentType": "OPTIDX",
        "symbol": symbol,
        "year": year,
        "expiryDate": expiry_date,
        "optionType": option_type,
        "strikePrice": f"{float(strike_price):.2f}",
    }

    payload = _request_json(
        session,
        params,
    )

    if isinstance(payload, dict):
        records = payload.get(
            "data",
            []
        )
    elif isinstance(payload, list):
        records = payload
    else:
        records = []

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    return normalize_nse_option_data(df)


def normalize_nse_option_data(df):
    """
    Normalize NSE option historical data
    into a consistent format.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # --------------------------------------------------
    # Standardize column names
    # --------------------------------------------------

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

    out = out.rename(
        columns=rename_map
    )

    # --------------------------------------------------
    # NSE current field names
    # --------------------------------------------------

    aliases = {

        "date": [
            "fh_timestamp",
            "date",
            "trade_date",
        ],

        "expiry": [
            "fh_expiry_dt",
            "expiry",
            "expiry_date",
        ],

        "strike": [
            "fh_strike_price",
            "strike_price",
            "strike",
        ],

        "option_type": [
            "fh_option_type",
            "option_type",
            "opt_type",
        ],

        "open": [
            "fh_opening_price",
            "open",
            "open_price",
        ],

        "high": [
            "fh_trade_high_price",
            "high",
            "high_price",
        ],

        "low": [
            "fh_trade_low_price",
            "low",
            "low_price",
        ],

        "close": [
            "fh_closing_price",
            "close",
            "close_price",
        ],

        "ltp": [
            "fh_last_traded_price",
            "ltp",
            "last_price",
        ],

        "volume": [
            "fh_tot_traded_qty",
            "total_traded_quantity",
            "no_of_contracts",
            "volume",
        ],

        "oi": [
            "fh_open_int",
            "open_interest",
            "oi",
        ],
    }

    for standard_name, candidates in aliases.items():

        for candidate in candidates:

            if candidate in out.columns:

                if standard_name not in out.columns:
                    out[standard_name] = out[candidate]

                break

    # --------------------------------------------------
    # Date
    # --------------------------------------------------

    if "date" in out.columns:

        out["date"] = pd.to_datetime(
            out["date"],
            errors="coerce",
            dayfirst=True,
        )

    # --------------------------------------------------
    # Numeric fields
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Clean / sort
    # --------------------------------------------------

    if "date" in out.columns:

        out = out.dropna(
            subset=["date"]
        )

        out = out.sort_values(
            "date"
        )

    return out.reset_index(
        drop=True
    )


def get_available_expiries(
    symbol="NIFTY",
    from_date=None,
    to_date=None,
):
    """
    Get available expiries from NSE historical
    contract-wise data.
    """

    if from_date is None:
        from_date = (
            datetime.now() - timedelta(days=90)
        )

    if to_date is None:
        to_date = datetime.now()

    from_date = _date_string(from_date)
    to_date = _date_string(to_date)

    year = str(
        pd.to_datetime(
            from_date,
            dayfirst=True,
        ).year
    )

    session = create_nse_session()

    params = {
        "from": from_date,
        "to": to_date,
        "instrumentType": "OPTIDX",
        "symbol": symbol,
        "year": year,
    }

    payload = _request_json(
        session,
        params,
    )

    if isinstance(payload, dict):
        records = payload.get(
            "data",
            []
        )
    elif isinstance(payload, list):
        records = payload
    else:
        records = []

    if not records:
        return []

    df = pd.DataFrame(records)

    if df.empty:
        return []

    expiry_column = None

    for candidate in [
        "FH_EXPIRY_DT",
        "expiryDate",
        "expiry",
        "EXPIRY",
    ]:

        if candidate in df.columns:
            expiry_column = candidate
            break

    if expiry_column is None:
        return []

    values = (
        pd.to_datetime(
            df[expiry_column],
            errors="coerce",
            dayfirst=True,
        )
        .dropna()
    )

    expiries = (
        values
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
    """
    Download both CE and PE for the same
    symbol / expiry / strike.
    """

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


def prepare_straddle_dataframe(
    ce_df,
    pe_df,
):
    """
    Combine CE + PE historical prices
    by trading date.
    """

    if ce_df is None or pe_df is None:
        return pd.DataFrame()

    if ce_df.empty or pe_df.empty:
        return pd.DataFrame()

    if "date" not in ce_df.columns:
        return pd.DataFrame()

    if "date" not in pe_df.columns:
        return pd.DataFrame()

    # Prefer closing price
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

    if ce_price_col is None:
        return pd.DataFrame()

    if pe_price_col is None:
        return pd.DataFrame()

    ce = ce_df[
        ["date", ce_price_col]
    ].copy()

    pe = pe_df[
        ["date", pe_price_col]
    ].copy()

    ce = ce.rename(
        columns={
            ce_price_col: "CE_PRICE"
        }
    )

    pe = pe.rename(
        columns={
            pe_price_col: "PE_PRICE"
        }
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
        merged["CE_PRICE"]
        + merged["PE_PRICE"]
    )

    return (
        merged
        .sort_values("date")
        .reset_index(drop=True)
        )
