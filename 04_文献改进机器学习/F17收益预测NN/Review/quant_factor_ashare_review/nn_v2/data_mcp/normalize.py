# Research project author: Gilmour Jia
"""Conservative normalization of MCP evidence, never an acceptance shortcut."""
import math
import re
import pandas as pd

CODE_CHANGES = [
    {"old":"sz000022", "new":"sz001872", "effective":"2018-12-26",
     "source":"https://static.cninfo.com.cn/finalpage/2018-12-26/1205690369.PDF",
     "shares_unchanged_verified":True},
    {"old":"sz000043", "new":"sz001914", "effective":"2019-12-16",
     "source":"https://www.szse.cn/disclosure/notice/general/t20191211_572534.html",
     "shares_unchanged_verified":False},
    {"old":"sz300114", "new":"sz302132", "effective":"2025-02-17",
     "source":"https://static.cninfo.com.cn/finalpage/2025-02-07/1222485220.PDF",
     "shares_unchanged_verified":True},
]
CDR = {"sh689009":"https://www.sse.com.cn/disclosure/announcement/listing/c/c_20201027_5242851.shtml"}


def scalar_number(value):
    """Blank and ambiguous text remain unknown, never choose a tax alternative."""
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        return float("nan")
    result = float(text)
    return result if math.isfinite(result) else float("nan")


def symbol(code):
    if not re.fullmatch(r"(sh|sz|bj)\.\d{6}", code):
        raise ValueError(f"Unexpected security identifier: {code}")
    return code.replace(".", "")


def bars(record):
    if not record["summary"]["ok"]:
        raise ValueError("Failed response cannot be normalized")
    df = pd.DataFrame(record["result"]["data"])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="raise").astype("datetime64[ns]")
    args = record["arguments"]
    start, end = args.get("date", args.get("start_date")), args.get("date", args.get("end_date"))
    if start and (df.date < pd.Timestamp(start)).any() or end and (df.date > pd.Timestamp(end)).any():
        raise ValueError("Provider violated the requested historical date range")
    if args.get("code") and not df.code.eq(args["code"]).all():
        raise ValueError("Provider returned a different security")
    if not df.adjustflag.eq("3").all():
        raise ValueError("Adjusted prices cannot be accepted as raw execution prices")
    df["symbol"] = df.code.map(symbol)
    if df.duplicated(["date","symbol"]).any():
        raise ValueError("Duplicate provider keys")
    out = df[["date","symbol"]].copy()
    for source, target in [("open","raw_open"),("high","raw_high"),("low","raw_low"),("close","raw_close"),
                           ("preclose","raw_pre_close"),("volume","raw_volume"),("amount","raw_amount")]:
        out[target] = df[source].map(scalar_number)
    out["turnover"] = df.turn.map(scalar_number)/100
    out["is_st"] = df.isST.map({"1":True,"0":False}).astype("boolean")
    out["daily_status"] = df.tradestatus.map({"1":"trading","0":"suspended"}).fillna("unknown")
    # Daily state does not establish the exact 09:30 trading permission.
    out["open_status"] = "unknown"
    out["source"] = "BaoStock via stdio MCP"
    out["source_symbol"] = out.symbol
    out["security_id"] = out.symbol
    out["active_code_on_date"] = True
    for change in CODE_CHANGES:
        old, new, date = change["old"], change["new"], pd.Timestamp(change["effective"])
        relevant = out.symbol.isin([old,new])
        out.loc[relevant,"security_id"] = old
        inactive = ((out.symbol.eq(new) & (out.date < date)) |
                    (out.symbol.eq(old) & (out.date >= date)))
        out.loc[inactive,"active_code_on_date"] = False
    # Diagnostic estimate only: rounded turnover is not a verified share register.
    out["implied_float_shares_unverified"] = out.raw_volume/out.turnover.where(out.turnover > 0)
    out["adjustment_verified"] = False
    return out


def distributions(record):
    if not record["summary"]["ok"]:
        raise ValueError("Failed dividend query")
    rows = []
    for event in record["result"]["data"]:
        gross = scalar_number(event.get("dividCashPsBeforeTax"))
        net = scalar_number(event.get("dividCashPsAfterTax"))
        stock = scalar_number(event.get("dividStocksPs"))
        reserve = scalar_number(event.get("dividReserveToStockPs"))
        rows.append({
            "symbol":symbol(event["code"]), "record_date":event.get("dividRegistDate"),
            "ex_date":event.get("dividOperateDate"),"pay_date":event.get("dividPayDate"),
            "stock_date":event.get("dividStockMarketDate"),"gross_cash_per_share":gross,
            "vendor_net_cash_per_share":net,"net_cash_per_share":float("nan"),
            "vendor_net_text":event.get("dividCashPsAfterTax"),
            "stock_bonus_per_share":stock,"reserve_per_share":reserve,
            "share_ratio":1+stock+reserve if math.isfinite(stock+reserve) else float("nan"),
            "verified":False,
            "reason":"Requires account tax convention, complete distribution ratios, dates and reconciliation",
        })
    return pd.DataFrame(rows)
