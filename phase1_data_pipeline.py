"""
Phase 1: Data collection & structuring for the LCUSD enrollment project.

Outputs (in data/processed/):
  lcusd_enrollment_by_grade.csv  - tidy: year, school, grade, enrollment
  lcusd_enrollment_totals.csv    - one row per year (district total + K-5 / 6-8 / 9-12)
  lcusd_enrollment_ethnicity.csv - year x ethnicity (if present in the source files)
  acs_91011.csv                  - ZCTA 91011 ACS 5-year indicators by vintage
  lcusd_panel.csv                - enrollment totals joined with ACS indicators

Usage:
  1. Put CDE enrollment files (cdenroll*.txt, enr*-v2.txt) in data/raw/cde/
  2. export CENSUS_API_KEY=...   (free: api.census.gov/data/key_signup.html)
  3. python phase1_data_pipeline.py
"""

import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DISTRICT_NAME = "la canada unified"     # matched accent/case-insensitively
ZCTA = "91011"
LCUSD_CDS = {}   # filled in from the newer files (county + district code), used as a fallback filter
ACS_YEARS = range(2013, 2025)           # ACS 5-year vintages (end year); failures are skipped

RAW_CDE = Path("data/raw/cde")
RAW_ACS = Path("data/raw/acs")
OUT = Path("data/processed")
for p in (RAW_CDE, RAW_ACS, OUT):
    p.mkdir(parents=True, exist_ok=True)

CDE_DOWNLOAD_NOTE = """
No CDE files found in data/raw/cde/.
Census Day files (2023-24 on): cde.ca.gov/Ds/ad/filesenrcensus.asp
Historical school files (1981-2022): cde.ca.gov/Ds/ad/fileshistenr8122.asp
Save them in data/raw/cde/ and rerun.
"""

GRADE_ORDER = ["TK", "K"] + [str(i) for i in range(1, 13)] + ["UNGR"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def norm(s: str) -> str:
    """Lowercase, strip accents/punctuation (so 'La Cañada' == 'la canada')."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def infer_year(path: Path):
    """Return the school-year END year (e.g. 2023-24 -> 2024) from a filename."""
    name = path.stem
    m = re.search(r"(20\d{2})[-_ ]?(\d{2})\b", name)          # 2023-24, 2023_24
    if m:
        return int(m.group(1)) + 1
    m = re.search(r"(?<!\d)(\d{2})(\d{2})(?!\d)", name)        # 2324
    if m and int(m.group(2)) == (int(m.group(1)) + 1) % 100:
        return 2000 + int(m.group(2))
    m = re.search(r"(20\d{2})", name)                          # lone year -> assume end year
    return int(m.group(1)) if m else None


def grade_col_to_label(col: str):
    c = col.lower().replace(" ", "_")
    if c in ("gr_tk", "tk", "tkdn", "tk_enr"):
        return "TK"
    if c in ("kdgn", "k", "gr_k", "gr_kn"):
        return "K"
    m = re.fullmatch(r"gr_?(\d{1,2})", c)
    if m and 1 <= int(m.group(1)) <= 12:
        return str(int(m.group(1)))
    if c in ("ungr_elm", "ungr_sec", "ungr"):
        return "UNGR"
    return None


def read_cde_file(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, encoding=enc) as fh:
                header = fh.readline()
            counts = {d: header.count(d) for d in ("\t", ",", "|", ";")}
            sep = max(counts, key=counts.get)
            if counts[sep] == 0:
                # Space-delimited with quoted fields (CDE's 2023-24 file); empty fields = consecutive spaces
                df = pd.read_csv(path, sep=" ", quotechar='"', encoding=enc, dtype=str)
            else:
                df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str)
            return df.apply(lambda c: c.str.strip())
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path}")


# ----------------------------------------------------------------------------
# CDE enrollment
# ----------------------------------------------------------------------------
def process_hist_df(df: pd.DataFrame, path: Path):
    """Historical files: many years per file (ACADEMIC_YEAR), school-level, ethnicity x gender x
    ENR_TYPE rows (P = primary, C = combined primary + short-term, from 2020-21)."""
    dcol = next((c for c in df.columns if c in ("district", "district_name", "districtname")), None)
    mask = None
    if dcol is not None and df[dcol].notna().any():
        mask = df[dcol].fillna("").map(norm).str.contains(DISTRICT_NAME, regex=False)
    elif "cds_code" in df.columns and LCUSD_CDS.get("prefix"):
        mask = df["cds_code"].str.startswith(LCUSD_CDS["prefix"])
    if mask is None:
        raise ValueError(f"{path.name}: no district name column and no CDS prefix known "
                         f"(process a newer cdenroll file first). Columns: {list(df.columns)[:12]}")
    df = df[mask].copy()
    if df.empty:
        print(f"  ! {path.name}: no LCUSD rows found")
        return None, None
    if "academic_year" not in df.columns:
        raise ValueError(f"{path.name}: no academic_year column. Columns: {list(df.columns)[:12]}")

    df["year"] = df["academic_year"].str.extract(r"(\d{4})")[0].astype(int) + 1   # school-year END year
    if "enr_type" in df.columns:        # avoid double counting P and C rows: prefer C when a year has it
        parts = []
        for _, g in df.groupby("year"):
            types = set(g["enr_type"].dropna().str.upper())
            pick = "C" if "C" in types else ("P" if "P" in types else None)
            parts.append(g if pick is None else g[g["enr_type"].str.upper() == pick])
        df = pd.concat(parts)
    if "school" not in df.columns:
        df["school"] = df["cds_code"]

    gcols = {c: grade_col_to_label(c) for c in df.columns if grade_col_to_label(c)}

    def to_long(extra=None):
        keep = ["year", "school"] + ([extra] if extra else [])
        long = df[keep + list(gcols)].melt(id_vars=keep, var_name="col", value_name="enrollment")
        long["grade"] = long["col"].map(gcols)
        long["enrollment"] = pd.to_numeric(long["enrollment"], errors="coerce").fillna(0)
        return long.groupby(keep + ["grade"], as_index=False)["enrollment"].sum()

    grade_df = to_long()
    eth_df = None
    if "race_ethnicity" in df.columns:
        eth_df = (to_long("race_ethnicity").rename(columns={"race_ethnicity": "ethnicity_code"})
                  .groupby(["year", "ethnicity_code"], as_index=False)["enrollment"].sum())
    yr_tot = grade_df.groupby("year")["enrollment"].sum().astype(int).to_dict()
    print(f"  {path.name}: years/totals={yr_tot}")
    return grade_df, eth_df


def process_cde_file(path: Path):
    """Return (grade_df, ethnicity_df) for the LCUSD rows of one CDE file."""
    df = read_cde_file(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"academicyear": "academic_year",
                            "aggregatelevel": "aggregate_level",
                            "reportingcategory": "reporting_category"})
    if "reporting_category" not in df.columns:      # historical multi-year files (2014-2022 etc.)
        return process_hist_df(df, path)

    # district filter
    dcol = next((c for c in df.columns if c in ("district", "district_name", "districtname")), None)
    if dcol is None:
        raise ValueError(f"{path.name}: no district column. Columns: {list(df.columns)[:15]}")
    df = df[df[dcol].fillna("").map(norm).str.contains(DISTRICT_NAME, regex=False)].copy()
    if df.empty:
        print(f"  ! {path.name}: no LCUSD rows found")
        return None, None

    # year
    year = None
    for c in ("academic_year", "year"):
        if c in df.columns:
            ys = df[c].dropna().iloc[0]
            m = re.search(r"(20\d{2})[-_/ ]?(\d{2,4})", str(ys))
            if m:
                end = m.group(2)
                year = int(end) if len(end) == 4 else int(m.group(1)) + 1
    year = year or infer_year(path)
    if year is None:
        raise ValueError(f"Cannot infer year for {path.name}; rename it to include e.g. 2023-24")

    # Newer files have rows at several levels (S=school, D=district...). Use the district-level
    # rows (Charter = ALL) so nothing is double counted. Older files are school-level only.
    if "aggregate_level" in df.columns:
        df = df[df["aggregate_level"].str.upper() == "D"]
        if "charter" in df.columns and (df["charter"].str.upper() == "ALL").any():
            df = df[df["charter"].str.upper() == "ALL"]
        df = df.assign(school="DISTRICT")
        if {"countycode", "districtcode"} <= set(df.columns) and len(df):
            LCUSD_CDS["prefix"] = df["countycode"].iloc[0].zfill(2) + df["districtcode"].iloc[0].zfill(5)
    if "school" not in df.columns:
        scol = next((c for c in df.columns if c in ("school_name", "schoolname")), None)
        df["school"] = df[scol] if scol else "ALL"

    gcols = {c: grade_col_to_label(c) for c in df.columns if grade_col_to_label(c)}

    # Newer format: 'reporting_category' (TA = total, R* = race/ethnicity)
    grade_rows = df[df["reporting_category"] == "TA"]
    eth_rows = df[df["reporting_category"].str.startswith("R", na=False)]
    eth_key = "reporting_category"

    def to_long(frame, extra=None):
        keep = ["school"] + ([extra] if extra else [])
        long = frame[keep + list(gcols)].melt(id_vars=keep, var_name="col", value_name="enrollment")
        long["grade"] = long["col"].map(gcols)
        long["enrollment"] = pd.to_numeric(long["enrollment"], errors="coerce").fillna(0)
        long = long.groupby(keep + ["grade"], as_index=False)["enrollment"].sum()
        long.insert(0, "year", year)
        return long

    grade_df = to_long(grade_rows)
    print("    by school:", grade_df.groupby("school")["enrollment"].sum().astype(int).to_dict())
    eth_df = None
    if eth_key and gcols:
        eth_long = to_long(eth_rows, extra=eth_key).rename(columns={eth_key: "ethnicity_code"})
        eth_df = eth_long.groupby(["year", "ethnicity_code"], as_index=False)["enrollment"].sum()
    print(f"  {path.name}: year={year}, rows={len(grade_df)}, total={int(grade_df.enrollment.sum()):,}")
    return grade_df, eth_df


def build_enrollment():
    files = sorted(p for p in RAW_CDE.glob("*") if p.suffix.lower() in (".txt", ".csv", ".tsv"))
    if not files:
        sys.exit(CDE_DOWNLOAD_NOTE)
    print(f"Processing {len(files)} CDE files...")
    grades, eths = [], []
    for f in files:
        try:
            g, e = process_cde_file(f)
        except ValueError as err:
            print(f"  ! Skipping {f.name}: {str(err)[:300]}")
            continue
        if g is not None:
            grades.append(g)
        if e is not None:
            eths.append(e)
    if not grades:
        sys.exit("No LCUSD data parsed. Check DISTRICT_NAME and file formats.")

    by_grade = pd.concat(grades, ignore_index=True)
    by_grade["grade"] = pd.Categorical(by_grade["grade"], GRADE_ORDER, ordered=True)
    by_grade = by_grade.sort_values(["year", "school", "grade"])
    by_grade.to_csv(OUT / "lcusd_enrollment_by_grade.csv", index=False)

    # totals + grade bands
    band = {**{g: "K-5" for g in ["TK", "K", "1", "2", "3", "4", "5"]},
            **{g: "6-8" for g in ["6", "7", "8"]},
            **{g: "9-12" for g in ["9", "10", "11", "12"]}, "UNGR": "Ungraded"}
    tmp = by_grade.assign(band=by_grade["grade"].astype(str).map(band))
    totals = tmp.pivot_table(index="year", columns="band", values="enrollment", aggfunc="sum").fillna(0)
    totals["total"] = totals.sum(axis=1)
    totals = totals.reset_index()
    totals["yoy_pct"] = totals["total"].pct_change() * 100
    totals.to_csv(OUT / "lcusd_enrollment_totals.csv", index=False)

    if eths:
        pd.concat(eths).sort_values(["year", "ethnicity_code"]).to_csv(
            OUT / "lcusd_enrollment_ethnicity.csv", index=False)
    return by_grade, totals


# ----------------------------------------------------------------------------
# ACS (Census API) for ZCTA 91011
# ----------------------------------------------------------------------------
ACS_VARS = {
    "B01003_001E": "total_pop",
    "B09001_001E": "pop_under18",
    "B11005_002E": "hh_with_children",
    "B11005_001E": "households",
    "B19013_001E": "median_hh_income",
    "B25077_001E": "median_home_value",
    "B25003_001E": "occupied_units",
    "B25003_002E": "owner_occupied",
    "B25002_003E": "vacant_units",
    "B25002_001E": "housing_units",
    # Age cohorts relevant to K-12 (male + female summed below)
    "B01001_003E": "m_u5", "B01001_004E": "m_5_9", "B01001_005E": "m_10_14", "B01001_006E": "m_15_17",
    "B01001_027E": "f_u5", "B01001_028E": "f_5_9", "B01001_029E": "f_10_14", "B01001_030E": "f_15_17",
}


def fetch_acs_year(year: int):
    # Build the URL by hand: requests would encode the spaces as '+', which the Census API dislikes.
    url = (f"https://api.census.gov/data/{year}/acs/acs5?get=NAME,{','.join(ACS_VARS)}"
           f"&for=zip%20code%20tabulation%20area:{ZCTA}")
    if year < 2020:                      # pre-2020 vintages require the state (CA = 06)
        url += "&in=state:06"
    if os.getenv("CENSUS_API_KEY"):
        url += f"&key={os.environ['CENSUS_API_KEY']}"
    cache = RAW_ACS / f"acs5_{year}_{ZCTA}.json"
    if cache.exists():
        import json
        rows = json.loads(cache.read_text())
    else:
        r = requests.get(url, timeout=30)
        if "Missing Key" in r.text or "Invalid Key" in r.text:
            raise RuntimeError("Census API needs a free key: https://api.census.gov/data/key_signup.html "
                               "then run: export CENSUS_API_KEY=your_key_here")
        try:
            rows = r.json()
        except ValueError:
            print(f"  ACS {year}: HTTP {r.status_code}, not JSON. Body starts: {r.text[:150]!r}")
            return None
        if r.status_code != 200:
            print(f"  ACS {year}: HTTP {r.status_code}: {str(rows)[:150]}")
            return None
        cache.write_text(r.text)
    header, values = rows[0], rows[1]
    rec = dict(zip(header, values))
    out = {"acs_year": year}
    for k, v in ACS_VARS.items():
        x = pd.to_numeric(rec.get(k), errors="coerce")
        out[v] = None if pd.isna(x) or x < 0 else x    # Census uses large negatives as null codes
    return out


def build_acs():
    print("Fetching ACS 5-year for ZCTA", ZCTA)
    recs = []
    for y in ACS_YEARS:
        try:
            rec = fetch_acs_year(y)
        except requests.RequestException as e:
            print(f"  ACS {y}: {e}")
            rec = None
        except RuntimeError as e:
            print(f"  {e}")
            break
        if rec:
            recs.append(rec)
        time.sleep(0.2)
    if not recs:
        print("  No ACS data retrieved (network blocked?). Continuing without it.")
        return None
    acs = pd.DataFrame(recs)
    acs["pop_5_9"] = acs.m_5_9 + acs.f_5_9
    acs["pop_10_14"] = acs.m_10_14 + acs.f_10_14
    acs["pop_u5"] = acs.m_u5 + acs.f_u5
    acs["pop_15_17"] = acs.m_15_17 + acs.f_15_17
    acs["school_age_5_17"] = acs.pop_5_9 + acs.pop_10_14 + acs.pop_15_17
    acs["pct_under18"] = acs.pop_under18 / acs.total_pop * 100
    acs["owner_occ_rate"] = acs.owner_occupied / acs.occupied_units * 100
    acs["vacancy_rate"] = acs.vacant_units / acs.housing_units * 100
    acs.drop(columns=[c for c in acs.columns if c[:2] in ("m_", "f_")], inplace=True)
    acs.to_csv(OUT / "acs_91011.csv", index=False)
    return acs


# ----------------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------------
def build_panel(totals: pd.DataFrame, acs):
    if acs is None:
        return totals
    # CDE year = school-year end. ACS vintage = end year of its 5-yr window.
    panel = totals.merge(acs, left_on="year", right_on="acs_year", how="left")
    panel.to_csv(OUT / "lcusd_panel.csv", index=False)
    return panel


if __name__ == "__main__":
    by_grade, totals = build_enrollment()
    acs = build_acs()
    panel = build_panel(totals, acs)
    print("\nDistrict totals:")
    print(totals[["year", "total", "yoy_pct"]].round(1).to_string(index=False))
    print(f"\nSaved files to {OUT.resolve()}")