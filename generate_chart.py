"""
generate_chart.py — Production Chart Generation Pipeline

Architecture:
  LLMClient              → prompt + parse + repair + fallback
  ChartConfigSchema      → Pydantic strict validation
  ChartConfigNormalizer  → business-rule normalisation (immutable)
  DataTransformer        → all DataFrame work, one aggregation block
  ChartBuilder           → assemble final chart payloads
  TableBuilder           → assemble table payloads
  ScorecardBuilder       → deterministic KPI scorecards (no LLM, max 5, rendered first)
  ChartGenerator         → thin orchestrator + LRU cache
"""

import json
import logging
import os
import re
import hashlib
from typing import Literal, Optional

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, model_validator, ValidationError

# ---------- LOGGER ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env")


# ─────────────────────────────────────────────
# 1. PYDANTIC SCHEMA  (strict validation of LLM output)
# ─────────────────────────────────────────────

ChartType       = Literal["bar", "line", "scatter", "histogram", "pie", "heatmap", "box"]
AggType         = Literal["sum", "mean", "count", "none"]
GranularityType = Literal["day", "week", "month", "year", "none"]
LayoutSize      = Literal["small", "medium", "large"]


class ChartConfigSchema(BaseModel):
    type:             ChartType
    x:                str
    y:                Optional[str]   = None
    color:            Optional[str]   = None
    aggregation:      AggType         = "none"
    time_granularity: GranularityType = "none"
    layout_size:      LayoutSize      = "medium"
    title:            str             = ""

    @field_validator("x", "y", "color", mode="before")
    @classmethod
    def strip_nullish(cls, v):
        """Coerce null-like strings ('null', 'none', '') to None."""
        if isinstance(v, str) and v.strip().lower() in ("null", "none", ""):
            return None
        return v

    @model_validator(mode="after")
    def histogram_has_no_y(self) -> "ChartConfigSchema":
        if self.type == "histogram":
            self.y = None
        return self

    @model_validator(mode="after")
    def non_histogram_needs_y(self) -> "ChartConfigSchema":
        if self.type != "histogram" and not self.y:
            if self.type == "pie":
                # LLM sent pie with only x — treat as value_counts, normalizer synthesises y
                self.aggregation = "count"
                # y intentionally left None; normalizer will handle it
            else:
                raise ValueError(f"Chart type '{self.type}' requires a y column")
        return self


TableType = Literal["pivot", "summary"]
AggType2  = Literal["sum", "mean", "count", "min", "max"]


class TableConfigSchema(BaseModel):
    type:        TableType
    title:       str            = ""
    index:       Optional[str]  = None          # pivot row grouping column
    columns:     Optional[str]  = None          # pivot column grouping column
    values:      str            = ""            # column to aggregate (required)
    aggregation: AggType2       = "sum"

    @field_validator("values", mode="before")
    @classmethod
    def values_must_be_string(cls, v):
        """Accept single-element lists from LLM (e.g. ["Sales"] -> "Sales")."""
        if isinstance(v, list):
            if len(v) == 1:
                return str(v[0])
            raise ValueError(f"'values' must be a single column name, got list: {v}")
        return v

    @model_validator(mode="after")
    def pivot_needs_index(self) -> "TableConfigSchema":
        if self.type == "pivot" and not self.index:
            raise ValueError("pivot table requires an 'index' column")
        return self

    @model_validator(mode="after")
    def values_not_empty(self) -> "TableConfigSchema":
        if not self.values:
            raise ValueError("'values' column is required")
        return self


class LLMResponseSchema(BaseModel):
    charts: list[ChartConfigSchema]
    tables: list[TableConfigSchema] = []

    @field_validator("charts", mode="before")
    @classmethod
    def drop_invalid_charts(cls, raw_charts):
        """
        Validate each chart individually.
        Drop invalid ones with a warning instead of failing the whole response.
        Raise only if EVERY chart is invalid (nothing left to render).
        """
        if not isinstance(raw_charts, list):
            raise ValueError("'charts' must be a list")

        valid = []
        for i, raw in enumerate(raw_charts):
            try:
                valid.append(ChartConfigSchema.model_validate(raw))
            except ValidationError as exc:
                first_err = exc.errors()[0].get("msg", str(exc))
                logger.warning("Dropping chart[%d] (%s): %s", i, raw.get("type", "?"), first_err)

        if not valid:
            raise ValueError("All charts failed validation — no renderable charts returned")

        return valid

    @field_validator("tables", mode="before")
    @classmethod
    def drop_invalid_tables(cls, raw_tables):
        """Drop invalid table configs individually — never fail the whole response."""
        if not isinstance(raw_tables, list):
            return []
        valid = []
        for i, raw in enumerate(raw_tables):
            try:
                valid.append(TableConfigSchema.model_validate(raw))
            except ValidationError as exc:
                first_err = exc.errors()[0].get("msg", str(exc))
                logger.warning("Dropping table[%d] (%s): %s", i, raw.get("type", "?"), first_err)
        return valid


# ─────────────────────────────────────────────
# 2. LLM CLIENT  (prompt + parse + repair + fallback)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a data visualization expert.
Return ONLY valid JSON — no explanation, no markdown fences.

OUTPUT FORMAT:
{
  "charts": [
    {
      "type": "bar | line | scatter | histogram | pie",
      "x": "column_name",
      "y": "column_name or null",
      "color": "column_name or null",
      "aggregation": "sum | mean | count | none",
      "time_granularity": "day | week | month | year | none",
      "layout_size": "small | medium | large",
      "title": "clear title"
    }
  ],
  "tables": []
}

CHART RULES:
1. Use ONLY columns listed in the user message.
2. Chart type priority (high → low): line > bar > pie > scatter > histogram.
   - trend over time     → line  (ALWAYS use when a date column exists)
   - category comparison → bar
   - part-of-whole       → pie   (only when ≤ 6 unique values)
   - relationship        → scatter
   - distribution        → histogram (last resort)
3. Date column present → ALWAYS produce at least one line chart aggregated by month or year.
4. Aggregation: sales/qty/revenue/cost → sum; price/rate/score → mean; else → count.
5. Use "color" ONLY when: query compares across categories AND the
   column has <= 6 unique values AND chart type is line or bar.
6. Max 20 categories per axis.
7. For dashboard/overview queries return at least 2-3 DIVERSE charts
   (e.g. one line + one bar, not two bars on the same column).

TABLE RULES — READ CAREFULLY:

HARD LIMIT: Return AT MOST 2 tables total. Never exceed this.

NEVER generate a summary table that produces only 1 row.
Single-value KPIs (total revenue, avg price, count) are already shown as
scorecards — DO NOT duplicate them as tables. Tables must have multiple rows.

ONLY generate a table when it shows grouped/cross-tabulated data that a
chart cannot easily show. Good table candidates:
  - pivot of category × category (e.g. gender × product)
  - grouped breakdown with 3+ metrics per group

ALWAYS prefer adding another chart over adding a table.
If you are unsure whether to add a table, add a chart instead.

ALWAYS generate at least 1 pivot table when the query contains any of:
  breakdown, by category, by region, by segment, by product,
  compare, comparison, across, per, group, distribution table

TABLE SCHEMAS:

1. Pivot table — use to cross-tabulate two categorical dimensions:
{
  "type": "pivot",
  "index": "categorical_column",
  "columns": "another_categorical_column_or_null",
  "values": "numeric_column",
  "aggregation": "sum | mean | count",
  "title": "clear title"
}

2. Summary table — ONLY use when grouped (multiple rows).
   NEVER use for a single scalar value.
{
  "type": "summary",
  "values": "numeric_column",
  "aggregation": "sum | mean | count | min | max",
  "title": "clear title"
}

RULES:
- "values" must be a single numeric column name (a string, NOT a list).
- "index" is required for pivot.
- MAX 2 tables. Prefer charts over tables when in doubt.
- For dashboard/analyze/overview queries: return at least 2-3 charts + 0-2 tables.
"""

REPAIR_PROMPT = """
The JSON you returned failed schema validation.

Error: {error}

Original response:
{original}

Fix ALL validation errors and return ONLY valid JSON matching this schema:
{{
  "charts": [
    {{
      "type": "bar | line | scatter | histogram | pie",
      "x": "column_name",
      "y": "column_name or null for histogram",
      "color": null,
      "aggregation": "sum | mean | count | none",
      "time_granularity": "none",
      "layout_size": "medium",
      "title": "title"
    }}
  ],
  "tables": []
}}

Available columns: {columns}
"""


def _strip_fences(text: str) -> str:
    return re.sub(r"```(?:json)?", "", text).replace("```", "").strip()


def _parse_and_validate(raw: str) -> LLMResponseSchema:
    data = json.loads(_strip_fences(raw))
    return LLMResponseSchema.model_validate(data)


class LLMClient:
    MAX_RETRIES = 2   # normal parse retries
    MAX_REPAIRS = 1   # repair-loop attempts after retries exhausted

    def __init__(self):
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    # ── public ───────────────────────────────────────────────────────
    def get_chart_config(
        self, col_dt_list: list, sample: str, stats: str, query: str
    ) -> LLMResponseSchema:
        """
        Returns a validated LLMResponseSchema.
        Attempt order:
          1. MAX_RETRIES normal LLM calls
          2. MAX_REPAIRS repair calls (error + original sent back to model)
          3. Deterministic fallback built from column metadata — never crashes
        """
        user_msg   = f"Columns: {col_dt_list}\n\nSample:\n{sample}\n\nStats:\n{stats}\n\nQuery:\n{query}"
        last_raw   = ""
        last_error = ""

        # --- normal retries ---
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                last_raw = self._call([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ])
                schema = _parse_and_validate(last_raw)
                logger.info("LLM config validated (attempt %d)", attempt)
                return schema
            except (json.JSONDecodeError, ValidationError, KeyError, IndexError) as exc:
                last_error = str(exc)
                logger.warning("LLM attempt %d failed: %s", attempt, last_error)

        # --- repair loop ---
        columns = [col for col, _ in col_dt_list]
        for rep in range(1, self.MAX_REPAIRS + 1):
            try:
                logger.info("Attempting LLM repair (repair %d)", rep)
                repair_msg = REPAIR_PROMPT.format(
                    error=last_error, original=last_raw, columns=columns
                )
                last_raw = self._call([
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": last_raw},
                    {"role": "user",      "content": repair_msg},
                ])
                schema = _parse_and_validate(last_raw)
                logger.info("LLM repair succeeded (repair %d)", rep)
                return schema
            except (json.JSONDecodeError, ValidationError, KeyError, IndexError) as exc:
                last_error = str(exc)
                logger.warning("LLM repair %d failed: %s", rep, last_error)

        # --- deterministic fallback ---
        logger.error("All LLM attempts failed — using deterministic fallback config")
        return self._fallback_config(col_dt_list)

    # ── private ──────────────────────────────────────────────────────
    def _call(self, messages: list) -> str:
        response = self._client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b:free",
            messages=messages,
        )
        return response.choices[0].message.content

    @staticmethod
    def _fallback_config(col_dt_list: list) -> LLMResponseSchema:
        """
        Build a safe minimal config from column dtypes alone.
        Priority: datetime+numeric -> line | two numerics -> scatter | one numeric -> histogram
        """
        cols     = {col: str(dt) for col, dt in col_dt_list}
        num_cols = [c for c, dt in cols.items() if "int" in dt or "float" in dt]
        dt_cols  = [c for c, dt in cols.items() if "datetime" in dt or "date" in dt.lower()]

        if dt_cols and num_cols:
            raw = [{
                "type": "line", "x": dt_cols[0], "y": num_cols[0],
                "aggregation": "sum", "time_granularity": "month",
                "layout_size": "large", "title": f"{num_cols[0]} over time",
            }]
        elif len(num_cols) >= 2:
            raw = [{
                "type": "scatter", "x": num_cols[0], "y": num_cols[1],
                "aggregation": "none", "time_granularity": "none",
                "layout_size": "medium", "title": f"{num_cols[0]} vs {num_cols[1]}",
            }]
        elif num_cols:
            raw = [{
                "type": "histogram", "x": num_cols[0], "y": None,
                "aggregation": "none", "time_granularity": "none",
                "layout_size": "medium", "title": f"Distribution of {num_cols[0]}",
            }]
        else:
            first = list(cols.keys())[0]
            raw = [{
                "type": "histogram", "x": first, "y": None,
                "aggregation": "none", "time_granularity": "none",
                "layout_size": "medium", "title": f"Distribution of {first}",
            }]

        return LLMResponseSchema.model_validate({"charts": raw, "tables": []})


# ─────────────────────────────────────────────
# 3. CONFIG NORMALIZER  (business rules, immutable output)
# ─────────────────────────────────────────────

class ChartConfigNormalizer:
    """
    Pure normalizer — never mutates the source DataFrame.
    Synthetic columns are passed as a Series in cfg["_synthetic"]
    so DataTransformer can join them on demand.
    """

    def __init__(self, df: pd.DataFrame):
        self._df         = df
        self._columns    = set(df.columns)
        self._num_cols   = set(df.select_dtypes(include="number").columns)
        self._cardinality: dict[str, int] = {}

    def _card(self, col: str) -> int:
        if col not in self._cardinality:
            self._cardinality[col] = self._df[col].nunique()
        return self._cardinality[col]

    def normalize(self, schema: ChartConfigSchema) -> dict | None:
        chart_type = schema.type
        x          = schema.x
        y          = schema.y
        color      = schema.color
        agg        = schema.aggregation

        if x not in self._columns:
            logger.debug("Dropping chart: x=%r not in columns", x)
            return None

        # ── pie without y: build synthetic count Series, never touch self._df ──
        synthetic: pd.Series | None = None
        synthetic_name: str | None  = None

        if chart_type == "pie" and y is None:
            synthetic_name = f"_count_{x}"
            # A Series of 1s — transformer will groupby-sum it into value_counts
            synthetic = pd.Series(1, index=self._df.index, name=synthetic_name, dtype="int64")
            y   = synthetic_name
            agg = "count"
            logger.debug("Pie chart: synthetic count column %r for x=%r (no mutation)", y, x)
        # ──────────────────────────────────────────────────────────────────────

        # effective column set = real columns + any synthetic
        effective_cols = self._columns | ({synthetic_name} if synthetic_name else set())
        effective_nums = self._num_cols | ({synthetic_name} if synthetic_name else set())

        if chart_type != "histogram" and (y is None or y not in effective_cols):
            logger.debug("Dropping chart: y=%r not in columns", y)
            return None

        if chart_type == "histogram" and x not in self._num_cols:
            return None
        if chart_type == "scatter" and (self._card(x) < 5 or (y and self._card(y) < 5)):
            return None
        if chart_type in ("bar", "pie") and y and y not in effective_nums:
            return None

        if color and (color not in self._columns or self._card(color) > 6):
            color = None

        if chart_type in ("bar", "pie", "line") and agg == "none":
            agg = "sum"

        return {
            "type":             chart_type,
            "x":                x,
            "y":                y,
            "color":            color,
            "aggregation":      agg,
            "time_granularity": schema.time_granularity,
            "layout_size":      schema.layout_size,
            "title":            schema.title,
            "limit_top":        chart_type in ("bar", "pie") and self._card(x) > 20,
            # synthetic Series passed through config — never written to source df
            "_synthetic":       synthetic,
        }


# ─────────────────────────────────────────────
# 4. DATA TRANSFORMER  (one aggregation block, minimal copies)
# ─────────────────────────────────────────────

_TIME_FREQ = {"month": "MS", "year": "YS", "week": "W-MON", "day": "D"}


class DataTransformer:
    def __init__(self, df: pd.DataFrame):
        self._source = df
        # time-column detection is O(n) — cache per column name
        self._time_cache: dict[str, bool] = {}

    def transform(self, cfg: dict) -> pd.DataFrame:
        x, y    = cfg["x"], cfg["y"]
        agg     = cfg["aggregation"]
        color   = cfg["color"]
        is_time = self._is_time(x, cfg["type"])

        # histogram / scatter: select only needed columns — no full copy
        if cfg["type"] == "histogram":
            return self._source[[x]].dropna()

        if cfg["type"] == "scatter":
            return self._source[[x, y]].drop_duplicates()

        # copy only the real columns this chart needs, then attach any synthetic
        real_cols = [c for c in ({x, y} | ({color} if color else set()))
                     if c in self._source.columns]
        df = self._source[real_cols].copy()

        synthetic: pd.Series | None = cfg.get("_synthetic")
        if synthetic is not None and synthetic.name not in df.columns:
            df[synthetic.name] = synthetic.values

        if is_time:
            self._parse_time_inplace(df, x)
            self._granularise_inplace(df, x, cfg["time_granularity"])

        # ── single aggregation block ──────────────────────────────────
        group_keys = [x] if not color else [x, color]
        if agg != "none":
            df = df.groupby(group_keys)[y].agg(agg).reset_index()
        # ─────────────────────────────────────────────────────────────

        if is_time:
            df = self._fill_gaps(df, x, y, cfg["time_granularity"], color=color)
            if cfg["type"] == "line":
                # rolling smoothing: for multi-series apply per-group so series
                # don't bleed into each other across the color boundary
                if color and color in df.columns:
                    df[y] = (
                        df.groupby(color)[y]
                        .transform(lambda s: s.rolling(2, min_periods=1).mean())
                    )
                else:
                    df[y] = df[y].rolling(2, min_periods=1).mean()
        else:
            df = self._limit_categories(df, cfg, x, y)

        df = self._safe_sort(df, x)

        if x in df.columns and pd.api.types.is_datetime64_any_dtype(df[x]):
            df[x] = df[x].astype(str)

        return df

    # ── helpers ──────────────────────────────────────────────────────

    def _is_time(self, col: str, chart_type: str) -> bool:
        if chart_type == "scatter":
            return False
        if col in self._time_cache:
            return self._time_cache[col]

        series = self._source[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            self._time_cache[col] = True
            return True
        if series.dtype != object:
            self._time_cache[col] = False
            return False

        parsed = pd.to_datetime(series, errors="coerce")
        result = parsed.notna().sum() / max(len(series), 1) >= 0.80
        self._time_cache[col] = result
        return result

    @staticmethod
    def _parse_time_inplace(df: pd.DataFrame, col: str) -> None:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    @staticmethod
    def _granularise_inplace(df: pd.DataFrame, col: str, granularity: str) -> None:
        freq_map = {"month": "M", "year": "Y", "week": "W", "day": "D"}
        freq = freq_map.get(granularity)
        if freq:
            df[col] = df[col].dt.to_period(freq).dt.to_timestamp()

    @staticmethod
    def _fill_gaps(
        df: pd.DataFrame, x: str, y: str, granularity: str,
        color: str | None = None,
    ) -> pd.DataFrame:
        """
        Fill time-series gaps with zeros.

        Single-series: straightforward asfreq on the x index.
        Multi-series:  asfreq must be applied per-group, otherwise the
                       (x, color) composite index breaks asfreq entirely.
                       Each group is reindexed against the GLOBAL date
                       range so all series share the same x-axis.
        """
        freq = _TIME_FREQ.get(granularity)
        if not freq:
            return df

        if color and color in df.columns:
            # build the global date range from the full aggregated data
            try:
                full_range = pd.date_range(
                    start=df[x].min(), end=df[x].max(), freq=freq
                )
            except Exception:
                return df

            filled_groups = []
            for group_name, group in df.groupby(color):
                try:
                    group = (
                        group.set_index(x)[y]
                        .reindex(full_range, fill_value=0)
                        .reset_index()
                        .rename(columns={"index": x})
                    )
                    group[color] = group_name
                    filled_groups.append(group)
                except Exception:
                    filled_groups.append(group)

            return pd.concat(filled_groups, ignore_index=True) if filled_groups else df

        # single-series path
        try:
            df = df.set_index(x).asfreq(freq).fillna(0).reset_index()
        except Exception:
            pass
        return df

    @staticmethod
    def _limit_categories(df: pd.DataFrame, cfg: dict, x: str, y: str) -> pd.DataFrame:
        if cfg.get("limit_top"):
            return df.nlargest(10, y)
        if df[x].nunique() > 20:
            return df.nlargest(20, y)
        return df

    @staticmethod
    def _safe_sort(df: pd.DataFrame, col: str) -> pd.DataFrame:
        try:
            return df.sort_values(by=col)
        except Exception:
            return df


# ─────────────────────────────────────────────
# 5. CHART BUILDER
# ─────────────────────────────────────────────

class ChartBuilder:
    def __init__(self, transformer: DataTransformer):
        self._transformer = transformer

    def build(self, cfg: dict) -> dict | None:
        try:
            df = self._transformer.transform(cfg)
            if df.empty:
                return None

            chart_type = cfg["type"]
            x, y       = cfg["x"], cfg["y"]
            color      = cfg["color"]

            if chart_type == "histogram":
                return {
                    "type": "histogram", "title": cfg["title"],
                    "values": df[x].tolist(), "x_label": x,
                    "layout_size": cfg["layout_size"],
                }

            if color and color in df.columns:
                return self._multi_series(df, cfg)

            if chart_type == "scatter" and (df[x].nunique() < 5 or df[y].nunique() < 5):
                return None

            return {
                "type": chart_type, "title": cfg["title"],
                "x": df[x].tolist(), "y": df[y].tolist(),
                "x_label": x, "y_label": y,
                "layout_size": cfg["layout_size"],
            }

        except Exception as exc:
            logger.error("Chart build failed for %r: %s", cfg.get("title"), exc)
            return None

    @staticmethod
    def _multi_series(df: pd.DataFrame, cfg: dict) -> dict:
        x, y, color = cfg["x"], cfg["y"], cfg["color"]
        return {
            "type": cfg["type"], "title": cfg["title"],
            "series": [
                {"name": str(n), "x": g[x].tolist(), "y": g[y].tolist()}
                for n, g in df.groupby(color)
            ],
            "x_label": x, "y_label": y,
            "layout_size": cfg["layout_size"],
        }


# ─────────────────────────────────────────────
# 6. TABLE BUILDER
# ─────────────────────────────────────────────

class TableBuilder:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def build_all(self, table_configs: list) -> list:
        """
        Accepts either list[TableConfigSchema] (from LLM path) or
        list[dict] (from _default_tables fallback path).
        Plain dicts are passed through as-is since they are already rendered.
        """
        results = []
        for cfg in table_configs:
            if isinstance(cfg, dict):
                # already a rendered table from _default_tables — pass through
                results.append(cfg)
            elif (r := self._build_one(cfg)):
                results.append(r)
        return results

    def _build_one(self, cfg) -> dict | None:
        """cfg is a TableConfigSchema instance."""
        try:
            df    = self._df
            title = cfg.title
            t     = cfg.type
            vals  = cfg.values
            agg   = cfg.aggregation

            # column guard — log and skip rather than raise
            if vals not in df.columns:
                logger.warning("Table skipped: values column %r not in DataFrame", vals)
                return None

            if t == "pivot":
                idx  = cfg.index
                cols = cfg.columns

                if idx not in df.columns:
                    logger.warning("Table skipped: index column %r not in DataFrame", idx)
                    return None
                # cols is optional for pivot — None means no column grouping
                if cols and cols not in df.columns:
                    logger.warning("Pivot: columns %r not found, building without it", cols)
                    cols = None

                pivot = (
                    pd.pivot_table(
                        df,
                        index=idx,
                        columns=cols if cols else None,
                        values=vals,
                        aggfunc=agg,
                        observed=True,       # silence pandas FutureWarning
                    )
                    .fillna(0)
                    .reset_index()
                    .head(20)
                )
                # flatten MultiIndex columns produced when cols is set
                pivot.columns = [
                    str(c) if not isinstance(c, tuple) else "_".join(str(x) for x in c if x)
                    for c in pivot.columns
                ]
                return {"title": title, "data": pivot.to_dict(orient="records")}

            if t == "summary":
                if not pd.api.types.is_numeric_dtype(df[vals]):
                    logger.warning("Summary skipped: %r is not numeric", vals)
                    return None
                return {"title": title, "data": [{vals: float(df[vals].agg(agg))}]}

        except Exception as exc:
            logger.error("Table build failed (%s / %s): %s", cfg.type, cfg.values, exc)

        return None


# ─────────────────────────────────────────────
# 7. SCORECARD BUILDER  (deterministic — no LLM)
# ─────────────────────────────────────────────

_SCORECARD_MAX = 5   # hard cap per your spec

_AGG_PRIORITY: list[tuple[str, str, str]] = [
    # (query_hint, pandas_agg, label_template)
    # Checked in order; first match wins per column.
    ("revenue",  "sum",  "Total {col}"),
    ("sales",    "sum",  "Total {col}"),
    ("cost",     "sum",  "Total {col}"),
    ("profit",   "sum",  "Total {col}"),
    ("qty",      "sum",  "Total {col}"),
    ("quantity", "sum",  "Total {col}"),
    ("amount",   "sum",  "Total {col}"),
    ("price",    "mean", "Avg {col}"),
    ("rate",     "mean", "Avg {col}"),
    ("score",    "mean", "Avg {col}"),
    ("discount", "mean", "Avg {col}"),
    ("count",    "count","Count of {col}"),
]
_DEFAULT_AGG = ("sum", "Total {col}")


def _pick_agg(col: str) -> tuple[str, str]:
    """
    Choose aggregation + label for a column by name heuristic.
    Returns (pandas_agg, label_template).
    """
    col_lower = col.lower()
    for hint, agg, label in _AGG_PRIORITY:
        if hint in col_lower:
            return agg, label
    return _DEFAULT_AGG


def _fmt_value(v: float) -> str:
    """Human-readable number: 1 234 567 -> '1.23M', 12345 -> '12.3K', else rounded."""
    abs_v = abs(v)
    if abs_v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if abs_v >= 1_000:
        return f"{v / 1_000:.1f}K"
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}"


class ScorecardBuilder:
    """
    Builds 3–5 KPI scorecards deterministically from numeric columns.
    No LLM involved — fast, token-free, always correct.

    Output format per card:
        {"label": str, "value": str, "raw": float, "column": str, "aggregation": str}
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._num_cols: list[str] = df.select_dtypes(include="number").columns.tolist()

    # ── public ────────────────────────────────────────────────────────

    def build(self) -> list[dict]:
        """Return a list of scorecard dicts (3–5 items, rendered first in UI)."""
        if not self._num_cols:
            logger.info("ScorecardBuilder: no numeric columns — returning empty")
            return []

        candidates = self._rank_columns()
        cards = []
        for col in candidates[:_SCORECARD_MAX]:
            card = self._build_one(col)
            if card:
                cards.append(card)

        logger.info("ScorecardBuilder: produced %d scorecard(s)", len(cards))
        return cards

    # ── private ───────────────────────────────────────────────────────

    def _rank_columns(self) -> list[str]:
        """
        Rank numeric columns by usefulness:
          1. Columns whose names match a known KPI hint (revenue, sales …) come first.
          2. Tie-break: higher variance = more interesting.
        Drops columns that are effectively IDs (all-unique integers).
        """
        df     = self._df
        ranked = []
        for col in self._num_cols:
            n_unique = df[col].nunique()
            n_rows   = len(df)
            # skip likely ID columns (>95 % unique integers)
            if n_unique / max(n_rows, 1) > 0.95 and pd.api.types.is_integer_dtype(df[col]):
                continue
            hint_score = sum(
                1 for hint, *_ in _AGG_PRIORITY if hint in col.lower()
            )
            variance = float(df[col].var(ddof=0)) if n_unique > 1 else 0.0
            ranked.append((col, hint_score, variance))

        # sort: higher hint_score first, then higher variance
        ranked.sort(key=lambda t: (t[1], t[2]), reverse=True)
        return [col for col, *_ in ranked]

    def _build_one(self, col: str) -> dict | None:
        series = self._df[col].dropna()
        if series.empty:
            return None

        agg, label_tpl = _pick_agg(col)

        try:
            if agg == "sum":
                raw = float(series.sum())
            elif agg == "mean":
                raw = float(series.mean())
            else:           # "count"
                raw = float(series.count())
        except Exception as exc:
            logger.warning("Scorecard failed for column %r: %s", col, exc)
            return None

        label = label_tpl.format(col=col)
        return {
            "label":       label,
            "value":       _fmt_value(raw),
            "raw":         raw,
            "column":      col,
            "aggregation": agg,
        }


# ─────────────────────────────────────────────
# 8. LRU RESULT CACHE  (was section 7)
# ─────────────────────────────────────────────

def _df_fingerprint(df: pd.DataFrame) -> str:
    """
    Full-content fingerprint using pandas' own row hashing.
    pd.util.hash_pandas_object hashes every cell of every row, then we
    XOR-sum all row hashes into a single uint64 and fold it into MD5.
    Cost: O(n*c) — acceptable because this runs once per request, not
    per chart. For very large DataFrames (>500k rows) a stratified
    sample of 10k rows gives collision resistance that is good enough
    in practice while keeping latency under 50 ms.
    """
    if len(df) == 0:
        return hashlib.md5(b"empty").hexdigest()

    sample_df = df if len(df) <= 500_000 else df.sample(10_000, random_state=0)
    row_hashes = pd.util.hash_pandas_object(sample_df, index=False)
    content_hash = format(int(row_hashes.sum()) & 0xFFFF_FFFF_FFFF_FFFF, "016x")
    meta = f"{df.shape[0]}x{df.shape[1]}|{','.join(df.columns)}"
    return hashlib.md5(f"{meta}|{content_hash}".encode()).hexdigest()


def _cache_key(df: pd.DataFrame, query: str) -> str:
    return hashlib.md5(
        f"{_df_fingerprint(df)}|{query.strip().lower()}".encode()
    ).hexdigest()


class _LRUCache:
    def __init__(self, max_size: int = 64):
        self._store: dict = {}
        self._max = max_size

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value) -> None:
        if len(self._store) >= self._max:
            del self._store[next(iter(self._store))]
        self._store[key] = value


_result_cache = _LRUCache(max_size=64)


# ─────────────────────────────────────────────
# 9. CHART GENERATOR  (thin orchestrator)
# ─────────────────────────────────────────────

class ChartGenerator:
    def __init__(self, data: pd.DataFrame):
        self.data                = data
        self._col_dt_list        = list(zip(data.columns, data.dtypes))
        self._llm                = LLMClient()
        self._normalizer         = ChartConfigNormalizer(data)
        self._transformer        = DataTransformer(data)
        self._builder            = ChartBuilder(self._transformer)
        self._table_builder      = TableBuilder(data)
        self._scorecard_builder  = ScorecardBuilder(data)  # deterministic, no LLM
        logger.info("ChartGenerator initialised (%d rows, %d cols)", *data.shape)

    # Intent keywords that should ALWAYS produce tables even if LLM forgets
    _BROAD_INTENT = {
        # explicit dashboard/analysis words
        "dashboard", "analyze", "analyse", "analysis", "analyses",
        "overview", "report", "explore", "exploration",
        "insights", "insight", "understand", "examine", "investigate",
        "full", "complete", "everything", "all", "whole", "entire",
        # natural-language phrasings
        "show me", "tell me", "give me", "what is", "what are",
        "create dashboard", "make dashboard", "build dashboard",
        "deep dive", "deep-dive",
        # single-word verbs people type
        "summarize", "summarise",
    }
    _SUMMARY_INTENT = {
        "total", "average", "mean", "count", "sum", "max", "min",
        "kpi", "metric", "performance", "summary", "summaries",
        "aggregate", "aggregation", "statistics", "stats",
    }
    _PIVOT_INTENT = {
        "breakdown", "break down", "by category", "by region", "by segment",
        "by product", "by month", "by year", "by date", "by customer",
        "compare", "comparison", "across", "per group", "per category",
        "distribution table", "tabulate", "cross tab", "crosstab",
    }

    @staticmethod
    def _query_matches(query: str, keywords: set) -> bool:
        q = query.lower()
        return any(kw in q for kw in keywords)

    def _augment_query(self, query: str) -> str:
        """
        For broad/dashboard queries inject a mandatory hint with concrete
        column names. This fires BEFORE the LLM call so the model sees it.
        We now push charts hard and cap tables at 2 (pivot-only, no scalars).
        """
        if not self._needs_table(query):
            return query

        num_cols = list(self.data.select_dtypes(include="number").columns[:3])
        cat_cols = list(self.data.select_dtypes(include=["object", "category"]).columns[:2])
        dt_cols  = [c for c in self.data.columns
                    if "date" in c.lower() or "time" in c.lower()
                    or str(self.data[c].dtype).startswith("datetime")]

        chart_hint = (
            f"Date/time columns (use for line charts): {dt_cols}\n"
            f"Numeric columns: {num_cols}\n"
            f"Categorical columns: {cat_cols}\n"
        ) if dt_cols else (
            f"Numeric columns: {num_cols}\n"
            f"Categorical columns: {cat_cols}\n"
        )

        hint = (
            "\n\n[MANDATORY INSTRUCTION - DO NOT IGNORE]\n"
            "Return at least 2-3 diverse charts (line, bar, pie — not all the same type).\n"
            + chart_hint +
            "Tables: MAX 2. Only pivot tables with multiple rows. "
            "NEVER generate a single-scalar summary table — those are shown as scorecards already.\n"
            "Every pivot table MUST have a non-null 'index' and 'values' field."
        )
        logger.info("Augmenting query with chart+table hint for: %r", query[:50])
        return query + hint

    def _needs_table(self, query: str) -> bool:
        return (
            self._query_matches(query, self._BROAD_INTENT)
            or self._query_matches(query, self._SUMMARY_INTENT)
            or self._query_matches(query, self._PIVOT_INTENT)
        )

    def _default_tables(self, query: str) -> list:
        """
        Deterministic table builder — fires when LLM ignores table instructions.
        Rules:
          - MAX 2 tables
          - Only grouped/pivot tables (never single-row scalar summaries — those are scorecards)
          - If no categorical column exists, return [] rather than a scalar table
        """
        df      = self.data
        tables  = []

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return []

        # pick up to 2 categorical columns with useful cardinality (2–30 unique values)
        cat_cols = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if 2 <= df[c].nunique() <= 30
        ]

        if not cat_cols:
            # No categorical columns → grouped tables impossible → skip entirely
            # (scalars are already covered by scorecards)
            logger.info("_default_tables: no categorical columns — skipping (scorecards cover KPIs)")
            return []

        # Table 1: best num col grouped by best cat col
        cat1, num1 = cat_cols[0], num_cols[0]
        try:
            tbl = (
                df.groupby(cat1, observed=True)[num1]
                .agg(["sum", "mean", "count"])
                .reset_index()
                .rename(columns={"sum": f"Total {num1}", "mean": f"Avg {num1}", "count": "Count"})
                .sort_values(f"Total {num1}", ascending=False)
                .head(20)
            )
            tbl[f"Total {num1}"] = tbl[f"Total {num1}"].round(2)
            tbl[f"Avg {num1}"]   = tbl[f"Avg {num1}"].round(2)
            tables.append({"title": f"{num1} by {cat1}", "data": tbl.to_dict(orient="records")})
        except Exception as exc:
            logger.error("Default table 1 failed: %s", exc)

        # Table 2: pivot cat1 × cat2 if a second cat col exists, else second num col by cat1
        if len(tables) < 2:
            if len(cat_cols) >= 2:
                cat2 = cat_cols[1]
                num_for_pivot = num_cols[0]
                try:
                    pivot = (
                        pd.pivot_table(df, index=cat1, columns=cat2,
                                       values=num_for_pivot, aggfunc="sum", observed=True)
                        .fillna(0).reset_index().head(20)
                    )
                    pivot.columns = [
                        str(c) if not isinstance(c, tuple) else "_".join(str(x) for x in c if x)
                        for c in pivot.columns
                    ]
                    tables.append({
                        "title": f"{num_for_pivot} by {cat1} and {cat2}",
                        "data": pivot.to_dict(orient="records"),
                    })
                except Exception as exc:
                    logger.error("Default pivot table failed: %s", exc)
            elif len(num_cols) >= 2:
                num2 = num_cols[1]
                try:
                    tbl2 = (
                        df.groupby(cat1, observed=True)[num2]
                        .agg(["sum", "mean"])
                        .reset_index()
                        .rename(columns={"sum": f"Total {num2}", "mean": f"Avg {num2}"})
                        .sort_values(f"Total {num2}", ascending=False)
                        .head(20)
                    )
                    tbl2[f"Total {num2}"] = tbl2[f"Total {num2}"].round(2)
                    tbl2[f"Avg {num2}"]   = tbl2[f"Avg {num2}"].round(2)
                    tables.append({"title": f"{num2} by {cat1}", "data": tbl2.to_dict(orient="records")})
                except Exception as exc:
                    logger.error("Default table 2 failed: %s", exc)

        logger.info("Built %d deterministic default table(s) for query: %r", len(tables), query[:60])
        return tables

    def _filter_table_configs(self, table_cfgs: list) -> list:
        """
        Remove table configs that would produce single-row scalar summaries.
        These duplicate scorecard data and add no value.

        A summary table is scalar if it has no 'index' column — meaning it
        aggregates the whole column into one number. Pivot tables always have
        an index, so they always pass through.
        """
        filtered = []
        for cfg in table_cfgs:
            if isinstance(cfg, dict):
                # Already-rendered dict from _default_tables — always grouped, keep it
                filtered.append(cfg)
            elif cfg.type == "summary":
                # summary tables with no index = single scalar row = scorecard duplicate
                # Keep only if we somehow have an index (shouldn't happen per schema, but be safe)
                logger.info(
                    "Dropping scalar summary table %r — already covered by scorecards", cfg.title
                )
                # Drop it
            else:
                # pivot or any other type — keep
                filtered.append(cfg)
        return filtered

    def generate(self, query: str) -> dict:
        # cache check
        key = _cache_key(self.data, query)
        if (cached := _result_cache.get(key)) is not None:
            logger.info("Cache hit for query: %r", query[:60])
            return cached

        # augment broad queries with mandatory table hint before LLM call
        effective_query = self._augment_query(query)

        sample     = self.data.head(5).to_string()
        stats      = self.data.describe(include="all").to_string()
        llm_schema = self._llm.get_chart_config(self._col_dt_list, sample, stats, effective_query)

        # ── scorecards first (deterministic, no LLM, max 5) ──────────────
        scorecards = self._scorecard_builder.build()

        charts     = self._build_charts(llm_schema)
        table_cfgs = llm_schema.tables

        # ── Drop single-row scalar summary tables (scorecards already show these) ──
        table_cfgs = self._filter_table_configs(table_cfgs)

        # ── Hard cap: max 2 tables ────────────────────────────────────────────────
        table_cfgs = table_cfgs[:2]

        # Safety net: LLM returned no tables but query clearly wants them
        if not table_cfgs and self._needs_table(query):
            logger.info("LLM returned no tables for broad query — injecting defaults")
            table_cfgs = self._default_tables(query)

        tables = self._table_builder.build_all(table_cfgs)
        # scorecards key comes first so the UI renders them at the top
        result = {"scorecards": scorecards, "charts": charts, "tables": tables}

        _result_cache.set(key, result)
        return result

    def _build_charts(self, llm_schema: LLMResponseSchema) -> list:
        charts = []
        for chart_schema in llm_schema.charts:
            cfg = self._normalizer.normalize(chart_schema)
            if cfg is None:
                continue
            if (result := self._builder.build(cfg)):
                charts.append(result)

        return charts or self._fallback_chart()

    def _fallback_chart(self) -> list:
        num_cols = self.data.select_dtypes(include="number").columns
        if not len(num_cols):
            return []
        col    = num_cols[0]
        values = self.data[col].dropna()
        if len(values) > 1000:
            values = values.sample(1000, random_state=0)
        logger.info("Using fallback histogram for column %r", col)
        return [{
            "type": "histogram", "title": f"Distribution of {col}",
            "values": values.tolist(), "x_label": col, "layout_size": "medium",
        }]