import asyncio
import logging
logger = logging.getLogger(__name__)
import pandas as pd
import concurrent.futures

# Single shared executor — sized for CPU work across concurrent requests.
# Rule of thumb: (os.cpu_count() or 4) * 2 for mixed I/O+CPU work.
# Adjust max_workers to your pod's vCPU count.
import os
EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=(os.cpu_count() or 4) * 2,
    thread_name_prefix="chart-worker",
)

from app.pipeline.llm_client import LLMClient
from app.pipeline.normalizer import ChartConfigNormalizer
from app.pipeline.transformer import DataTransformer
from app.pipeline.chart_builder import ChartBuilder
from app.pipeline.table_builder import TableBuilder
from app.pipeline.scorecard import ScorecardBuilder
from app.schemas.chart_schema import LLMResponseSchema
from app.utils.cache import _cache_key, _result_cache


def _set_loop_executor(loop: asyncio.AbstractEventLoop | None = None) -> None:
    """
    Call once at startup (e.g. in your FastAPI lifespan) so that
    asyncio.to_thread() uses the same sized pool as _build_charts.

        from app.pipeline.chart_service import EXECUTOR, _set_loop_executor
        _set_loop_executor()          # or pass loop explicitly
    """
    (loop or asyncio.get_event_loop()).set_default_executor(EXECUTOR)


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
                logger.info(
                    "Dropping scalar summary table %r — already covered by scorecards", cfg.title
                )
                # Drop it
            else:
                # pivot or any other type — keep
                filtered.append(cfg)
        return filtered

    async def generate(self, query: str) -> dict:
        # cache check
        key = _cache_key(self.data, query)
        if (cached := _result_cache.get(key)) is not None:
            logger.info("Cache hit for query: %r", query[:60])
            return cached

        # augment broad queries with mandatory table hint before LLM call
        effective_query = self._augment_query(query)

        # FIX: run _prepare_llm_input on the shared EXECUTOR so it doesn't
        # consume a slot from the default loop executor
        loop = asyncio.get_running_loop()
        sample, stats = await loop.run_in_executor(EXECUTOR, self._prepare_llm_input)

        llm_schema = await self._llm.get_chart_config(self._col_dt_list, sample, stats, effective_query)

        # FIX: run CPU processing on the same shared EXECUTOR
        result = await loop.run_in_executor(
            EXECUTOR,
            self._process_sync,
            llm_schema,
            query,
        )

        _result_cache.set(key, result)
        return result

    def _prepare_llm_input(self):
        sample = self.data.head(5).to_string()
        stats  = self.data.describe(include="all").to_string()
        return sample, stats

    def _process_sync(self, llm_schema, query):
        logger.info("START processing")
        import time
        time.sleep(3)

        # scorecards
        scorecards = self._scorecard_builder.build()

        # charts
        charts = self._build_charts(llm_schema)

        # tables
        table_cfgs = llm_schema.tables
        table_cfgs = self._filter_table_configs(table_cfgs)
        table_cfgs = table_cfgs[:2]

        if not table_cfgs and self._needs_table(query):
            logger.info("LLM returned no tables for broad query — injecting defaults")
            table_cfgs = self._default_tables(query)

        tables = self._table_builder.build_all(table_cfgs)

        result = {"scorecards": scorecards, "charts": charts, "tables": tables}

        logger.info(
            f"Charts generated: {len(result['charts'])}  |  Tables: {len(result['tables'])}"
        )
        logger.info("END processing")

        return result

    def _build_charts(self, llm_schema: LLMResponseSchema) -> list:
        """
        FIX: reuse the module-level EXECUTOR instead of spawning a new
        ThreadPoolExecutor() on every call. Creating a fresh pool per request
        was the second source of thread starvation.
        """
        def process_one(chart_schema):
            cfg = self._normalizer.normalize(chart_schema)
            if cfg is None:
                return None
            return self._builder.build(cfg)

        # submit all chart jobs to the shared pool; futures resolve in parallel
        futures = [EXECUTOR.submit(process_one, cs) for cs in llm_schema.charts]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

        charts = [r for r in results if r is not None]
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