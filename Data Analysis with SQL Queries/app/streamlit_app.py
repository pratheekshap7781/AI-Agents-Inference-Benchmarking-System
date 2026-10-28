"""
AI Agents Inference Benchmarking System — Streamlit dashboard.

Run from the "Data Analysis with SQL Queries" folder with:
    streamlit run app/streamlit_app.py

This file only renders the UI. All benchmarking logic lives in the real
project files (the CrewAI/LangGraph scripts, utils/metrics_collector.py)
and in app/{framework_loader,runner,config_validate,metrics_loader}.py,
which call those real files rather than reimplementing them.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root on sys.path for `app.*` imports
load_dotenv(Path(__file__).parent.parent / ".env")  # Streamlit doesn't auto-load .env like the original scripts do

from app import config_validate, known_questions, metrics_loader, runner
from app.framework_loader import dataset_status, DB_PATH, QUESTIONS_PATH, EXPECTED_RESULTS_PATH, RESULTS_DIR

st.set_page_config(
    page_title="AI Agents Inference Benchmarking System",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------------------------------
# Credentials: env vars / st.secrets only. Never hardcoded, never displayed.
# --------------------------------------------------------------------------
def get_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            key = ""
    return key


def ds_status():
    return dataset_status()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("AI Agents Benchmarking")
    st.caption("CrewAI vs LangGraph — SQL data-analysis tasks")

    st.markdown("### API key")
    existing_key = get_groq_key()
    if existing_key:
        st.success("GROQ_API_KEY loaded from environment/secrets.")
    else:
        st.warning("No GROQ_API_KEY found in environment or secrets.")
        pasted_key = st.text_input("Paste a Groq API key for this session", type="password")
        if pasted_key:
            os.environ["GROQ_API_KEY"] = pasted_key
            st.success("Key set for this session (not saved to disk).")

    st.markdown("### Dataset status")
    status = ds_status()
    st.write(("✅" if status["db_exists"] else "❌") + " `data/amazon_cleaned.db`")
    st.write(("✅" if status["questions_exists"] else "❌") + " `data/questions.py`")
    st.write(("✅" if status["expected_results_exists"] else "❌") + " `data/expected_results.csv`")
    if not status["db_exists"]:
        st.caption(
            "Live execution (Tabs 1 & 2) needs these files. See **app/data_setup.py** — "
            "it builds them from a CSV you supply. Saved-results mode (Tab 3) works now."
        )

    st.markdown("### Mode")
    live_possible = status["db_exists"] and status["questions_exists"] and bool(get_groq_key())
    st.write("🟢 Live mode available" if live_possible else "🟡 Saved-results mode only")

    n_saved = len(metrics_loader.discover_metrics_files())
    st.caption(f"{n_saved} saved benchmark result file(s) found in results/.")


st.title("AI Agents Inference Benchmarking System")
st.caption("A comparative evaluation of CrewAI and LangGraph for SQL data-analysis tasks.")

tab1, tab2, tab3 = st.tabs([
    "1 · Standardized Pipeline",
    "2 · Live CrewAI vs LangGraph",
    "3 · Metrics & Analysis",
])

# ==========================================================================
# TAB 1 — Objective 1: Standardized Benchmarking Pipeline
# ==========================================================================
with tab1:
    st.header("Standardized Benchmarking Pipeline")
    st.write(
        "Both frameworks are driven by the same `config.yaml`, the same SQLite "
        "database, the same three tools, and the same question set — run through "
        "two different agent orchestration layers."
    )

    st.subheader("Pipeline architecture")
    diagram_cols = st.columns(7)
    diagram_steps = ["Benchmark\nQuestions", "Shared\nConfiguration", "CrewAI /\nLangGraph",
                      "Shared SQL\nTools", "Same SQLite\nDatabase", "Metrics\nCollection", "Results &\nCharts"]
    for i, (col, step) in enumerate(zip(diagram_cols, diagram_steps)):
        with col:
            with st.container(border=True):
                st.markdown(f"**{step}**")
    st.caption("Benchmark Questions → Shared Configuration → CrewAI and LangGraph → Shared SQL Tools → Same SQLite Database → Metrics Collection → Results and Comparison Charts")

    config = config_validate.load_config()

    st.subheader("Configuration actually loaded from config.yaml")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("LLM model", config.get("model", {}).get("name", "—"))
        st.metric("Temperature", config.get("model", {}).get("temperature", "—"))
    with c2:
        st.metric("Iterations", config.get("benchmarks", {}).get("iterations", "—"))
        st.metric("Questions per iteration", config.get("benchmarks", {}).get("total_questions", "—"))
    with c3:
        st.metric("Max tokens", config.get("model", {}).get("max_tokens", "—"))
        st.metric("Results directory", config.get("results", {}).get("directory", "—"))

    with st.expander("Full config.yaml"):
        st.code(Path(config_validate.CONFIG_PATH).read_text(), language="yaml")

    st.markdown(f"**Database path (both scripts):** `{DB_PATH}` — "
                + ("exists ✅" if DB_PATH.exists() else "**missing** ❌"))

    prompts = config_validate.get_prompt_texts()
    with st.expander("Shared system prompt / instructions (as actually defined in each script)"):
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown("**CrewAI — Agent `goal`**")
            st.text_area("crewai_goal", prompts["crewai_goal"], height=220, label_visibility="collapsed")
        with pcol2:
            st.markdown("**LangGraph — `query_gen_system`**")
            st.text_area("langgraph_prompt", prompts["langgraph_system_prompt"], height=220, label_visibility="collapsed")

    st.subheader("What's shared vs. what differs by design")
    compare_df = pd.DataFrame([
        {"Aspect": "LLM model & temperature", "Shared?": "Shared", "Detail": "Both use config['model'] via the same Groq model string and temperature."},
        {"Aspect": "SQLite database", "Shared?": "Shared", "Detail": "Both connect to the same data/amazon_cleaned.db path."},
        {"Aspect": "Tools", "Shared?": "Shared", "Detail": "list_tables_tool, get_schema_tool, execute_sql_tool — same LangChain SQL utilities underneath."},
        {"Aspect": "Question set & iteration count", "Shared?": "Shared", "Detail": "Both loop over the same data/questions.py and config['benchmarks']['iterations']."},
        {"Aspect": "Prompt wording", "Shared?": "Framework-specific", "Detail": "CrewAI expresses instructions as an Agent `goal`; LangGraph as a system prompt. Same rules, different exact text."},
        {"Aspect": "Orchestration", "Shared?": "Framework-specific", "Detail": "CrewAI: role-based Crew/Task/Process.sequential. LangGraph: explicit StateGraph with assistant/tools nodes."},
        {"Aspect": "Retry / error handling", "Shared?": "Framework-specific", "Detail": "LangGraph wraps tool errors explicitly (handle_tool_error); CrewAI relies on its own internal agent loop."},
    ])
    st.dataframe(compare_df, hide_index=True, use_container_width=True)

    st.subheader("Configuration validation")
    checks = config_validate.run_validation(config)
    overall = config_validate.overall_status(checks)
    banner = {"pass": st.success, "warning": st.warning, "fail": st.error}[overall]
    banner({
        "pass": "All checks passed — the pipeline is standardized.",
        "warning": "Some aspects could not be fully verified as equivalent — see warnings below. The pipeline is standardized in its core configuration, but not claimed to be identical in every respect.",
        "fail": "Standardization could not be established.",
    }[overall])

    icon = {"pass": "✅", "warning": "⚠️", "fail": "❌"}
    for check in checks:
        st.markdown(f"{icon[check.status]} **{check.name}**")
        st.caption(check.detail)

    st.subheader("Run the real benchmark")
    status = ds_status()
    groq_key = get_groq_key()

    if "benchmark_running" not in st.session_state:
        st.session_state.benchmark_running = False

    run_col1, run_col2 = st.columns([2, 1])
    with run_col1:
        framework_choice = st.radio("Framework(s) to run", ["CrewAI", "LangGraph", "Both"], horizontal=True)
    with run_col2:
        st.caption(f"{config.get('benchmarks', {}).get('iterations', '?')} iteration(s) × "
                   f"{config.get('benchmarks', {}).get('total_questions', '?')} questions")

    can_run = status["db_exists"] and status["questions_exists"] and bool(groq_key)
    if not can_run:
        reasons = []
        if not status["db_exists"]:
            reasons.append("`data/amazon_cleaned.db` is missing")
        if not status["questions_exists"]:
            reasons.append("`data/questions.py` is missing")
        if not groq_key:
            reasons.append("no GROQ_API_KEY is set")
        st.error("Cannot run live benchmark: " + "; ".join(reasons) + ". See the sidebar for setup steps.")

    start_disabled = (not can_run) or st.session_state.benchmark_running
    if st.button("▶ Start Benchmark", disabled=start_disabled, type="primary"):
        st.session_state.benchmark_running = True
        frameworks_to_run = ["CrewAI", "LangGraph"] if framework_choice == "Both" else [framework_choice]
        for fw in frameworks_to_run:
            with st.status(f"Running real {fw} benchmark ({config['benchmarks']['iterations']} iteration(s))...", expanded=True) as s:
                outcome = runner.run_full_benchmark(fw, groq_key)
                if outcome["status"] == "success":
                    s.update(label=f"{fw} benchmark completed in {outcome['elapsed_s']:.1f}s", state="complete")
                else:
                    s.update(label=f"{fw} benchmark failed: {outcome.get('error')}", state="error")
                if outcome.get("log"):
                    st.code("\n".join(outcome["log"][-50:]))
        st.session_state.benchmark_running = False
        st.success("Done. Open Tab 3 to see the new results alongside past runs.")

    if st.session_state.benchmark_running:
        st.info("A benchmark run is in progress — please wait for it to finish.")

# ==========================================================================
# TAB 2 — Objective 2: Live CrewAI vs LangGraph comparison
# ==========================================================================
with tab2:
    st.header("Live CrewAI vs LangGraph Comparison")

    status = ds_status()
    groq_key = get_groq_key()

    if status["questions_exists"]:
        import importlib.util
        spec = importlib.util.spec_from_file_location("questions_only", QUESTIONS_PATH)
        qmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qmod)
        question_options = [v["question"] for v in qmod.queries.values()]
        source_note = "Loaded from the real data/questions.py."
    else:
        question_options = known_questions.RECOVERED_QUESTIONS
        source_note = known_questions.SOURCE_NOTE

    st.caption(source_note)
    selected_question = st.selectbox("Choose a benchmark question", question_options)
    custom_question = st.text_input("...or enter your own SQL-analysis question (optional)")
    question_to_run = custom_question.strip() if custom_question.strip() else selected_question

    expected_answer = None
    if EXPECTED_RESULTS_PATH.exists() and not custom_question.strip():
        try:
            exp_df = pd.read_csv(EXPECTED_RESULTS_PATH)
            match = exp_df[exp_df["question"] == selected_question]
            if not match.empty:
                expected_answer = match.iloc[0]["answer"]
        except Exception:
            expected_answer = None

    can_run = status["db_exists"] and status["questions_exists"] and bool(groq_key)
    if not can_run:
        reasons = []
        if not status["db_exists"]:
            reasons.append("`data/amazon_cleaned.db` is missing")
        if not status["questions_exists"]:
            reasons.append("`data/questions.py` is missing")
        if not groq_key:
            reasons.append("no GROQ_API_KEY is set")
        st.error("Cannot run live comparison: " + "; ".join(reasons) + ".")

    if "tab2_running" not in st.session_state:
        st.session_state.tab2_running = False
    if "tab2_results" not in st.session_state:
        st.session_state.tab2_results = {}

    if st.button("▶ Run CrewAI vs LangGraph on this question", disabled=(not can_run) or st.session_state.tab2_running, type="primary"):
        st.session_state.tab2_running = True
        st.session_state.tab2_results = {"question": question_to_run}

        progress = st.progress(0, text="Running CrewAI...")
        try:
            st.session_state.tab2_results["CrewAI"] = runner.run_single_question_crewai(question_to_run, groq_key)
        except Exception as e:
            st.session_state.tab2_results["CrewAI"] = {"status": "error", "error": str(e)}
        progress.progress(50, text="Running LangGraph...")
        try:
            st.session_state.tab2_results["LangGraph"] = runner.run_single_question_langgraph(question_to_run, groq_key)
        except Exception as e:
            st.session_state.tab2_results["LangGraph"] = {"status": "error", "error": str(e)}
        progress.progress(100, text="Done")
        st.session_state.tab2_running = False

    results = st.session_state.get("tab2_results", {})
    if results.get("question"):
        st.markdown(f"**Question:** {results['question']}")
        if expected_answer is not None:
            st.caption(f"Expected answer (from data/expected_results.csv): {expected_answer}")

        col_a, col_b = st.columns(2)
        for col, fw in [(col_a, "CrewAI"), (col_b, "LangGraph")]:
            with col:
                st.subheader(fw)
                r = results.get(fw)
                if not r:
                    st.info("Not run yet.")
                    continue
                if r["status"] == "error":
                    st.error(f"Execution failed: {r.get('error')}")
                    continue
                st.metric("Elapsed time", f"{r['elapsed_s']:.2f}s")
                st.metric("LLM calls", r.get("llm_call_count", "Unavailable"))
                st.caption(r.get("llm_call_count_basis", ""))
                st.markdown("**Answer:**")
                st.write(r.get("raw_answer", "")[:1500])
                st.markdown(f"**Extracted numerical answer:** `{r.get('numerical_answer')}`")

                sql_queries = r.get("sql_queries") or []
                if sql_queries:
                    st.markdown("**Generated SQL (captured from execution trace):**")
                    for q in sql_queries:
                        st.code(q, language="sql")
                else:
                    st.caption("No SQL captured in the execution trace for this run.")

                if expected_answer is not None and r.get("numerical_answer") not in (None, "Error retrieving numeric answer"):
                    try:
                        correct = float(r["numerical_answer"]) == float(expected_answer)
                        st.markdown(("✅ **Correct**" if correct else "❌ **Incorrect**") + " vs. expected answer")
                    except (ValueError, TypeError):
                        st.caption("Not verifiable — expected answer isn't numeric.")
                else:
                    st.caption("Not verifiable — no expected answer available for this question.")

# ==========================================================================
# TAB 3 — Objective 3: Performance Metrics and Analysis
# ==========================================================================
with tab3:
    st.header("Performance Metrics and Analysis")
    st.caption(f"Loaded from every metrics_*.json in {RESULTS_DIR} — {len(metrics_loader.discover_metrics_files())} file(s) found.")

    df = metrics_loader.load_all_runs()
    if df.empty:
        st.warning("No saved benchmark results found in results/.")
    else:
        frameworks_available = sorted(df["framework"].unique())
        chosen = st.multiselect("Framework filter", frameworks_available, default=frameworks_available)
        filtered = df[df["framework"].isin(chosen)] if chosen else df

        summary = metrics_loader.summarize(filtered)

        st.subheader("Summary")
        cols = st.columns(len(summary)) if len(summary) else [st]
        for col, (_, row) in zip(cols, summary.iterrows()):
            with col:
                st.markdown(f"**{row['framework']}**")
                st.metric("Avg accuracy", f"{row['avg_accuracy_pct']:.1f}%" if row["avg_accuracy_pct"] is not None else "Unavailable")
                st.metric("Avg latency", f"{row['avg_latency_s']:.2f}s" if row["avg_latency_s"] is not None else "Unavailable")
                st.metric("Median latency", f"{row['median_latency_s']:.2f}s" if row["median_latency_s"] is not None else "Unavailable")
                st.metric("Runs recorded", int(row["runs_recorded"]))

        st.markdown("##### Consistency (kept separate, not conflated)")
        cons_col1, cons_col2 = st.columns(2)
        with cons_col1:
            st.markdown("**Accuracy consistency across repeated runs** (σ of match_percentage)")
            for _, row in summary.iterrows():
                val = f"{row['accuracy_stdev_pct']:.2f} pts" if row["accuracy_stdev_pct"] is not None else "Unavailable (need ≥2 runs)"
                st.write(f"- {row['framework']}: {val}")
        with cons_col2:
            st.markdown("**Latency variability** (σ of iteration latency, seconds)")
            for _, row in summary.iterrows():
                val = f"{row['latency_stdev_s']:.2f}s" if row["latency_stdev_s"] is not None else "Unavailable (need ≥2 runs)"
                st.write(f"- {row['framework']}: {val}")

        st.caption(
            "⚠️ The saved field historically labeled 'api_calls' is a **tasks-completed-per-iteration** "
            "counter (MetricsCollector.increment_api_calls() is called once per question, not once per "
            "underlying LLM request) — it always equals the question count and is NOT a true LLM API "
            "request count. Shown below as 'tasks_completed_per_iteration' to avoid overstating precision. "
            "A true per-question LLM call count is only measurable for **new** live runs in Tab 2, "
            "which instrument it directly."
        )

        st.subheader("How accuracy is computed")
        st.markdown(
            "- Each run extracts the first number (`\\d+(\\.\\d+)?`) from the agent's text answer via "
            "`extract_numerical_value()` in `utils/helper_functions.py`.\n"
            "- `compare_answers()` checks this extracted value against `data/expected_results.csv` with "
            "**exact equality** (no tolerance) and reports the percentage of exact matches as `match_percentage`.\n"
            "- This is strict by design (inherited from the original project) — a correct answer formatted "
            "slightly differently than expected (e.g. rounding) would count as incorrect."
        )

        st.subheader("Raw results")
        st.dataframe(filtered, use_container_width=True)
        st.download_button(
            "Download filtered results as CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_benchmark_results.csv",
            mime="text/csv",
        )

        st.subheader("Interpretation")
        st.markdown(metrics_loader.interpretation_text(summary))
