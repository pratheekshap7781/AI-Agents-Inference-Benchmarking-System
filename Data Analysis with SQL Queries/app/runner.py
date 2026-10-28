"""
Live execution wrappers around the REAL CrewAI and LangGraph scripts.

Nothing here reimplements the agents. For a single question, CrewAI reuses
the exact `senior_data_analyst` Agent and `write_sql_query` Task objects
defined in the original script; LangGraph reuses the exact compiled `graph`
object. A small, additive capture layer (a step_callback for CrewAI, and
reading the native LangGraph stream events) is used only to surface the
generated SQL and a real LLM-call count to the UI — it does not change what
either framework does.
"""
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from app.framework_loader import load_crewai_module, load_langgraph_module


class ListLogHandler(logging.Handler):
    """Captures log records into a list so the UI can show real run status."""

    def __init__(self):
        super().__init__()
        self.records: List[str] = []

    def emit(self, record):
        self.records.append(self.format(record))


def run_single_question_crewai(question: str, groq_api_key: Optional[str] = None) -> Dict[str, Any]:
    module = load_crewai_module(groq_api_key)
    from crewai import Crew, Process
    from crewai.events import crewai_event_bus
    from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent
    from crewai.events.types.llm_events import LLMCallCompletedEvent

    # crewai 1.15.x routes tool calls through its "native tool calling" path,
    # which does NOT invoke the classic step_callback — so SQL/tool capture
    # and LLM-call counting are done via crewai's event bus instead, which
    # fires regardless of which internal execution path crewai takes. The
    # event bus is a process-wide singleton, so handlers are registered and
    # torn down around this single kickoff() call to avoid leaking counts
    # into a later run in the same Streamlit session.
    trace = {"sql_queries": [], "tool_calls": [], "llm_calls": 0}

    def _on_tool(source, event):
        try:
            trace["tool_calls"].append({"tool": event.tool_name, "input": event.tool_args})
            if event.tool_name == "execute_sql_tool" and event.tool_args:
                args = event.tool_args
                query = args.get("query") if isinstance(args, dict) else None
                trace["sql_queries"].append(str(query) if query else str(args))
        except Exception:
            pass  # instrumentation must never break the real run

    def _on_llm(source, event):
        trace["llm_calls"] += 1

    crewai_event_bus.on(ToolUsageFinishedEvent)(_on_tool)
    crewai_event_bus.on(LLMCallCompletedEvent)(_on_llm)

    crew = Crew(
        agents=[module.senior_data_analyst],
        tasks=[module.write_sql_query],
        process=Process.sequential,
        verbose=False,
        memory=False,
        manager_llm=module.crew_llm,
        output_log_file="crew.log",
    )

    result: Dict[str, Any] = {"framework": "CrewAI", "question": question, "status": "running"}
    start = time.time()
    try:
        output = crew.kickoff(inputs={"question": question})
        elapsed = time.time() - start
        answer_text = str(output)
        numerical = module.extract_numerical_value(answer_text)
        result.update({
            "status": "success",
            "raw_answer": answer_text,
            "numerical_answer": numerical,
            "elapsed_s": elapsed,
            "sql_queries": trace["sql_queries"],
            "tool_calls": trace["tool_calls"],
            "llm_call_count": trace["llm_calls"],
            "llm_call_count_basis": (
                "Counted via crewai's own event bus (LLMCallCompletedEvent) — "
                "reflects how many LLM completions crewai itself reports for "
                "this run. May differ from a raw HTTP request count if crewai's "
                "native tool-calling path handles a multi-tool turn as one "
                "logical call internally."
            ),
        })
    except Exception as e:
        result.update({"status": "error", "error": str(e), "elapsed_s": time.time() - start})
    finally:
        crewai_event_bus.off(ToolUsageFinishedEvent, _on_tool)
        crewai_event_bus.off(LLMCallCompletedEvent, _on_llm)
    return result


def run_single_question_langgraph(question: str, groq_api_key: Optional[str] = None) -> Dict[str, Any]:
    module = load_langgraph_module(groq_api_key)

    result: Dict[str, Any] = {"framework": "LangGraph", "question": question, "status": "running"}
    start = time.time()
    events: List[dict] = []
    assistant_calls = 0
    sql_queries: List[str] = []
    try:
        for event in module.graph.stream({"messages": [("user", question)]}):
            events.append(event)
            if "assistant" in event:
                assistant_calls += 1
                msg = event["assistant"]["messages"]
                for tc in (getattr(msg, "tool_calls", None) or []):
                    if tc.get("name") == "execute_sql_tool":
                        q = (tc.get("args") or {}).get("query")
                        if q:
                            sql_queries.append(q)
        elapsed = time.time() - start

        final_content = ""
        if events and "assistant" in events[-1]:
            final_content = events[-1]["assistant"]["messages"].content

        numerical = module.extract_numerical_value(final_content) if final_content else "Error retrieving numeric answer"
        result.update({
            "status": "success",
            "raw_answer": final_content,
            "numerical_answer": numerical,
            "elapsed_s": elapsed,
            "sql_queries": sql_queries,
            "llm_call_count": assistant_calls,
            "llm_call_count_basis": (
                "Exact: every 'assistant' node firing in the compiled LangGraph "
                "state machine corresponds to exactly one LLM invocation."
            ),
        })
    except Exception as e:
        result.update({"status": "error", "error": str(e), "elapsed_s": time.time() - start})
    return result


def run_full_benchmark(framework: str, groq_api_key: Optional[str] = None) -> Dict[str, Any]:
    """Runs the REAL run_benchmark() from the chosen script for all configured
    iterations/questions, saving results via the existing MetricsCollector /
    save_results_to_csv / compare_answers conventions (same results/ dir,
    same filename patterns as a manual `python ... .py` run)."""
    if framework == "CrewAI":
        module = load_crewai_module(groq_api_key)
    elif framework == "LangGraph":
        module = load_langgraph_module(groq_api_key)
    else:
        raise ValueError(f"Unknown framework: {framework}")

    handler = ListLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger = module.setup_logging(
        log_file=module.config["logging"]["file"],
        log_level=module.config["logging"]["level"],
        log_format=module.config["logging"]["format"],
    )
    logger.addHandler(handler)
    start = time.time()
    try:
        module.run_benchmark(module.config, module.llm, logger)
        return {"status": "success", "elapsed_s": time.time() - start, "log": handler.records}
    except Exception as e:
        return {"status": "error", "error": str(e), "elapsed_s": time.time() - start, "log": handler.records}
    finally:
        logger.removeHandler(handler)
