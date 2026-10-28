"""
Loads the two original benchmark scripts as Python modules.

Both scripts are named with spaces and commas
("Data Analysis with SQL Queries, CrewAI.py") so they cannot be imported
with a normal `import` statement. This uses importlib to load them by file
path instead, with zero changes to the files themselves beyond the two bug
fixes documented in those files (removed hardcoded API key overwrite, wired
the Groq LLM into the CrewAI Agent). Everything the app runs — agents,
tasks, tools, the compiled graph, kickoff_crew, build_graph, run_benchmark —
is the exact object defined in those files.
"""
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional

APP_DIR = Path(__file__).parent
PROJECT_DIR = APP_DIR.parent  # "Data Analysis with SQL Queries/"

CREWAI_SCRIPT = PROJECT_DIR / "Data Analysis with SQL Queries, CrewAI.py"
LANGGRAPH_SCRIPT = PROJECT_DIR / "Data Analysis with SQL Queries, LangGraph.py"
DB_PATH = PROJECT_DIR / "data" / "amazon_cleaned.db"
QUESTIONS_PATH = PROJECT_DIR / "data" / "questions.py"
EXPECTED_RESULTS_PATH = PROJECT_DIR / "data" / "expected_results.csv"
CONFIG_PATH = PROJECT_DIR / "config.yaml"
RESULTS_DIR = PROJECT_DIR / "results"


class DatasetMissingError(RuntimeError):
    """Raised when data/amazon_cleaned.db or data/questions.py are absent."""


def dataset_status() -> dict:
    return {
        "db_exists": DB_PATH.exists(),
        "db_path": str(DB_PATH),
        "questions_exists": QUESTIONS_PATH.exists(),
        "questions_path": str(QUESTIONS_PATH),
        "expected_results_exists": EXPECTED_RESULTS_PATH.exists(),
        "expected_results_path": str(EXPECTED_RESULTS_PATH),
    }


def require_dataset() -> None:
    status = dataset_status()
    missing = [v for k, v in [("db_exists", "data/amazon_cleaned.db"),
                               ("questions_exists", "data/questions.py")]
               if not status[k]]
    if missing:
        raise DatasetMissingError(
            "Missing required file(s): " + ", ".join(missing) +
            ". Run `python app/data_setup.py` after placing the Amazon "
            "Sales Dataset CSV in data/raw/ (see the app's setup notice)."
        )


def _load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")

    # Make `from data.questions import queries` and `from utils...` resolve
    # the same way they do when the script is run directly from its own
    # directory.
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_crewai_module = None
_langgraph_module = None


def _patch_crewai_groq_cache_bug() -> None:
    """Workaround for a crewai + litellm + Groq compatibility bug (as of
    crewai 1.15.32 / litellm 1.106.5): crewai marks every message with a
    'cache_breakpoint' hint, expecting the provider adapter to translate or
    strip it (see crewai/llms/cache.py). litellm's Groq adapter does
    neither, so the raw flag leaks into the request body and Groq's API
    rejects it outright ("property 'cache_breakpoint' is unsupported").
    This neutralizes the marker — it does not change any prompt, tool, or
    agent behavior, only removes a provider hint Groq can't accept."""
    try:
        import crewai.llms.cache as _cache_mod
        _cache_mod.mark_cache_breakpoint = lambda message: dict(message)
    except Exception:
        pass  # if crewai's internals change, fail open rather than block the app


def load_crewai_module(groq_api_key: Optional[str] = None):
    """Import the real CrewAI script. Caches the module after first load."""
    global _crewai_module
    require_dataset()
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not set.")
    _patch_crewai_groq_cache_bug()
    if _crewai_module is None:
        _crewai_module = _load_module(CREWAI_SCRIPT, "crewai_benchmark_script")
    return _crewai_module


def load_langgraph_module(groq_api_key: Optional[str] = None):
    """Import the real LangGraph script. Caches the module after first load."""
    global _langgraph_module
    require_dataset()
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is not set.")
    if _langgraph_module is None:
        _langgraph_module = _load_module(LANGGRAPH_SCRIPT, "langgraph_benchmark_script")
    return _langgraph_module
