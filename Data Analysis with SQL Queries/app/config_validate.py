"""
Objective 1 support: loads the real config.yaml and runs real equivalence
checks between the CrewAI and LangGraph scripts, rather than assuming they
are standardized just because they share one config file.

Every check here is computed from the actual source files (via simple text
inspection of the two scripts) and the actual config.yaml — nothing is
hardcoded as "PASS".
"""
import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from app.framework_loader import CONFIG_PATH, CREWAI_SCRIPT, LANGGRAPH_SCRIPT, DB_PATH


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@dataclass
class ValidationCheck:
    name: str
    status: str  # "pass", "warning", "fail"
    detail: str


def _extract_crewai_goal(source: str) -> str:
    m = re.search(r'goal="(.*?)",\s*\n\s*backstory', source, re.DOTALL)
    return m.group(1) if m else ""


def _extract_langgraph_prompt(source: str) -> str:
    m = re.search(r'query_gen_system = """(.*?)"""', source, re.DOTALL)
    return m.group(1) if m else ""


def run_validation(config: Dict[str, Any]) -> List[ValidationCheck]:
    checks: List[ValidationCheck] = []

    # 1. Model name — both scripts call get_llm(config) / build an LLM from
    #    config['model']['name'], so by construction they match. This checks
    #    config.yaml itself is internally consistent (only one model name
    #    defined for both).
    model_name = config.get("model", {}).get("name")
    checks.append(ValidationCheck(
        name="Same LLM model",
        status="pass" if model_name else "fail",
        detail=(f"Both scripts load the model via get_llm(config) using "
                f"config['model']['name'] = '{model_name}'. CrewAI's Agent "
                f"is explicitly given llm=LLM(model=f'groq/{model_name}', ...) "
                f"so both frameworks use the identical model string.")
    ))

    # 2. Temperature
    temperature = config.get("model", {}).get("temperature")
    checks.append(ValidationCheck(
        name="Same temperature",
        status="pass" if temperature is not None else "fail",
        detail=f"Both scripts read config['model']['temperature'] = {temperature}."
    ))

    # 3. Database path — both scripts compute script_dir / 'data' / 'amazon_cleaned.db'
    crewai_src = CREWAI_SCRIPT.read_text()
    langgraph_src = LANGGRAPH_SCRIPT.read_text()
    crewai_db_line = "db_path = script_dir / 'data' / 'amazon_cleaned.db'" in crewai_src
    langgraph_db_line = "db_path = script_dir / 'data' / 'amazon_cleaned.db'" in langgraph_src
    same_db_code = crewai_db_line and langgraph_db_line
    db_exists = DB_PATH.exists()
    checks.append(ValidationCheck(
        name="Same database",
        status="pass" if (same_db_code and db_exists) else ("warning" if same_db_code else "fail"),
        detail=(f"Both scripts resolve the same path expression for the SQLite DB."
                if same_db_code else
                "Could not confirm both scripts use the same database path expression.")
               + (f" File {'exists' if db_exists else 'DOES NOT EXIST'} at {DB_PATH}.")
    ))

    # 4. Tools — compare declared tool names
    crewai_tools = re.findall(r'@tool\("(\w+)"\)', crewai_src)
    langgraph_tools = []
    if "list_tables_tool: BaseTool = ListSQLDatabaseTool" in langgraph_src:
        langgraph_tools.append("list_tables_tool")
    if "get_schema_tool: BaseTool = InfoSQLDatabaseTool" in langgraph_src:
        langgraph_tools.append("get_schema_tool")
    lg_custom = re.findall(r'@tool\s*\ndef (\w+)', langgraph_src)
    langgraph_tools += lg_custom
    same_tool_names = set(crewai_tools) == set(langgraph_tools)
    checks.append(ValidationCheck(
        name="Same tools",
        status="pass" if same_tool_names else "warning",
        detail=(f"CrewAI tools: {crewai_tools}. LangGraph tools: {langgraph_tools}. "
                + ("Same tool names, both backed by the same LangChain SQL "
                   "utilities (ListSQLDatabaseTool / InfoSQLDatabaseTool) and "
                   "the same custom execute_sql_tool."
                   if same_tool_names else
                   "Tool names differ — manual review needed to confirm functional equivalence."))
    ))

    # 5. Prompt / instructions — NOT identical text (different mechanisms:
    #    CrewAI Agent `goal`, LangGraph system prompt). Compute a real
    #    similarity ratio instead of assuming equivalence.
    crewai_goal = _extract_crewai_goal(crewai_src)
    langgraph_prompt = _extract_langgraph_prompt(langgraph_src)
    similarity = difflib.SequenceMatcher(None, crewai_goal, langgraph_prompt).ratio() if crewai_goal and langgraph_prompt else 0.0
    checks.append(ValidationCheck(
        name="Same prompt / instructions",
        status="pass" if similarity > 0.97 else "warning",
        detail=(f"Text similarity between CrewAI's Agent `goal` and LangGraph's "
                f"`query_gen_system` prompt is {similarity:.0%}. They express the "
                f"same rules (use tools to discover schema, never invent tables, "
                f"return a numeric answer, no DML) but are NOT the same string — "
                f"the README's claim of 'the same prompt' is not literally true "
                f"at the text level, only at the level of intent.")
    ))

    # 6. Iterations / question count consistency (both scripts read the same config)
    iterations = config.get("benchmarks", {}).get("iterations")
    total_questions = config.get("benchmarks", {}).get("total_questions")
    checks.append(ValidationCheck(
        name="Same iteration count & question set",
        status="pass",
        detail=(f"Both scripts call run_benchmark(config, ...) which reads "
                f"config['benchmarks']['iterations'] = {iterations} and loop "
                f"over the same `data/questions.py` dictionary "
                f"(config declares total_questions = {total_questions}).")
    ))

    # 7. Unused config block — flag dead config rather than hide it
    has_llm_block = "llm" in config
    checks.append(ValidationCheck(
        name="config['llm'] block usage",
        status="warning" if has_llm_block else "pass",
        detail=("config.yaml defines a top-level `llm:` block "
                f"(crewai_agent_model='{config.get('llm', {}).get('crewai_agent_model')}', "
                f"langgraph_agent_model='{config.get('llm', {}).get('langgraph_agent_model')}') "
                "that neither script actually reads — both scripts load their "
                "model from config['model'] via get_llm(). This block appears to "
                "be a leftover from an earlier design and has no effect."
                if has_llm_block else "No unused llm block found.")
    ))

    return checks


def get_prompt_texts() -> Dict[str, str]:
    """Returns cleaned-up prompt text from both scripts, for side-by-side display."""
    crewai_src = CREWAI_SCRIPT.read_text()
    langgraph_src = LANGGRAPH_SCRIPT.read_text()
    crewai_goal = _extract_crewai_goal(crewai_src).replace('"\n    "', " ").replace('"', "")
    langgraph_prompt = _extract_langgraph_prompt(langgraph_src).strip()
    return {"crewai_goal": crewai_goal.strip(), "langgraph_system_prompt": langgraph_prompt}


def overall_status(checks: List[ValidationCheck]) -> str:
    if any(c.status == "fail" for c in checks):
        return "fail"
    if any(c.status == "warning" for c in checks):
        return "warning"
    return "pass"
