# Import libraries
import os
import yaml
import time
import getpass
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing_extensions import Dict, Any, List, Optional, TypedDict, Annotated
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, START
from data.questions import queries  # Import the dictionary of queries containing the questions and expected results
from utils.metrics_collector import MetricsCollector
from utils.logging_config import setup_logging
from utils.helper_functions import get_llm, extract_questions, extract_numerical_value, save_results_to_csv, compare_answers
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks, Runnable
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import BaseTool, tool
from langchain_community.tools.sql_database.tool import ListSQLDatabaseTool, InfoSQLDatabaseTool
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import AnyMessage, add_messages

# Set the framework that will be benchmarked
FRAMEWORK = "LangGraph"

# Load environment variables
# Returns "True" if successful
load_dotenv()

# Load config.yaml file
script_dir = Path(__file__).parent
config_path = script_dir / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Get API key for GROQ
# (FIX: this used to be overwritten with a hardcoded placeholder, which
# silently broke every run regardless of the real key in .env/env vars)
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

# Load the LLM
llm = get_llm(config)

# Connect to database
db_path = script_dir / 'data' / 'amazon_cleaned.db'
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# Extract the questions from the queries dictionary
questions = extract_questions(queries)

# Define the state for the graph
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def handle_tool_error(state) -> dict:
    """
    Handles errors that occur during the execution of a tool. It surfaces the error to the agent
    by generating a message with the error details.

    Args:
        state: The current state of the system, which contains error information.

    Returns:
        dict: A dictionary containing error messages to be returned to the agent.
    """
    # Retrieve the error from the state
    error = state.get("error")

    # Retrieve the tool calls made in the last message in the state
    tool_calls = state["messages"][-1].tool_calls

    # Return a dictionary with error messages formatted for each tool call
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],  # Include the tool call ID for reference
            )
            for tc in tool_calls  # Create an error message for each tool call
        ]
    }

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Creates a ToolNode with a fallback mechanism to handle errors and surface them to the agent.

    Args:
        tools (list): A list of tools that the ToolNode will interact with.

    Returns:
        RunnableWithFallbacks: A ToolNode with a fallback that handles errors and returns an appropriate response.
    """
    # Create a ToolNode using the provided list of tools
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

# Define tools that the agent will use to interact with the database
# Tool to fetch the available tables from the database
list_tables_tool: BaseTool = ListSQLDatabaseTool(db=db)

# Tool to fetch the DDL for a table
get_schema_tool: BaseTool = InfoSQLDatabaseTool(db=db)

@tool
def execute_sql_tool(query: str) -> str:
    """
    Executes a SQL query against the database and returns the result. If the query fails,
    it returns an error message prompting the user to rewrite the query and try again.

    Args:
        query (str): The SQL query to be executed against the database.

    Returns:
        str: The result of the query if successful, or an error message if the query fails.
    """
    # Run the query against the database without throwing errors (i.e., returns None or failure without raising an exception)
    result = db.run_no_throw(query)

    # If the result is empty or None, it means the query failed, so return an error message
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."

    # Return the result of the successful query execution
    return result

# Define the tools the assistant will use
all_tools = [list_tables_tool, get_schema_tool, execute_sql_tool]

class Assistant:
    """
    The Assistant class is responsible for running the AI agent and interacting with the tools.
    It invokes the tools, ensures they return appropriate results, and handles any 
    re-prompts or errors that may occur during execution.
    Its core functionality includes invoking the Runnable, which defines the process 
    of calling the LLM and tools, and then monitoring the results.
    """
    def __init__(self, runnable: Runnable):
        # Initialize with the runnable that defines the process for interacting with the tools
        self.runnable = runnable

    def __call__(self, state: State):
        while True:
            # Invoke the runnable with the current state (messages and context)
            result = self.runnable.invoke(state)
            
            # If the tool fails to return valid output, re-prompt the user to clarify or retry
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                # Add a message to request a valid response
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                # Break the loop when valid output is obtained
                break

        # Return the final state after processing the runnable
        return {"messages": result}
    
# Add a node for a model to generate a query based on the question and schema
query_gen_system = """You are a SQL expert with a strong attention to detail.

Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

CRITICAL WORKFLOW - Follow these steps SEQUENTIALLY:
1. FIRST: Call sql_db_list_tables to get the actual table names in the database. DO NOT assume table names.
2. SECOND: Wait for the result, then call sql_db_schema with the EXACT table names returned from step 1.
3. THIRD: Wait for the schema result, then use that information to write an SQL query.
4. FOURTH: Execute the SQL query using execute_sql_tool.
5. FINALLY: Return the answer as a simple numerical value.

IMPORTANT RULES:
- NEVER assume table or column names. ALWAYS use the tools to discover them.
- NEVER call sql_db_schema with table names you haven't verified exist in the database.
- NEVER make multiple tool calls at once - wait for results before proceeding to the next step.
- If you get an error, check what table names actually exist using sql_db_list_tables first.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
- Never query for all columns, only ask for relevant columns given the question.
- If you get an error while executing a query, rewrite the query using the correct table/column names from the schema.
- If you get an empty result set, try to rewrite the query to get a non-empty result set.
- NEVER make stuff up if you don't have enough information - just say you don't have enough information."""
query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}")]
)
query_gen = query_gen_prompt | ChatGroq(model=config['model']['name'], temperature=config['model']['temperature']).bind_tools(all_tools)

def query_gen_node(state: State):
    """
    Generates a query based on the given state and checks for tool call errors.

    Args:
        state (State): The current state of the graph that will be used to generate a query.

    Returns:
        dict: A dictionary containing the generated message and any tool call error messages (if applicable).
    """
    # Generate a message using the query generator, which involves calling an LLM to create a query based on the current state.
    message = query_gen.invoke(state)

    # Initialize an empty list to store any error messages related to tool calls
    tool_messages = []

    # Check if there are any tool calls generated by the message
    if message.tool_calls:
        # Iterate through the tool calls to check if they are correct
        for tc in message.tool_calls:
            # If the tool called is not 'SubmitFinalAnswer', it's considered an error
            if tc["name"] != "SubmitFinalAnswer":
                # Add an error message to the tool_messages list indicating the wrong tool was called
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        # If no tool calls are made, leave the tool_messages list empty
        tool_messages = []

    # Return a dictionary containing the generated message and any error tool messages
    # The 'messages' key will hold both the original message and any error messages from wrong tool calls
    return {"messages": [message] + tool_messages}

# Bind the tools to the assistant's workflow
assistant_runnable = query_gen_prompt | llm.bind_tools(all_tools)

# Build the graph by adding the nodes and edges
builder = StateGraph(State)
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(all_tools))

builder.add_edge(START, "assistant")  # Start with the assistant
builder.add_conditional_edges("assistant", tools_condition)  # Move to tools after input
builder.add_edge("tools", "assistant")  # Return to assistant after tool execution

# Compile the graph
graph = builder.compile()

# Create initial state for the workflow
def create_initial_state() -> State:
    """Create initial state for the workflow.
    
    Returns:
        State: Initial workflow state
    """
    return State(
        messages=[]
    )

# Build the graph
def build_graph(
    logger: logging.Logger,
    llm: Optional[ChatGroq] = None,
    metrics: Optional[MetricsCollector] = None
) -> StateGraph:
    
    """Build the complete agent workflow graph.
    Args:
        logger: Logger instance
        llm: Optional ChatGroq instance (will be created if not provided)
        
    Returns:
        StateGraph: Configured workflow graph
    """
    
    # Create LLM if not provided
    if llm is None:
        raise ValueError("LLM is not provided")

    # Create initial state
    initial_state = create_initial_state()

    # Save the results of each query in a list that will later be saved in a CSV file
    results = []

    # Orchestrate with LangGraph
    graph = builder.compile()

    # Loop over the queries in the dictionary with questions
    for key, value in questions.items():
        start_time = time.time()
   
        for event in graph.stream({"messages": [("user", value)]}):
            print(event)
      
        # Get the content from the final answer
        content = event['assistant']['messages'].content

        # Extract the numerical value from the answer
        numerical_answer = extract_numerical_value(content)

        # Append the results to the list
        results.append(
                        (
                            value,  # The question value
                            numerical_answer # The extracted numerical answer
                        )
                    )
        api_latency = time.time() - start_time
        metrics.increment_api_calls()
        metrics.add_api_latency(api_latency)
                    
    # After looping, print the results to verify
    logger.debug("Results generated")

    # Create final state
    final_state = StateGraph(State)

    return results, initial_state, final_state

# Run the benchmark process with iterations
def run_benchmark(config: Dict[str, Any], llm: ChatGroq, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Run the benchmark process with iterations"""
    logger.debug("Starting benchmark run...")

    # Get number of iterations from config
    num_iterations = config['benchmarks']['iterations']
    logger.info(f"Running benchmark for {num_iterations} iterations")
    
        # Set up path for logging and metrics    
    # Get the current directory where the script is running
    current_dir = Path(__file__).parent

    # Define the 'results' directory in the same folder as the script
    log_dir = current_dir / 'results'

    # Create the directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to: {log_dir.resolve()}")

    # Initialize MetricsCollector with the dynamic log_dir
    metrics = MetricsCollector(log_dir)

    # Run iterations
    for iteration in range(num_iterations):
        logger.info(f"Starting iteration {iteration + 1}/{num_iterations}")

        metrics.start_iteration(FRAMEWORK, iteration)

        results, initial_state, final_state = build_graph(logger, llm, metrics)
        
        save_results_to_csv(results, "results", iteration, log_dir, FRAMEWORK)

        logger.info(f"Completed iteration {iteration + 1}")

        metrics.end_iteration()
        metrics.save_iteration(FRAMEWORK)
    
    metrics.save_metrics()
    metrics.generate_plots()

    # Generate metric for calculating the percentage of correct answers
    expected_file = current_dir / 'data' / 'expected_results.csv'
    file_name = 'results'
    compare_answers(log_dir, str(expected_file), file_name, num_iterations, FRAMEWORK)
    
    logger.info("Benchmark completed successfully")

# Main execution function
def main():
    """Main entry point for the application."""
    
    # Setup logging
    logger = setup_logging(
        log_file=config['logging']['file'],
        log_level=config['logging']['level'],
        log_format=config['logging']['format']
    )
    try:        
        # Run workflow
        run_benchmark(config, llm, logger)  
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate_limit" in error_msg.lower() or "RateLimitError" in error_msg:
            logger.error(f"Rate limit error: {error_msg}")
            logger.error("Please wait for the rate limit to reset (typically 10+ minutes) or upgrade your Groq API tier.")
        else:
            logger.error(f"Error in workflow: {error_msg}")
        raise

if __name__ == "__main__":
    main()