# src/tracing/langsmith_tools.py
import os
import functools
import sys
from typing import Optional, Dict, Any, List, Union

from langsmith import Client, traceable, RunTree
from langsmith.schemas import Run

# Import the new client class
from .llm_client import create_llm_client, log_feedback, get_run_history, TracedLLMClient


# For backward compatibility
def get_traced_llm_client(provider="openai", api_key=None):
    """
    Get a traced LLM client based on provider.

    Args:
        provider (str): The LLM provider ("openai" or "gemini")
        api_key (str, optional): API key for the provider

    Returns:
        The traced LLM client
    """
    return create_llm_client(provider=provider, api_key=api_key, enable_tracing=True)


@traceable(name="generate_completion")
def traced_generate_completion(
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
) -> str:
    """
    Generate a completion using the selected LLM with LangSmith tracing.
    This mirrors the interface of the generate_completion function in llm_utils.py.

    Args:
        prompt (str): The user prompt
        system_prompt (str, optional): The system prompt
        max_tokens (int, optional): Maximum tokens to generate
        temperature (float, optional): Temperature for generation
        model (str, optional): Override the model selection
        metadata (dict, optional): Additional metadata for LangSmith

    Returns:
        str: Generated text
    """
    # Create a traced client
    client = create_llm_client(model=model, enable_tracing=True)

    try:
        return client.generate_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
            metadata=metadata,
            **kwargs
        )
    except Exception as e:
        return f"Error generating completion: {str(e)}"


@traceable
def traced_pipeline(
        user_input: str,
        system_prompt: Optional[str] = None,
        model: str = "gpt-4o",
        metadata: Optional[Dict[str, Any]] = None,  # Changed parameter name from _trace_metadata
        **kwargs
):
    """
    A traced pipeline function that processes user input through an LLM.

    Args:
        user_input (str): The user's input text
        system_prompt (str, optional): The system prompt
        model (str): The model to use
        metadata (dict, optional): Additional trace metadata

    Returns:
        str: The generated response
    """
    # Handle None metadata
    trace_metadata = metadata or {}

    # Remove any kwargs we don't want to pass to generate_completion
    kwargs_for_completion = {k: v for k, v in kwargs.items()
                             if k not in ['_trace_name', '_trace_metadata']}

    return traced_generate_completion(
        prompt=user_input,
        system_prompt=system_prompt,
        model=model,
        metadata=trace_metadata,
        **kwargs_for_completion
    )


# Example utilities for easier integration
def with_tracing(func):
    """
    Decorator to add LangSmith tracing to any function.

    Example:
        @with_tracing
        def my_function(arg1, arg2):
            ...
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # Extract a name from kwargs or use function name
        name = kwargs.pop("_trace_name", func.__name__)

        # Extract metadata from kwargs or create empty dict
        metadata = kwargs.pop("_trace_metadata", {})

        # Determine the project name
        project_name = os.environ.get("LANGSMITH_PROJECT_NAME", None)
        if "pytest" in sys.argv[0]:  # Check if pytest is in sys.argv[0]
            project_name = "lightbox-test"

        trace_func = traceable(name=name, project_name=project_name)(func)
        return trace_func(*args, **kwargs, metadata=metadata)

    return wrapped