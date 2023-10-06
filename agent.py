from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain.agents import AgentType


# Setting up the api key
# import environ

# env = environ.Env()
# environ.Env.read_env()
# API_KEY = env("apikey")

API_KEY = st.secrets["apikey"]


def create_agent(filename: str):
    """
    Create an agent that can access and use a large language model (LLM).

    Args:
        filename: The path to the CSV file that contains the data.

    Returns:
        An agent that can access and use the LLM.
    """

    # Create an OpenAI object.
    # llm = OpenAI(openai_api_key=API_KEY, model_name="gpt-4", temperature=0)
    llm = ChatOpenAI(
        openai_api_key=API_KEY,
        temperature=0,
        model_name="gpt-4",
    )

    # Read the CSV file into a Pandas DataFrame.
    df = pd.read_csv(filename)

    # Create a Pandas DataFrame agent.
    return create_pandas_dataframe_agent(
        llm,
        df,
        # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        # max_execution_time=1,
        early_stopping_method="generate",
        verbose=False,
    )


def query_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """

    prompt = (
        """
            Query:
        """
        + query
    )

    # Run the prompt through the agent.
    response = agent.run(prompt)

    # Convert the response to a string.
    return response.__str__()
