import streamlit as st
import pandas as pd
import json

from agent import query_agent, create_agent


def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """

    # print("res ---->", response)
    # answer = json.loads(response)
    answer = response
    # print("res -->", response)
    return answer


def write_response(response_dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    st.write(response_dict)


st.title("MoneyFlo GPT")

# st.write("Please upload your CSV file below.")

data = st.file_uploader("Upload a CSV")

query = st.text_area("Insert your query")

if st.button("Submit Query", type="primary"):
    # Create an agent from the CSV file.
    agent = create_agent(data)

    # Query the agent.
    response = query_agent(agent=agent, query=query)

    # Decode the response.
    decoded_response = decode_response(response)

    # Write the response to the Streamlit app.
    write_response(decoded_response)
