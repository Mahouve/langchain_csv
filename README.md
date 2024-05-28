# Chat with your CSV: Visualize Your Data with Langchain and Streamlit

Repository for the application built in [this](https://dev.to/ngonidzashe/chat-with-your-csv-visualize-your-data-with-langchain-and-streamlit-ej7) article.

## Requirements

Install the required packages by running

```
pip install -r requirements.txt
```

## Interface
![interace](https://github.com/Ngonie-x/langchain_csv/assets/28601809/0f27a2da-1128-4b23-9d01-b509b55761eb)
---
![prompt](https://github.com/Ngonie-x/langchain_csv/assets/28601809/9e90ba35-c45e-4ea4-b632-2c9203b373d2)
---
![bar graph](https://github.com/Ngonie-x/langchain_csv/assets/28601809/2fb4f9fe-cd6e-46ed-afad-66e8606fca3c)
---
![create a table](https://github.com/Ngonie-x/langchain_csv/assets/28601809/b49c50d2-c12e-43a0-a593-33e508dbf4a6)
---
![line chart](https://github.com/Ngonie-x/langchain_csv/assets/28601809/f4e94c50-e505-4f32-a4e4-f0ede5158b3b)

from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd

# Setting up the api key
import environ

env = environ.Env()
environ.Env.read_env()

API_KEY = env("apikey")


def create_agent(filename: str):
    """
    Create an agent that can access and use a large language model (LLM).

    Args:
        filename: The path to the CSV file that contains the data.

    Returns:
        An agent that can access and use the LLM.
    """

    # Create an OpenAI object.
    llm = OpenAI(openai_api_key=API_KEY)

    # Read the CSV file into a Pandas DataFrame.
    df = pd.read_csv(filename)

    # Create a Pandas DataFrame agent.
    return create_pandas_dataframe_agent(llm, df, verbose=False)


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
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
            
            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
            
            There can only be two types of chart, "bar" and "line".
            
            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}
            
            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}
            
            Return all output as a string.
            
            All strings in "columns" list and data list, should be in double quotes,
            
            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}
            
            Lets think step by step.
            
            Below is the query.
            Query: 
            """
        + query
    )

    # Run the prompt through the agent.
    response = agent.run(prompt)

    # Convert the response to a string.
    return response.__str__()











#interface.py

    import streamlit as st
import pandas as pd
import json
import plotly.express as px
from agent import query_agent, create_agent

def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    return json.loads(response)

def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.bar_chart(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.line_chart(df)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

def display_dashboard(df):
    """Function to display a dashboard with various visualizations."""
    st.header("Dataset Overview")
    st.write(df)

    st.header("Basic Statistics")
    st.write(df.describe())

    st.header("Visualizations")

    # Plotly bar chart
    st.subheader("Bar Chart")
    column = st.selectbox("Select column for bar chart:", df.columns)
    bar_fig = px.bar(df, x=df.index, y=column)
    st.plotly_chart(bar_fig)

    # Plotly line chart
    st.subheader("Line Chart")
    column = st.selectbox("Select column for line chart:", df.columns, key='line_chart')
    line_fig = px.line(df, x=df.index, y=column)
    st.plotly_chart(line_fig)

    # Plotly scatter plot
    st.subheader("Scatter Plot")
    x_axis = st.selectbox("Select X-axis for scatter plot:", df.columns)
    y_axis = st.selectbox("Select Y-axis for scatter plot:", df.columns)
    scatter_fig = px.scatter(df, x=x_axis, y=y_axis)
    st.plotly_chart(scatter_fig)

st.title("👨‍💻 Chat with your CSV or Excel file")

st.write("Please upload your CSV, XLSX, or XLS file below.")

data = st.file_uploader("Upload a CSV, XLSX, or XLS file", type=["csv", "xlsx", "xls"])

if data is not None:
    try:
        # Determine the file type and read the file accordingly
        if data.name.endswith('.csv'):
            # Try reading with different encodings and handle bad lines
            df = pd.read_csv(data, encoding='latin1', error_bad_lines=False)
        elif data.name.endswith('.xlsx'):
            df = pd.read_excel(data, engine='openpyxl')
        elif data.name.endswith('.xls'):
            df = pd.read_excel(data, engine='xlrd')

        display_dashboard(df)

        query = st.text_area("Insert your query")

        if st.button("Submit Query"):
            # Create an agent from the data frame
            agent = create_agent(df)

            # Query the agent
            response = query_agent(agent=agent, query=query)

            # Decode the response
            decoded_response = decode_response(response)

            # Write the response to the Streamlit app
            write_response(decoded_response)
    except Exception as e:
        st.error(f"An error occurred: {e}")









#agent.py


import openai
import os
import dotenv

# Charger les variables d'environnement
dotenv.load_dotenv()

# Récupérer la clé API depuis les variables d'environnement
API_KEY = os.getenv("apikey")
openai.api_key = API_KEY

def create_agent(dataframe):
    """
    Crée un agent qui peut accéder et utiliser un modèle de langage (LLM).

    Args:
        dataframe: Le DataFrame Pandas qui contient les données.

    Returns:
        Un agent qui peut accéder et utiliser le LLM.
    """
    llm = OpenAI(api_key=API_KEY, engine="text-davinci-003")
    return create_pandas_dataframe_agent(llm, dataframe)

def query_agent(agent, query):
    """
    Interroge un agent et retourne la réponse sous forme de chaîne de caractères.

    Args:
        agent: L'agent à interroger.
        query: La requête à poser à l'agent.

    Returns:
        La réponse de l'agent sous forme de chaîne de caractères.
    """
    prompt = (
        """
        Pour la requête suivante, si elle nécessite de dessiner un tableau, répondez comme suit :
        {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

        Si la requête nécessite de créer un diagramme à barres, répondez comme suit :
        {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        Si la requête nécessite de créer un graphique linéaire, répondez comme suit :
        {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        Il ne peut y avoir que deux types de graphiques, "bar" et "line".

        Si la requête pose simplement une question qui ne nécessite ni tableau ni graphique, répondez comme suit :
        {"answer": "réponse"}
        Exemple :
        {"answer": "Le titre avec le meilleur classement est 'Gilead'"}

        Si vous ne connaissez pas la réponse, répondez comme suit :
        {"answer": "Je ne sais pas."}

        Renvoyez toutes les sorties sous forme de chaîne de caractères.

        Toutes les chaînes dans la liste "columns" et dans la liste des données doivent être entre guillemets doubles,

        Par exemple : {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

        Prenons les choses étape par étape.

        Voici la requête.
        Requête : 
        """ + query
    )

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )

    return response.choices[0].text.strip()
import openai
import os
import dotenv

# Charger les variables d'environnement
dotenv.load_dotenv()

# Récupérer la clé API depuis les variables d'environnement
API_KEY = os.getenv("apikey")
openai.api_key = API_KEY

def create_agent(dataframe):
    """
    Crée un agent qui peut accéder et utiliser un modèle de langage (LLM).

    Args:
        dataframe: Le DataFrame Pandas qui contient les données.

    Returns:
        Un agent qui peut accéder et utiliser le LLM.
    """
    llm = OpenAI(api_key=API_KEY, engine="text-davinci-003")
    return create_pandas_dataframe_agent(llm, dataframe)

def query_agent(agent, query):
    """
    Interroge un agent et retourne la réponse sous forme de chaîne de caractères.

    Args:
        agent: L'agent à interroger.
        query: La requête à poser à l'agent.

    Returns:
        La réponse de l'agent sous forme de chaîne de caractères.
    """
    prompt = (
        """
        Pour la requête suivante, si elle nécessite de dessiner un tableau, répondez comme suit :
        {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

        Si la requête nécessite de créer un diagramme à barres, répondez comme suit :
        {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        Si la requête nécessite de créer un graphique linéaire, répondez comme suit :
        {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        Il ne peut y avoir que deux types de graphiques, "bar" et "line".

        Si la requête pose simplement une question qui ne nécessite ni tableau ni graphique, répondez comme suit :
        {"answer": "réponse"}
        Exemple :
        {"answer": "Le titre avec le meilleur classement est 'Gilead'"}

        Si vous ne connaissez pas la réponse, répondez comme suit :
        {"answer": "Je ne sais pas."}

        Renvoyez toutes les sorties sous forme de chaîne de caractères.

        Toutes les chaînes dans la liste "columns" et dans la liste des données doivent être entre guillemets doubles,

        Par exemple : {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

        Prenons les choses étape par étape.

        Voici la requête.
        Requête : 
        """ + query
    )

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )

    return response.choices[0].text.strip()
