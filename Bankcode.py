#Interface.py

import streamlit as st
import pandas as pd
import json
import plotly.express as px
from agent import query_agent, create_agent
import plotly.graph_objects as go

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

    # Plotly scatter plot (Circle Chart)
    st.subheader("Circle Chart")
    x_axis_circle = st.selectbox("Select X-axis for circle chart:", df.columns)
    y_axis_circle = st.selectbox("Select Y-axis for circle chart:", df.columns)
    circle_fig = go.Figure(data=[go.Scatter(x=df[x_axis_circle], y=df[y_axis_circle], mode='markers', marker=dict(size=10, symbol='circle'))])
    st.plotly_chart(circle_fig)

    # Plotly pie chart
    st.subheader("Pie Chart")
    pie_column = st.selectbox("Select column for pie chart:", df.columns)
    pie_fig = px.pie(df, names=pie_column)
    st.plotly_chart(pie_fig)

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
        
        
#Agent.py

import openai
import os
import dotenv
import pandas as pd

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
        Le DataFrame Pandas, agissant comme un agent simplifié.
    """
    return dataframe

def query_agent(agent, query):
    """
    Interroge un agent et retourne la réponse sous forme de chaîne de caractères.

    Args:
        agent: Le DataFrame Pandas agissant comme agent.
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

