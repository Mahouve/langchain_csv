import streamlit as st
import os
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Image
import plotly.io as pio
from agent import query_agent, create_agent  # Import the agent functions

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
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.bar_chart(df)

    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.line_chart(df)

    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

def save_plotly_fig_as_image(fig, file_name):
    pio.write_image(fig, file_name, format='png')

def generate_pdf_report(df, bar_fig, line_fig, combo_fig, circle_fig, pie_fig, clustered_fig):
    """Generates a PDF report from a DataFrame and figures."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica", 20)
    c.drawString(30, height - 40, "Dataset Overview")

    c.setFont("Helvetica", 12)
    text = df.head().to_string(index=False)
    y_position = height - 60
    for line in text.split('\n'):
        c.drawString(30, y_position, line)
        y_position -= 15

    c.drawString(30, y_position - 40, "Basic Statistics")
    stats_text = df.describe().to_string()
    y_position -= 60
    for line in stats_text.split('\n'):
        c.drawString(30, y_position, line)
        y_position -= 15

    chart_y_position = y_position - 80

    # Convert Plotly figures to PNG images and save to temporary files
    bar_img_file = "bar_chart.png"
    line_img_file = "line_chart.png"
    combo_img_file = "combo_chart.png"
    circle_img_file = "circle_chart.png"
    pie_img_file = "pie_chart.png"
    clustered_img_file = "clustered_chart.png"

    pio.write_image(bar_fig, bar_img_file, format='png')
    pio.write_image(line_fig, line_img_file, format='png')
    pio.write_image(combo_fig, combo_img_file, format='png')
    pio.write_image(circle_fig, circle_img_file, format='png')
    pio.write_image(pie_fig, pie_img_file, format='png')
    pio.write_image(clustered_fig, clustered_img_file, format='png')

    # Insert images into PDF
    c.drawImage(bar_img_file, 30, chart_y_position, width=400, height=200)
    chart_y_position -= 220
    c.drawImage(line_img_file, 30, chart_y_position, width=400, height=200)
    chart_y_position -= 220
    c.drawImage(combo_img_file, 30, chart_y_position, width=400, height=200)
    chart_y_position -= 220
    c.drawImage(circle_img_file, 30, chart_y_position, width=400, height=200)
    chart_y_position -= 220
    c.drawImage(pie_img_file, 30, chart_y_position, width=400, height=200)
    chart_y_position -= 220
    c.drawImage(clustered_img_file, 30, chart_y_position, width=400, height=200)

    c.showPage()
    c.save()

    # Supprimer les fichiers temporaires
    os.remove(bar_img_file)
    os.remove(line_img_file)
    os.remove(combo_img_file)
    os.remove(circle_img_file)
    os.remove(pie_img_file)
    os.remove(clustered_img_file)

    buffer.seek(0)
    return buffer



def display_dashboard(df):
    """Function to display a dashboard with various visualizations."""
    st.header("Dataset Overview")
    st.write(df)

    st.header("Basic Statistics")
    st.write(df.describe())

    st.header("Visualizations")

    st.subheader("Bar Chart")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) == 0:
        st.error("No numeric columns available for bar chart.")
    else:
        column = st.selectbox("Select column for bar chart:", numeric_columns)
        bar_fig = px.bar(df, x=df.index, y=column)
        st.plotly_chart(bar_fig)

    st.subheader("Clustered Column Chart")
    if len(numeric_columns) < 2:
        st.error("At least two numeric columns are required for clustered column chart.")
    else:
        column1 = st.selectbox("Select first column for clustered column chart:", numeric_columns)
        column2 = st.selectbox("Select second column for clustered column chart:", numeric_columns, index=1)
        clustered_fig = go.Figure(data=[
            go.Bar(name=column1, x=df.index, y=df[column1]),
            go.Bar(name=column2, x=df.index, y=df[column2])
        ])
        clustered_fig.update_layout(barmode='group')
        st.plotly_chart(clustered_fig)

    st.subheader("Line Chart")
    column = st.selectbox("Select column for line chart:", df.columns, key='line_chart')
    line_fig = px.line(df, x=df.index, y=column)
    st.plotly_chart(line_fig)

    st.subheader("Combo Chart")
    if len(numeric_columns) < 2:
        st.error("At least two numeric columns are required for combo chart.")
    else:
        bar_column = st.selectbox("Select column for bar part of combo chart:", numeric_columns, key='combo_bar')
        line_column = st.selectbox("Select column for line part of combo chart:", numeric_columns, key='combo_line')
        combo_fig = go.Figure(data=[
            go.Bar(name=bar_column, x=df.index, y=df[bar_column]),
            go.Scatter(name=line_column, x=df.index, y=df[line_column], mode='lines+markers')
        ])
        combo_fig.update_layout(barmode='group')
        st.plotly_chart(combo_fig)

    st.subheader("Circle Chart")
    x_axis_circle = st.selectbox("Select X-axis for circle chart:", df.columns)
    y_axis_circle = st.selectbox("Select Y-axis for circle chart:", df.columns)
    circle_fig = go.Figure(data=[go.Scatter(x=df[x_axis_circle], y=df[y_axis_circle], mode='markers', marker=dict(size=10, symbol='circle'))])
    st.plotly_chart(circle_fig)

    st.subheader("Pie Chart")
    pie_column = st.selectbox("Select column for pie chart:", df.columns)
    pie_fig = px.pie(df, names=pie_column)
    st.plotly_chart(pie_fig)

    pdf_buffer = generate_pdf_report(df, bar_fig, line_fig, combo_fig, circle_fig, pie_fig, clustered_fig)
    st.download_button(label="Download PDF Report", data=pdf_buffer, file_name="report.pdf", mime="application/pdf")

# Sidebar
st.sidebar.title("User Authentication")

# User authentication radio button
auth_option = st.sidebar.radio("Choose an option:", ("Sign In", "Sign Up"))

if auth_option == "Sign In":
    st.sidebar.subheader("Sign In")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    sign_in_button = st.sidebar.button("Sign In")
    if sign_in_button:
        st.sidebar.success(f"Signed in as {username}")

elif auth_option == "Sign Up":
    st.sidebar.subheader("Sign Up")
    new_username = st.sidebar.text_input("New Username")
    new_password = st.sidebar.text_input("New Password", type="password")
    confirm_password = st.sidebar.text_input("Confirm Password", type="password")
    sign_up_button = st.sidebar.button("Sign Up")
    if sign_up_button:
        if new_password == confirm_password:
            st.sidebar.success(f"Account created for {new_username}")
        else:
            st.sidebar.error("Passwords do not match")

# Main content
st.image("1712677670457.png", use_column_width=True)
st.title("Chat with your CSV or Excel file")

st.write("Please upload your CSV, XLSX, or XLS file below.")

data = st.file_uploader("Upload a CSV, XLSX, or XLS file", type=["csv", "xlsx", "xls"])

if data is not None:
    st.write("Votre fichier est en cours de traitement. Veuillez patienter...")

    try:
        encodings = ['latin1']
        for encoding in encodings:
            try:
                if data.name.endswith('.csv'):
                    #df = pd.read_csv(data, delimiter=';', encoding='latin1', on_bad_lines='skip')
                    df = pd.read_csv(data, delimiter=';', encoding='latin1', error_bad_lines=False)
                    break
                elif data.name.endswith('.xlsx'):
                    df = pd.read_excel(data, engine='openpyxl')
                    break
                elif data.name.endswith('.xls'):
                    df = pd.read_excel(data, engine='xlrd')
                    break
            except Exception as e:
                st.warning(f"Failed to read file with encoding {encoding}: {e}")
                continue

        # Convert non-numeric columns to string to avoid conversion errors
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)

        display_dashboard(df)

        query = st.text_area("Insert your query")

        if st.button("Submit Query"):
            agent = create_agent(df)
            response = query_agent(agent=agent, query=query)
            decoded_response = decode_response(response)
            write_response(decoded_response)
    
    except Exception as e:
        st.error(f"Error processing the file: {e}")
        print(e)  # Ajout de cette ligne pour afficher l'erreur complète dans la console



