import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

# Set the OpenAI API key
apikey = "sk-proj-eLPhMcPUlLYbCrUw6a6RT3BlbkFJ09Wo4wl6IOHYGmgjBSqk"
os.environ['OPENAI_API_KEY'] = apikey
load_dotenv(find_dotenv())

# Streamlit app code...
# Title
st.image("1712677670457.png", use_column_width=True)
st.title("Insights, KPI & Dashboards")

# Welcoming message
st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")

# Explanation sidebar
# User Authentication
st.sidebar.title("User Authentication")
auth_option = st.sidebar.radio("Choose an option:", ("Sign In", "Sign Up"))

if auth_option == "Sign In":
    st.sidebar.subheader("Sign In")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    sign_in_button = st.sidebar.button("Sign In")
    if sign_in_button:
        st.sidebar.success("Signed in as {}".format(username))
elif auth_option == "Sign Up":
    st.sidebar.subheader("Sign Up")
    new_username = st.sidebar.text_input("New Username")
    new_password = st.sidebar.text_input("New Password", type="password")
    confirm_password = st.sidebar.text_input("Confirm Password", type="password")
    sign_up_button = st.sidebar.button("Sign Up")
    if sign_up_button:
        if new_password == confirm_password:
            st.sidebar.success("Account created for {}".format(new_username))
        else:
            st.sidebar.error("Passwords do not match")

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

# Function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True
st.button("Let's get started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    user_file = st.file_uploader("Upload your file here", type=["csv", "xlsx", "xls"])
    if user_file is not None:
        file_extension = user_file.name.split('.')[-1]
        if file_extension == 'csv':
            df = pd.read_csv(user_file, delimiter=';', low_memory=False, encoding='latin1')
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(user_file, engine='openpyxl' if file_extension == 'xlsx' else None)

        # Function to display the dashboard
        def display_dashboard(df):
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

            # llm model
            llm = OpenAI(temperature=0)

            # Function sidebar
            def steps_eda():
                steps_eda = llm('What are the steps of EDA')
                return steps_eda

            # Pandas agent
            pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True)

            # Functions main
            def function_agent():
                st.write("**Data Cleaning**")
                columns_df = pandas_agent.run("What are the meaning of the columns?")
                st.write(columns_df)
                missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
                st.write(missing_values)
                duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
                st.write(duplicates)

            def function_question_variable():
                st.write("**Data Overview**")
                st.write(df.head())
                st.write("**Data Summarisation**")
                st.write(df.describe())
                correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
                st.write(correlation_analysis)
                outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
                st.write(outliers)
                new_features = pandas_agent.run("What new features would be interesting to create?.")
                st.write(new_features)

            def function_question_dataframe(user_question_dataframe):
                dataframe_info = pandas_agent.run(user_question_dataframe)
                st.write(dataframe_info)

            # Main
            st.header('Exploratory data analysis')
            st.subheader('General information about the dataset')

            with st.sidebar:
                with st.expander('What are the steps of EDA'):
                    st.write(steps_eda())

            function_agent()

            st.subheader('Variable of study')
            user_question_variable = st.text_input('What variable are you interested in')
            if user_question_variable is not None and user_question_variable != "":
                function_question_variable()

                st.subheader('Further study')

            if user_question_variable:
                user_question_dataframe = st.text_input("Is there anything else you would like to know about your dataframe?")
                if user_question_dataframe and user_question_dataframe != "":
                    function_question_dataframe(user_question_dataframe)

        display_dashboard(df)

def generate_pdf_report(df, bar_fig, line_fig, combo_fig, circle_fig, pie_fig, clustered_fig):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    def save_plotly_fig_as_image(fig, filename):
        fig.write_image(filename)

    c.drawString(30, height - 50, "Dataset Overview")
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
    chart_files = []

    for fig, name in zip([bar_fig, line_fig, combo_fig, circle_fig, pie_fig, clustered_fig],
        ["bar_chart.png", "line_chart.png", "combo_chart.png", "circle_chart.png", "pie_chart.png", "clustered_chart.png"]):
        save_plotly_fig_as_image(fig, name)
        chart_files.append(name)

    for chart_file in chart_files:
        c.drawImage(chart_file, 30, chart_y_position, width=400, height=200)
        chart_y_position -= 220
        if chart_y_position < 100:
            c.showPage()
            chart_y_position = height - 80

    c.showPage()
    c.save()

    for chart_file in chart_files:
        os.remove(chart_file)

    buffer.seek(0)
    return buffer

footer = """
<style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: left;
        font-size: small;
        padding: 10px;
    }
</style>
<div class="footer">
    Analytiis est une marque Optiis-2024
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
