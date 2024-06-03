    # Generate and save charts as images
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

    # Clean up temporary files
    for chart_file in chart_files:
        os.remove(chart_file)

    buffer.seek(0)
    return buffer

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