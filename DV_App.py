import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import plotly.subplots as ps
import numpy as np
import scipy.stats as st
import base64
from plotly.subplots import make_subplots
from dash.exceptions import PreventUpdate



dfn = pd.read_csv('/Users/bhoomikan/Documents/Reza_Dataviz/Project/IncidentFileF.csv')
#dfn = dfn[0:20000]

image1_path = '/Users/bhoomikan/PycharmProjects/pythonProject/assets/rfc_bhn1.png'
image2_path = '/Users/bhoomikan/PycharmProjects/pythonProject/assets/rfc_bhn2.png'
image3_path = '/Users/bhoomikan/PycharmProjects/pythonProject/assets/rfc_bhn3.png'
image4_path = '//Users/bhoomikan/PycharmProjects/pythonProject/assets/rfc_bhn4.png'
image5_path = '/Users/bhoomikan/PycharmProjects/pythonProject/assets/hmap.png'
image6_path = '/Users/bhoomikan/PycharmProjects/pythonProject/assets/Heatmap_TargetE.png'
image7_path = '/Users/bhoomikan/PycharmProjects/pythonProject/assets/featureimp_TargetE.png'
image8_path = '/Users/bhoomikan/PycharmProjects/pythonProject/assets/Scatter1_E.png'
image9_path = '/Users/bhoomikan/PycharmProjects/pythonProject/assets/Scatter2_E.png'

# Define Dash app
my_app = dash.Dash(__name__)

# Define features for dropdowns
continuous_features = ['reassignment_count', 'reopen_count', 'sys_mod_count', 'completion_time_days']
categorical_features = ['knowledge', 'contact_type', 'impact', 'urgency', 'priority', 'notify', 'closed_code', 'category', 'subcategory', 'assignment_group_num', 'resolved_by_num', 'location', 'made_sla', 'u_priority_confirmation']
cat_1 = ['u_symptom_num','category_num','subcategory_num','closed_code_num','assignment_group_num','assigned_to_num','location_num','resolved_by_num']
def encode_image(image_path):
    with open(image_path, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('ascii')
    return f'data:image/png;base64,{encoded_image}'

image1_encoded = encode_image(image1_path)
image2_encoded = encode_image(image2_path)
image3_encoded = encode_image(image3_path)
image4_encoded = encode_image(image4_path)
image5_encoded = encode_image(image5_path)
image6_encoded = encode_image(image6_path)
image7_encoded = encode_image(image7_path)
image8_encoded = encode_image(image8_path)
image9_encoded = encode_image(image9_path)

image_row = html.Div([
    html.Div([
        html.Img(src=image1_encoded, style={'width': '100%', 'height': 'auto'}),
    ], style={'width': '33%', 'display': 'inline-block'}),
    html.Div([
        html.Img(src=image2_encoded, style={'width': '100%', 'height': 'auto'}),
    ], style={'width': '33%', 'display': 'inline-block'}),
    html.Div([
        html.Img(src=image3_encoded, style={'width': '100%', 'height': 'auto'}),
    ], style={'width': '33%', 'display': 'inline-block'}),
])

chart_row = html.Div([
    html.Div([
        dcc.Graph(
            id='feature-graph4'
        ),
    ], style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(
            id='feature-graph5'
        ),
    ], style={'width': '100%', 'display': 'inline-block'}),
])

def outlier(df, f):
    Q1 = np.percentile(df[f], 25)
    Q3 = np.percentile(df[f], 75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    cleaned_df_cd = df[(df[f] >= lower_bound) & (df[f] <= upper_bound)]

    return cleaned_df_cd

def out_plots(f, plot_type, removal_status):  # Example data
    plot_type = plot_type.lower()
    if removal_status == 'After':
        # Outlier removal logic
        Q1 = dfn[f].quantile(0.25)
        Q3 = dfn[f].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_a = dfn[(dfn[f] >= lower_bound) & (dfn[f] <= upper_bound)]
        data_df = df_a
    else:
        data_df = dfn

    if 'histogram' in plot_type:
        fig = go.Figure(data=[go.Histogram(x=data_df[f])])
    elif 'boxplot' in plot_type:
        fig = go.Figure(data=[go.Box(y=data_df[f])])
    else:
        fig = go.Figure()
        fig.add_annotation(text="Invalid plot type entered. Please use 'Histogram' or 'Boxplot'.",
                           xref="paper", yref="paper", showarrow=False, font=dict(size=16, color="red"))
    fig.update_layout(title=f'{plot_type.capitalize()} of {f} with {removal_status} Outlier Removal')
    return fig

def dist_plots(dfn, f):
    dfn = outlier(dfn, f)
    fig = ps.make_subplots(rows=1, cols=3, subplot_titles=("Histogram", "Boxplot"))
    fig.add_trace(go.Histogram(x=dfn[f], histnorm='probability density', name='Histogram'), row=1, col=1)
    fig.add_trace(go.Box(y=dfn[f], boxmean=True, name='Boxplot'), row=1, col=2)
    qq_data = st.probplot(dfn[f].dropna(), dist="norm")  # Assuming a normal distribution
    fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='Q-Q Plot'), row=1, col=3)
    fig.add_trace(
        go.Scatter(x=qq_data[0][0], y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0], mode='lines', name='Fit'), row=1,
        col=3)
    fig.update_layout(height=600, width=1000, title_text=f"Histogram, Boxplot, QQ-Plot for {f}")
    return fig

def plotly_plot(f,merged_df):
    fig = go.Figure()

                # Add bar plot for unique_incident_count
    fig.add_trace(go.Bar(
                    x=merged_df[f],
                    y=merged_df['unique_incident_count'],
                    name='Unique Incident Count'
                ))

                # Add line plot for completion_time_days (avg)
    fig.add_trace(go.Scatter(
                    x=merged_df[f],
                    y=merged_df['completion_time_days (avg)'],
                    mode='lines',
                    name='Completion Time Days (Avg)',
                    yaxis='y2'
                ))

                # Update layout
    fig.update_layout(
                    title=f'Unique Incident Count and Completion Time Days (avg) by {f}',
                    xaxis=dict(title=f'Feature {f}'),
                    yaxis=dict(title='Unique Incident Count'),
                    yaxis2=dict(title='Completion Time Days (Avg)', overlaying='y', side='right'),
                    legend=dict(x=0, y=1.1, traceorder='normal')
                )

                # Show plot
    return fig
def freqplot(dfn,f):
    dfn = outlier(dfn, f)
    proportions_df = dfn.groupby(f)['number'].nunique().reset_index(name='unique_incident_count')
    median_response_time_df = dfn.groupby([f])['completion_time_days'].median().reset_index(name='completion_time_days (avg)')
    merged_df = pd.merge(proportions_df, median_response_time_df, on=f, how='inner')
    fig = plotly_plot(f, merged_df)
    return fig

def create_category_table(dfn, feature):
    # Aggregate data similar to what you would display in the bar chart
    proportions_df = dfn.groupby(feature)['number'].nunique().reset_index(name='unique_incident_count')
    median_response_time_df = dfn.groupby([feature])['completion_time_days'].median().reset_index(
        name='completion_time_days (avg)')
    aggregated_data = pd.merge(proportions_df, median_response_time_df, on=feature, how='inner')

    # Create the table trace
    table_trace = go.Table(
        header=dict(values=[feature, 'unique_incident_count','completion_time_days (avg)'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[aggregated_data[feature], aggregated_data['unique_incident_count'],aggregated_data['completion_time_days (avg)']],
                   fill_color='lavender',
                   align='left'))

    # Create a figure and add the table trace
    fig = go.Figure(data=[table_trace])
    fig.update_layout(width=500, height=300)  # Adjust size as needed
    return fig

def category_table(dfn, feature):
    proportions_df = dfn.groupby(feature)['number'].nunique().reset_index(name='unique_incident_count')
    median_response_time_df = dfn.groupby([feature])['completion_time_days'].median().reset_index(
        name='completion_time_days (avg)')
    aggregated_data = pd.merge(proportions_df, median_response_time_df, on=feature, how='inner')
    return aggregated_data

my_app.layout = html.Div([
    html.H1(children='Project', style={'textAlign': 'center'}),
    html.Br(),
    dcc.Tabs(id='hw-questions', children=[
        dcc.Tab(label='Data Preprocessing', value='q1'),
        dcc.Tab(label='Distribution Plots', value='q2') ,
        dcc.Tab(label='Normality Test', value='q3'),
        dcc.Tab(label='Frequency_CompletionTime (Cat)', value='q4'),
        dcc.Tab(label='Feature Importance', value='q5'),
        dcc.Tab(label='Dynamic_Plot', value='q6')
    ]),
    html.Div(id='layout')
])

get_question2_layout = html.Div([
    html.H4('Pick the type of feature'),
    dcc.RadioItems(
        id='feature-type',
        options=[
            {'label': 'Continuous Feature', 'value': 'cont'},
            {'label': 'Categorical Feature', 'value': 'cat'}
        ],
        value='cont',
        style={'padding': '20px'}  # Adds padding around each option for more space
    ),
    html.Br(),
    html.H4('Select the feature'),
    dcc.Dropdown(
        id='feature-dropdown'
    ),
    html.Br(),
    html.H4('Following is the distribution plot for the selected feature'),
    dcc.Graph(
        id='feature-graph'
    ),
    html.Div([
    html.Button("Download Table", id="download-button",n_clicks=0,style={
        'display': 'block',   # This makes the button a block element which can be centered
        'margin-left': 'auto',  # These two auto margins center the block element
        'margin-right': 'auto',
        'font-weight': 'bold',  # Makes the text bold
        'width': '10%',        # Sets the width of the button to 50% of its container
        'text-align': 'center'  # Centers the text inside the button
    }),
    html.Div('Tooltip Content', id='tooltip', style={'display': 'none'})
    ]),
    dcc.Download(id="download-table")
])

@my_app.callback(
    dash.dependencies.Output('tooltip', 'style'),
    [dash.dependencies.Input('download-button', 'n_clicks')]
)
def show_tooltip(n_clicks):
    if n_clicks % 2 == 1:
        return {'display': 'block', 'position': 'absolute', 'zIndex': '2', 'top': '20px', 'left': '20px'}
    else:
        return {'display': 'none'}

@my_app.callback(
    Output('feature-dropdown', 'options'),
    Input('feature-type', 'value')
)
def set_features_options(selected_feature_type):
    if selected_feature_type == 'cont':
        return [{'label': i, 'value': i} for i in continuous_features]
    elif selected_feature_type == 'cat':
        return [{'label': i, 'value': i} for i in cat_1]

@my_app.callback(
    Output('feature-graph', 'figure'),
    [Input('feature-dropdown', 'value'),
     Input('feature-type', 'value')]
)

def update_graph(selected_feature, feature_type):
    if not selected_feature:
        return go.Figure()  # Return an empty figure if no feature is selected

    if feature_type == 'cat':
        # Generate bar chart as before
        bar_fig = freqplot(dfn,selected_feature)

        table_fig = create_category_table(dfn, selected_feature)

        # Combine both figures into one (stacked vertically)
        combined_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     vertical_spacing=0.05,
                                     subplot_titles=('Bar And Line Chart', 'Data Table'),
                                     specs=[[{"type": "xy"}], [{"type": "table"}]])

        for trace in bar_fig['data']:
            combined_fig.add_trace(trace, row=1, col=1)

        # Add table to the combined figure
        combined_fig.add_trace(table_fig['data'][0], row=2, col=1)

        # Update layout if necessary
        combined_fig.update_layout(height=800, showlegend=False)
        return combined_fig

    elif feature_type == 'cont':
        return dist_plots(dfn, selected_feature)
@my_app.callback(
    Output("download-table", "data"),
    [Input("download-button", "n_clicks"),
     Input('feature-dropdown', 'value')],
    prevent_initial_call=True
)
def download_table(n_clicks, selected_feature):
    if n_clicks is None:
        raise PreventUpdate
    df_to_download = category_table(dfn,selected_feature)
    return dict(content=df_to_download.to_csv(index=False, encoding='utf-8-sig'), filename=f"{selected_feature}_data.csv")

cols = ['knowledge_encoded', 'contact_type_encoded', 'impact_num', 'urgency_num', 'priority_num', 'notify_encoded',
        'closed_code_num', 'category_num', 'subcategory_num', 'assignment_group_num', 'resolved_by_num', 'location_num',
        'made_sla_encoded', 'u_priority_confirmation_encoded']

get_question4_layout = html.Div([
    html.H1("Data Exploration App", style={'text-align': 'center'}),
    html.Div([
        html.Label("Select Feature:"),
        dcc.Dropdown(
            id='feature-dropdown2',
            options=[{'label': col, 'value': col} for col in cols],
            value='Feature1'
        ),
    ]),
    html.Br(),
    html.Div(id='slider-container'), # Container for slider component
    html.Br(),
    html.Div([
        html.Label("Select Subfeatures:"),
        dcc.Checklist(
            id='subfeature-checklist',
            options=[{'label': col, 'value': col} for col in cols],
            inline=False,  # Set to False if you want vertical stacking
            style={'width': '100%', 'display': 'block'}  # Adjust the width as desired
        ),
    ], style={'width': '100%', 'display': 'block'}),
    html.Br(),
    dcc.Loading(
    id="loading-1",
    type="default",  # You can choose from 'graph', 'cube', 'circle', 'dot', or 'default'
    children=html.Div(dcc.Graph(id='feature-graph2'))
    )
])


@my_app.callback(
    Output('slider-container', 'children'),  # Update slider component
    Input('feature-dropdown2', 'value')
)
def update_slider(feature):
    min_value = int(dfn[feature].min())
    max_value = int(dfn[feature].max())
    # Create marks for each integer value between min and max
    marks = {str(i): {'label': str(i), 'style': {'transform': 'rotate(45deg)', 'white-space': 'nowrap'}}
             for i in range(min_value, max_value + 1)}

    return dcc.Slider(
        id='value-slider',
        min=min_value,
        max=max_value,
        value=min_value,
        marks=marks,
        step=10
    )


@my_app.callback(
    Output('feature-graph2', 'figure'),
    [Input('feature-dropdown2', 'value'),
     Input('value-slider', 'value'),
     Input('subfeature-checklist', 'value')]
)
def p_table(f, val, gf):
    fdf = dfn[dfn[f] == val]
    print(f)
    print(gf)
    print(val)
    proportions_df = fdf.groupby(gf)['number'].nunique().reset_index(name='unique_incident_count')
    median_response_time_df = fdf.groupby(gf)['completion_time_days'].median().reset_index(
        name='completion_time_days (avg)')
    merged_df = pd.merge(proportions_df, median_response_time_df, on=gf, how='inner')

    # Create the table trace
    table_trace = go.Table(
        header=dict(values=merged_df.columns,
                    fill=dict(color='#C2D4FF'),
                    align='left'),
        cells=dict(values=merged_df.values.T.tolist(),
                   fill=dict(color='#F5F8FF'),
                   align='left')
    )

    # Create the figure
    fig = go.Figure(data=table_trace)
    return fig

get_question1_layout = html.Div([
    html.Div([
        html.Div([
            html.Img(src=image1_encoded, style={'width': '100%', 'height': 'auto'}),
        ], style={'width': '50%', 'display': 'inline-block'}),
        html.Div([
            html.Img(src=image2_encoded, style={'width': '100%', 'height': 'auto'}),
        ], style={'width': '50%', 'display': 'inline-block'}),
    ]),
    html.Div([
        html.Div([
            html.Img(src=image3_encoded, style={'width': '100%', 'height': 'auto'}),
        ], style={'width': '50%', 'display': 'inline-block'}),
        html.Div([
            html.Img(src=image4_encoded, style={'width': '100%', 'height': 'auto'}),
        ], style={'width': '50%', 'display': 'inline-block'}),
    ]),
    html.Div([
        html.Div([
            html.Img(src=image5_encoded, style={'width': '100%', 'height': 'auto'}),
        ], style={'width': '100%', 'display': 'inline-block'}),
    ]),
    html.Br(),
    dcc.RadioItems(
        id='outlier-removal',
        options=[
            {'label': 'Before Outlier Removal', 'value': 'Before'},
            {'label': 'After Outlier Removal', 'value': 'After'}
        ],
        value='Before'
    ),
    html.Br(),
    html.Label('Enter Plot Type (Histogram or Boxplot):',style={'font-size': '18px', 'font-weight': 'bold'}),
    dcc.Input(
        id='plot-type',
        type='text',
        value='Histogram',
        placeholder='Enter "Histogram" or "Boxplot"',
        style={'width': '100%'}
    ),
    html.Br(),
    dcc.Dropdown(
        id='feature-dropdown3',
        options=[{'label': f, 'value': f} for f in continuous_features],
        value=continuous_features[0]
    ),
    html.Br(),
    dcc.Graph(
        id='feature-graph3'
    )
])


@my_app.callback(
    Output('feature-graph3', 'figure'),
    [Input('feature-dropdown3', 'value'),
     Input('plot-type', 'value'),
     Input('outlier-removal', 'value')]
)
def update_graph(f, plot_type, removal_status):
    if f is None or plot_type is None or removal_status is None:
        return go.Figure()  # Return empty plot if any input is None
    return out_plots(f, plot_type, removal_status)

get_question3_layout = html.Div([
    html.H1('Normality Test'),
    dcc.Dropdown(
        id='drop3',
        options=[{'label': i, 'value': i} for i in continuous_features],
        placeholder="Select a feature",  # Custom placeholder text
        searchable=True  # Enables search functionality
    ),
    html.Br(),
    dcc.Dropdown(
        id='drop4',
        options=[
            {'label': 'D’Agostino’s K-squared test', 'value': 'Da_k_squared'},
            {'label': 'Kolmogorov-Smirnov test', 'value': 'K_S test'},
            {'label': 'Shapiro-Wilk test', 'value': 'Shapiro Test'}
        ],
        placeholder="Select a normality test",  # Custom placeholder text
        searchable=True
    ),
    html.Br(),
    html.Div(id='output3')
])


@my_app.callback(
    Output(component_id='output3', component_property='children'),
    [Input(component_id='drop3', component_property='value'),
     Input(component_id='drop4', component_property='value')]
)
def update_display(input1, input2):
    if input1 is None or input2 is None:
        return ""  # or you can return an empty div

    # Calculate mean and standard deviation of the input feature
    mean_f = np.mean(dfn[input1])
    std_f = np.std(dfn[input1])

    # Initialize stat and pvalue variables
    stat = None
    pvalue = None
    fstr = ""

    # Determine which statistical test to perform based on input2
    if input2 == 'K_S test':
        result = st.kstest(dfn[input1], 'norm', args=(mean_f, std_f))
        stat = result.statistic.round(2)
        pvalue = result.pvalue.round(2)
    elif input2 == 'Da_k_squared':
        k2_stat_f, p_value_f = st.normaltest(dfn[input1])
        stat = k2_stat_f.round(2)
        pvalue = p_value_f.round(2)
    elif input2 == 'Shapiro Test':
        result = st.shapiro(dfn[input1])
        stat = result[0].round(2)
        pvalue = result[1].round(2)

    # Check the p-value to determine if the distribution is normal
    if pvalue < 0.05:  # Typically, p < 0.05 is considered significant
        fstr = f'Distribution is not normal for {input1}'
    else:
        fstr = f'Distribution is normal for {input1}'

    # Return a tuple containing the statistics and the formatted string
    return [html.Div(f'The selected Output is ({stat}, {pvalue})', style={'font-weight': 'bold'}),
            html.Div(fstr, style={'font-weight': 'bold'})]

get_question5_layout = html.Div([
    html.H1("Feature Importance"),
    html.Div([
        html.Div([
            html.Img(src=image6_encoded, style={'width': '50%', 'height': 'auto', 'display': 'inline-block'}),
            html.Img(src=image7_encoded, style={'width': '50%', 'height': 'auto', 'display': 'inline-block'})
        ], style={'width': '100%', 'display': 'flex'}),
        html.Br(),
        html.Div([
            html.Img(src=image8_encoded, style={'width': '50%', 'height': 'auto', 'display': 'inline-block'}),
            html.Img(src=image9_encoded, style={'width': '50%', 'height': 'auto', 'display': 'inline-block'})
        ], style={'width': '100%', 'display': 'flex'})
    ])
])

dfn['opened_at_custom'] = pd.to_datetime(dfn['opened_at_custom'])

# Extract month and day name
dfn['Month_Opened'] = dfn['opened_at_custom'].dt.month
dfn['Day_Opened'] = dfn['opened_at_custom'].dt.day_name()

# Define the layout of the app
question6_layout = html.Div([
    html.Br(),
    html.Label('Select the month for which you want to see the graph:',style={'font-size': '18px', 'font-weight': 'bold'}),
    html.Br(),
    html.Br(),
    dcc.Slider(
        id='month-slider',
        min=min(dfn['Month_Opened']),
        max=max(dfn['Month_Opened']),
        step=1,
        value=min(dfn['Month_Opened'])
        #marks={val: str(val) for val in dfn['Month_Opened'].unique()}
    ),
    html.Br(),
    dcc.Graph(id='output-graph')
])

# Define callback to update the graph based on the slider value
@my_app.callback(
    Output('output-graph', 'figure'),
    [Input('month-slider', 'value')]
)
def update_graph(val):
    dft = dfn[dfn['Month_Opened'] == val]
    dft = dft.groupby('Day_Opened')['completion_time_days'].median().reset_index(name='completion_time_days (avg)')

    # Create the line plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dft['Day_Opened'], y=dft['completion_time_days (avg)'],
                             mode='lines+markers', name='Completion Time Days (Avg)'))

    fig.update_layout(title=f'Completion Time Days (Avg) By Day For Month: {val}',
                      xaxis_title='Day_Opened',
                      yaxis_title='Completion Time Days (Avg)',
                      showlegend=True)

    return fig

@my_app.callback(
    Output(component_id='layout', component_property='children'),
    Input(component_id='hw-questions', component_property='value')
)

def update_layout(ques):
    if ques == 'q1':
        return get_question1_layout
    elif ques == 'q4':
        return get_question4_layout
    elif ques == 'q3':
        return get_question3_layout
    elif ques == 'q2':
        return get_question2_layout
    elif ques == 'q5':
       return get_question5_layout
    elif ques == 'q6':
        return question6_layout


# if __name__ == 'main':
my_app.run_server(
    debug=True,
    port=8031,
    host='0.0.0.0'
)

