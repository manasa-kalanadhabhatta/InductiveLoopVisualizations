# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:54:00 2017

@author: Manasa
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import calendar
from dateutil.parser import parse
import datetime as dt
from ast import literal_eval

app = dash.Dash()

df = pd.read_csv("data/filteredAprilData.csv", header=0)
df[df.columns.tolist()[1:]] = df[df.columns.tolist()[1:]].apply(pd.Series.round)
df['day'] = df.apply(lambda row: calendar.day_name[parse(row['dateTime']).weekday()], axis=1)
df['hour'] = df.apply(lambda row: parse(row['dateTime']).hour, axis=1)

loopPositions = pd.read_csv("data/loopPositions.csv", header=0)
loopPositions['loop'] = loopPositions.apply(lambda row: "X"+format(int(row['zone']), '02d')+"."+format(int(row['loop_id']), '02d'), axis=1)

noisy_loops = ['X02.05', 'X03.04', 'X03.12', 'X03.13', 'X03.22', 'X04.08', 'X05.05', 'X07.05', 'X07.07', 'X08.02', 'X08.04', 'X08.09', 'X08.10', 'X08.11', 'X09.05', 'X09.13', 'X10.04']
df = df[df.columns.difference(noisy_loops)]

#plotly default
mapbox_access_token = 'pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w'

colorscale = [[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'], [0.2222222222222222, 'rgb(244,109,67)'], [0.3333333333333333, 'rgb(253,174,97)'], [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'], [0.6666666666666666, 'rgb(171,217,233)'], [0.7777777777777778, 'rgb(116,173,209)'], [0.8888888888888888, 'rgb(69,117,180)'], [1.0, 'rgb(49,54,149)']]
plotly_scale, plotly_colors = np.array(colorscale)[:,0].astype(np.float), np.array(colorscale)[:,1]

cols=[color[4:-1] for color in plotly_colors]
colormap=np.array([literal_eval(col) for col in cols])/255.0

def get_color_from_val(val, plotly_scale=plotly_scale, colormap=colormap, vmax=-1, vmin=1):
    v = (val-vmin)/(vmax-vmin)
    idx = np.searchsorted(plotly_scale, v)-1
    v_norm = (v-plotly_scale[idx])/(plotly_scale[idx+1]-plotly_scale[idx])
    
    val_color = colormap[idx] + v_norm*(colormap[idx+1]-colormap[idx])
    val_color_scale = val_color*255.0
    color_code = 'rgb('+str(val_color_scale[0])+', '+str(val_color_scale[1])+', '+str(val_color_scale[2])+')'
    return color_code

all_options = {
        'Zone': ['X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08', 'X09', 'X10'],
        'Time': ['00-01', '01-02', '02-03', '03-04', '04-05', '05-06', '06-07', '07-08', '08-09', '09-10', '10-11', '11-12', '12-13', '13-14', '14-15', '15-16', '16-17', '17-18', '18-19', '19-20', '20-21', '21-22', '22-23', '23-0'],
        'Day': ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
}

app.layout = html.Div([

    html.Div([
        dcc.Graph(id='map', style={'height': '80%', 'display': 'block'}),
        dcc.Graph(id='tseries', style={'height': '20%', 'display': 'block', 'padding': '0 10 0 0'})
    ], style={'width': '80%', 'height': '100%', 'display': 'inline-block', 'background-color': '#D3D3D3'}),

    html.Div([
        html.Div([
            'Select Zone:',
            dcc.Dropdown(
                id = 'zone-dropdown',
                options = [{'label': k, 'value': k} for k in all_options['Zone']],
                value = ['X01'],
                multi = True
            ),

            'Select Time of the Day:',
            dcc.Dropdown(
                id = 'time-dropdown',
                options = [{'label': k, 'value': k} for k in all_options['Time']],
                value = ['00-01'],
                multi=True
            ),

            'Select Day of the Week:',
            dcc.Dropdown(
                id='day-dropdown',
                options = [{'label': k, 'value': k} for k in all_options['Day']],
                value = ['Sunday'],
                multi=True
            ),
            'Select Correlation Range:',
            dcc.RangeSlider(
                id='corr_slider',
                min=-100,
                max=100,
                value=[-100, 100]
            ),
            html.P('-1.00 to 1.00', id='corr-text', style={'width': '30%', 'display': 'inline'}),
            html.Button('Update Map', id='button', style={'width': '70%', 'display': 'inline'}),
        ], style={'width': '100%', 'height': '55%', 'display': 'block', 'padding': '5 10 0 5'}),
        dcc.Graph(id='histogram', style={'height': '15%', 'display': 'block'}),
        dcc.Graph(id='graph', style={'height': '30%', 'display': 'block', 'background-color': '#D3D3D3'})
    ], style={'width': '20%', 'height': '100%', 'display': 'inline-block', 'float': 'right', 'background-color': '#D3D3D3'}),

], style={'width': '100%', 'height': '100%', 'background-color': '#D3D3D3'})

@app.callback(
    dash.dependencies.Output('corr-text', 'children'),
    [dash.dependencies.Input('corr_slider', 'value')])
def update_corr_text(corr_slider):
    return "{} to {}".format(corr_slider[0]/100,corr_slider[1]/100)

@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [dash.dependencies.Input('zone-dropdown', 'value'),
    dash.dependencies.Input('time-dropdown', 'value'),
    dash.dependencies.Input('day-dropdown', 'value'),
    dash.dependencies.Input('corr_slider', 'value')])
def update_graph(zone, time, day, corr_val):
    hour = [int(str(t).split('-')[0]) for t in time]
    min_cor = corr_val[0]/100
    max_cor = corr_val[1]/100

    df_new = df[df['day'].isin(day)]
    df_new = df_new[df['hour'].isin(hour)]
    df_new = df_new.select(lambda col: col.startswith(tuple(zone)), axis=1)
    
    cor_matrix = df_new.corr()
    cor_matrix = cor_matrix.dropna(axis=0, how='all')
    cor_matrix = cor_matrix.dropna(axis=1, how='all')
    
    cor_matrix[cor_matrix < min_cor] = np.nan
    cor_matrix[cor_matrix > max_cor] = np.nan

    return{
        'data':[go.Heatmap(
            z=cor_matrix.values.tolist(),
            x=list(cor_matrix),
            y=list(cor_matrix),
            zmin=-1.0,
            zmax=1.0,
            colorscale=colorscale,
            reversescale=True
        )],

        'layout': {
            'title': '',
            'width': 250,
            'height': 250,
            'autosize': True,
            'margin': {'t':0, 'b':0, 'l':0, 'r':0},
            'plot_bgcolor':'#DCDCDC',
            'paper_bgcolor':'#D3D3D3',
            'xaxis': {'showticklabels':'False'},
            'yaxis': {'showticklabels':'False'}
        }
    }

@app.callback(
    dash.dependencies.Output('map', 'figure'),
    [dash.dependencies.Input('button', 'n_clicks')],
    state=[dash.dependencies.State('zone-dropdown', 'value'),
    dash.dependencies.State('time-dropdown', 'value'),
    dash.dependencies.State('day-dropdown', 'value'),
    dash.dependencies.State('corr_slider', 'value')])
def update_map(n_clicks, zone, time, day, corr_val):
    hour = [int(str(t).split('-')[0]) for t in time]
    min_cor = corr_val[0]/100
    max_cor = corr_val[1]/100

    #get observations corresponding to filter
    df_new = df[df['day'].isin(day)]
    df_new = df_new[df['hour'].isin(hour)]
    df_new = df_new.select(lambda col: col.startswith(tuple(zone)), axis=1)
    
    #calculate correlation
    cor_matrix = df_new.corr()
    #drop rows and columns with all NAs
    cor_matrix = cor_matrix.dropna(axis=0, how='all')
    cor_matrix = cor_matrix.dropna(axis=1, how='all')
    
    #make values above and below filter NA
    cor_matrix[cor_matrix < min_cor] = np.nan
    cor_matrix[cor_matrix > max_cor] = np.nan

    #loops which are in the heatmap
    loopNames = cor_matrix.columns.tolist()
    loopCombinations = [(loopNames[i], loopNames[j]) for i in range(len(loopNames)) for j in range(i+1,len(loopNames))]

    loopPositions_new = loopPositions.loc[loopPositions['loop'].str.startswith(tuple(str(zone)))]
    loopPositions_new = loopPositions_new[loopPositions_new['loop'].isin(loopNames)]
    
    cor_max = cor_matrix.values.max()

    # begin scattergeo

    # loops = [ dict(
    #     type = 'scattergeo',
    #     locationmode = 'europe',
    #     lon = loopPositions_new['long'],
    #     lat = loopPositions_new['lat'],
    #     hoverinfo = 'text',
    #     text = loopPositions_new['loop'],
    #     mode = 'markers',
    #     marker = dict( 
    #         size=5, 
    #         color='rgb(255, 0, 0)'
    #     ))]

    # lines = []
    # for l in loopCombinations:
    #     print(l)
    #     try:
    #         line = dict(
    #             type = 'scattergeo',
    #             locationmode = 'europe',
    #             lon = [ loopPositions_new.loc[loopPositions_new['loop'] == l[0], 'long'].iloc[0], loopPositions_new.loc[loopPositions_new['loop'] == l[1], 'long'].iloc[0] ],
    #             lat = [ loopPositions_new.loc[loopPositions_new['loop'] == l[0], 'lat'].iloc[0], loopPositions_new.loc[loopPositions_new['loop'] == l[1], 'lat'].iloc[0] ],
    #             mode = 'lines',
    #             line = dict(
    #                 width = cor_matrix[l[0]][l[1]]/cor_max,
    #                 color = 'red',
    #             )
    #         )
    #         lines.append(line)
    #     except IndexError:
    #         pass

    # layout = dict(
    #     title = 'Loop Correlations<br>(Hover for loop names)',
    #     showlegend = False, 
    #     geo = dict(
    #         scope='europe',
    #         projection=dict( type='azimuthal equal area' ),
    #         showland = True,
    #         landcolor = 'rgb(243, 243, 243)',
    #         countrycolor = 'rgb(204, 204, 204)',
    #         lonaxis=dict(range = [-8.688748, -8.583575]),
    #         lataxis=dict(range = [41.130600, 41.173612])
    #     ),
    # )

    #end scattergeo

    #begin mapbox
    
    trace = dict(
        type='scattermapbox',
        lon=loopPositions_new['long'],
        lat=loopPositions_new['lat'],
        text=loopPositions_new['loop'],
        mode='markers',
        hoverinfo='text'
    )

    layers = []

    for l in loopCombinations:
        try:
            if(cor_matrix[l[0]][l[1]] != np.nan):
                features = [{
                    'type': 'Feature',
                    'properties': {},
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [
                            [loopPositions_new.loc[loopPositions_new['loop'] == l[0], 'long'].iloc[0], loopPositions_new.loc[loopPositions_new['loop'] == l[0], 'lat'].iloc[0]],
                            [loopPositions_new.loc[loopPositions_new['loop'] == l[1], 'long'].iloc[0], loopPositions_new.loc[loopPositions_new['loop'] == l[1], 'lat'].iloc[0]]
                        ]
                    }
                }]
                
                source = {
                    'type': 'FeatureCollection',
                    'features': features
                }

                layer = dict(
                    sourcetype = 'geojson',
                    source = source,
                    type = 'line',
                    color=get_color_from_val(cor_matrix[l[0]][l[1]])
                )

                layers.append(layer)

        except IndexError:
            pass
    
    layout = go.Layout(
        title="",
        autosize=True,
        width=1050,
        height=450,
        hovermode='closest',
        margin=dict(t=0, l=0, r=0, b=0),
        plot_bgcolor="#DCDCDC",
        paper_bgcolor="#D3D3D3",
        mapbox=dict(
            accesstoken=mapbox_access_token,
            layers=layers,
            bearing=0,
            center=dict(
                lat=loopPositions_new['lat'].median(),
                lon=loopPositions_new['long'].median()
            ),
            pitch=0,
            zoom=13
        )
    )
    
    fig = dict(data=go.Data([trace]), layout=layout)
    
    return fig


@app.callback(
    dash.dependencies.Output('histogram', 'figure'),
    [dash.dependencies.Input('zone-dropdown', 'value'),
    dash.dependencies.Input('time-dropdown', 'value'),
    dash.dependencies.Input('day-dropdown', 'value'),
    dash.dependencies.Input('corr_slider', 'value')])
def update_hist(zone, time, day, corr_val):
    hour = [int(str(t).split('-')[0]) for t in time]
    min_cor = corr_val[0]/100
    max_cor = corr_val[1]/100

    overall_corr = df.select(lambda col: col.startswith('X'), axis=1).corr()
    overall_corr = overall_corr.applymap(lambda x: round(x / 0.05) * 0.05)
    overall_corr_count = overall_corr.apply(pd.Series.value_counts).sum(axis=1)    

    df_new = df[df['day'].isin(day)]
    df_new = df_new[df['hour'].isin(hour)]
    df_new = df_new.select(lambda col: col.startswith(tuple(zone)), axis=1)
    
    cor_matrix = df_new.corr()
    cor_matrix = cor_matrix.dropna(axis=0, how='all')
    cor_matrix = cor_matrix.dropna(axis=1, how='all')
    
    cor_matrix[cor_matrix < min_cor] = np.nan
    cor_matrix[cor_matrix > max_cor] = np.nan

    cor_frame = cor_matrix.stack().reset_index()
    cor_frame.columns = ['loop1', 'loop2', 'correlation']

    col_comb = []
    drop_index = []
    for index, row in cor_frame.iterrows():
        comb = row['loop1']+row['loop2']
        comb_rev = row['loop2']+row['loop1']
        if (comb or comb_rev) in col_comb:
            drop_index.append(index)
        col_comb.append(comb)
        col_comb.append(comb_rev)

    cor_frame_new = cor_frame.drop(cor_frame.index[drop_index])
    cor_frame_new['bucket'] = cor_frame_new.apply(lambda row: round(row['correlation'] / 0.05) * 0.05, axis=1)

    cor_df = pd.DataFrame(cor_frame_new.bucket.value_counts().reset_index())
    cor_df = cor_df.sort_values('bucket')
    
    fig = dict(
        data=go.Data([go.Bar(
            x=cor_df['index'],
            #y=[num-min(cor_df['bucket'])/max(cor_df['bucket'])-min(cor_df['bucket']) for num in cor_df['bucket']],
            y=[num/np.linalg.norm(cor_df['bucket']) for num in cor_df['bucket']],
            name='Selected',
            showlegend=False
        ), go.Scatter(
            x=overall_corr_count.index.tolist(),
            #y=[num-min(overall_corr_count.tolist())/max(overall_corr_count.tolist())-min(overall_corr_count.tolist()) for num in overall_corr_count.tolist()],
            y=[num/np.linalg.norm(overall_corr_count.tolist()) for num in overall_corr_count.tolist()],
            name='Overall',
            showlegend=False
        )]),

        layout=go.Layout(
            title="",
            xaxis=dict(
                range=[-1,1]),
            margin=dict(t=0, l=10, r=0, b=0),
            autosize=True,
            plot_bgcolor="#D3D3D3",
            paper_bgcolor="#D3D3D3",
            height=120,
            width=250
        )
    )

    return fig

@app.callback(
    dash.dependencies.Output('tseries', 'figure'),
    [dash.dependencies.Input('zone-dropdown', 'value'),
    dash.dependencies.Input('time-dropdown', 'value'),
    dash.dependencies.Input('day-dropdown', 'value'),
    dash.dependencies.Input('corr_slider', 'value')])
def update_tseries(zone, time, day, corr_val):
    hour = [int(str(t).split('-')[0]) for t in time]
    min_cor = corr_val[0]/100
    max_cor = corr_val[1]/100

    df_new = df[df['day'].isin(day)]
    df_new = df_new[df['hour'].isin(hour)]
    df_new = df_new.select(lambda col: col.startswith(tuple(zone))|col.startswith('dateTime'), axis=1)
    
    df_new['label'] = df_new.apply(lambda row: row['dateTime'][8:10]+'/'+row['dateTime'][5:7]+'/'+row['dateTime'][2:4]+' '+row['dateTime'][11:13]+':'+row['dateTime'][14:16], axis=1)
    
    traces = []
    for l in df_new.columns:
        if(l!='dateTime' and l!='label'):
            trace = go.Scatter(
                x=df_new['label'],
                y=df_new[l],
                name=l
            )
            traces.append(trace)

    data = go.Data(traces)

    layout = go.Layout(
        title = "",
        margin=dict(b=25, t=0, l=20),
        width=1050,
        height=200,
        plot_bgcolor="#DCDCDC",
        paper_bgcolor="#D3D3D3",
        xaxis=dict(showticklabels=False)
    )

    fig = dict(data=data, layout=layout)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)