from django.http import HttpResponse
from .models import status
from django.template import loader
from django.shortcuts import render
import plotly.express as px

#plotting Libraries
from plotly.offline import plot
import plotly.graph_objs as go
import networkx as nx
import plotly.graph_objects as go
import pandas
import numpy as np
import math
from math import inf
#Url libraries
from PIL import Image
import requests
from io import BytesIO
# Create your views here.
from django.views.generic import(
    CreateView,
    DetailView,
    ListView,
    UpdateView,
    DeleteView
)

class sensorListView(ListView):
    template_name = 'list.html'
    queryset = status.objects.all()


def feature(request, sensor_id):
    return HttpResponse("<h2>Features for sensor id:" +str(sensor_id) +"</h2>")

def index(request):
    

    response = requests.get(
        'https://maps.googleapis.com/maps/api/staticmap?center=Quatre+Bornes,mauritius&zoom=15&size=4000x4000&key=AIzaSyA4MYO5w9d-ZVH55heiBF5ZaOCZ02wJPbY')
    img = Image.open(BytesIO(response.content))

    graph = {'nodeID': ['Winners candos', 'Victoria hospital', 'Intermart express', 'La City trianon', 'Textile Market',
                    ],
         'location': [(1, 2), (2.2, 0.9), (6, 2), (4, 5), (2, 2)],
         'garbage_volume': [1, 3, 6, 4, 10]}

    df = pandas.DataFrame(data=graph)
    j = 0
    i = 0
    j1 = 0
    i1 = 0


    def euclidian(location1, location2):
        x1 = location1[0]
        x2 = location2[0]
        y1 = location1[1]
        y2 = location2[1]

        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist


    d = euclidian(df['location'][1], df['location'][1])

    a = np.empty((0, 3), int)
    euclidian_data = []
    arraylist = []
    while j < len(df['location']):
        data = []
        while i < len(df['location']):
            data.append(int(euclidian(df['location'][i], df['location'][j])))
            i = i + 1

        euclidian_data.append(min([x for x in data if x != 0]))
        arraylist.append(data)
        i = 0
        j = j + 1

    arr = np.array(arraylist)

    dz = pandas.DataFrame(np.column_stack(arr), columns=['a', 'b', 'c', 'd', 'e'])
    dz.replace(to_replace=0, value=10, inplace=True)
    ds = pandas.DataFrame(np.column_stack([df, arr]),
                        columns=['columnID', 'location', 'garbage volume', 'a', 'b', 'c', 'd', 'e'])
    dx = pandas.DataFrame(np.column_stack([ds, euclidian_data]),
                        columns=['columnID', 'location', 'garbage volume', 'a', 'b', 'c', 'd', 'e', 'shortest'])
    dx['corresponding shortest'] = dz.idxmin(axis=1)
    data_heuristic = []
    heuristic_array = []
    while j1 < len(dx['garbage volume']):
        data_heuristic = []
        while i1 < len(dx['garbage volume']):
            data_heuristic.append(int(math.exp((dx['garbage volume'][j1] + dx['garbage volume'][i1]) / 2)))
            i1 = i1 + 1

        heuristic_array.append(data_heuristic)
        i1 = 0
        j1 = j1 + 1

    arr1 = np.array(heuristic_array)
    arr2 = ((1 / arr) + arr1)
    arr3 = np.nan_to_num(arr2)
    arr2[arr2 == inf] = 0

    dx = dx.drop(columns=['a', 'b', 'c', 'd', 'e'])
    j2 = 0
    i2 = 0
    data_source = []
    data_target = []
    data_position = []
    while i2 < len(dx['columnID']):
        while j2 < len(dx['columnID']):
            data_source.append(dx['columnID'][i2])
            data_target.append(dx['columnID'][j2])
            data_position.append(df['location'][j2])
            j2 = j2 + 1
        i2 = i2 + 1
        j2 = 0

    final_graph = {'source': data_source, 'target': data_target, 'pos': data_position,
                'euclidian distance': [i for i in np.nditer(arr)]
        , 'heuristic': [j for j in np.nditer(arr2)]}

    df_with_heuristics = pandas.DataFrame(data=final_graph)

    # build graph
    G = nx.from_pandas_edgelist(df_with_heuristics, 'source', 'target', ['pos', 'euclidian distance', 'heuristic'])

    i3 = 0
    while i3 < len(df['nodeID']):
        G.add_node(df['nodeID'][i3], pos=df['location'][i3])
        i3 = i3 + 1

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='text', mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Reds',
            reversescale=False,
            color=[],
            size=12,
            colorbar=dict(
                thickness=15,
                title='Garbage Volume',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_volume = []
    node_text = []
    i5 = 0
    for node in df['nodeID']:
        node_volume.append(df['garbage_volume'][i5])
        node_text.append(str(df['nodeID'][i5]) + '; Garbage volume is : ' + str(df['garbage_volume'][i5]) + '\n'
                        + 'Next optimal node is : '
                        + str((df_with_heuristics.iloc[0+5*i5:5+5*i5,][df_with_heuristics.iloc[0+5*i5:5+5*i5,]['heuristic']
                                                                    == max(df_with_heuristics.iloc[0+5*i5:5+5*i5, -1])]['target']).values.tolist()[0])
                        )
        i5 = i5 + 1


    node_trace.marker.color = node_volume
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        # annotations=[dict(
                        #     showarrow=False,
                        #     xref="paper", yref="paper",
                        #     x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=True))
                    )

    # Add images

    fig.add_layout_image(
        dict(
            source=img,
            xref="x",
            yref="y",
            x=0,
            y=5,
            sizex=6,
            sizey=5,
            # sizing="stretch",
            opacity=1,
            layer="below")
    )

    # # Set templates
    fig.update_layout( height=600, width=500,
                    )


    #Render components
    graph_div = plot(fig, output_type='div')

    return render(request, 'graph.html', {'graph_div': graph_div})

      