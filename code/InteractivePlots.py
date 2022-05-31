import pickle

import altair as alt
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, Dash
from dash import html
from dash import Dash, dcc, html, Input, Output
a_file = open(r"C:\Users\dell\Desktop\Babylab\python\Babygym python\pythonProcessingScript\sensor.pkl", "rb")
b_file = open(r"C:\Users\dell\Desktop\Babylab\python\Babygym python\pythonProcessingScript\coded_raw.pkl", "rb")
c_file = open(r"C:\Users\dell\Desktop\Babylab\python\Babygym python\pythonProcessingScript\coded_res.pkl", "rb")
d_file = open(r"C:\Users\dell\Desktop\Babylab\python\Babygym python\pythonProcessingScript\coded_res_1d.pkl", "rb")
h_file = open(r"C:\Users\dell\Desktop\Babylab\python\Babygym python\pythonProcessingScript\diff.pkl", "rb")
r_file = open(r"C:\Users\dell\Desktop\Babylab\python\Babygym python\pythonProcessingScript\all_ann.pkl", "rb")
sen = pickle.load(a_file)
cod = pickle.load(b_file)
cod_res = pickle.load(c_file)
cod_1d = pickle.load(d_file)
diff = pickle.load(h_file)
all_ann = pickle.load(r_file)

sensors_wide, coding, coding_res, sensors_long, coding_1d = {}, {}, {}, {}, {}
diff_df = {}
ann = {}

for i in cod:
    if i in sen:
        sensors_wide[i] = sen[i]
        coding[i] = cod[i]
        coding_res[i] = cod_res[i]
        #coding_1d[i] = cod_1d[i]
        ann[i] = all_ann[i]
for i in diff:
    diff_df[i] = diff[i]

ann_df = pd.concat(ann.values())

diffs = pd.DataFrame.from_dict(data=diff_df, columns = ['Difference in milliseconds between sensors and video'], orient='index')
diffs.to_excel(r"C:\Users\dell\Desktop\Babylab\python\Babygym python\pythonProcessingScript\difference.xlsx")
ann_df.to_excel(r"C:\Users\dell\Desktop\Babylab\python\Babygym python\pythonProcessingScript\all_ann.xlsx")
# for id in coding_1d:
#     for label in coding_1d[id]:
#         coding_1d[id][label] = coding_1d[id][label].transpose()
#     coding_1d[id] = pd.concat(coding_1d[id].values()).transpose()
#
#     coding_1d[id]['id'] = id
#     coding_1d[id]['idx'] = coding_1d[id].index
#     coding_1d[id] = pd.melt(coding_1d[id], id_vars=['id', 'idx'], var_name='label')

#coding_raw_1d = pd.concat(coding_1d.values())

for id in coding:
    coding[id] = coding[id].transpose()
    coding[id]['id'] = id
    coding[id]['idx'] = coding[id].index
    coding[id] = pd.melt(coding[id], id_vars=['id', 'idx'], var_name='label')

coding_raw_df = pd.concat(coding.values())

for id in coding_res:
    coding_res[id]['id'] = id
    coding_res[id]['idx'] = coding_res[id].index
    coding_res[id] = pd.melt(coding_res[id], id_vars=['id', 'idx'], var_name='label')

coding_res_df = pd.concat(coding_res.values())


for id in sensors_wide:
    sensors_wide[id] = pd.DataFrame(sensors_wide[id])
    sensors_wide[id]['id'] = id
    sensors_wide[id]['idx'] = sensors_wide[id].index
    sensors_long[id] = pd.melt(sensors_wide[id], id_vars=['id', 'idx'], var_name='limb')

sensors_wide_df = pd.concat(sensors_wide.values())
sensors_long_df = pd.concat(sensors_long.values())

coding_raw_df.to_csv(r"C:\Users\dell\Desktop\Babylab\python\Babygym python\pythonProcessingScript\coding_raw.csv")
#coding_raw_1d.to_csv(r"C:\Users\dell\Desktop\Babylab\python\Babygym python\pythonProcessingScript\coding_res_1d.csv")
coding_res_df.to_csv(r"C:\Users\dell\Desktop\Babylab\python\Babygym python\pythonProcessingScript\coding_res.csv")
sensors_wide_df.to_csv(r"C:\Users\dell\Desktop\Babylab\python\Babygym python\pythonProcessingScript\sensors_wide.csv")
sensors_long_df.to_csv(r"C:\Users\dell\Desktop\Babylab\python\Babygym python\pythonProcessingScript\sensors_long.csv")

def InteractivePlots():
    # We have a list for every day
    # In your case will be gropuby('RuleName')
    # here for every element d
    # d[0] is the name(key) and d[1] is the dataframe
    dfs = list(sensors_df.groupby("id"))

    first_title = dfs[0][0]
    traces = []
    buttons = []

    for i, d in enumerate(dfs):
        app = Dash(__name__)
        continents = list(d[1]['limb'].unique())
        visible = [False] * len(dfs)
        visible[i] = True
        name = d[0]
        df = d[1]
        # r = list(d[1]['InfantRightArm'].to_numpy())
        # l = list(d[1]['InfantLeftArm'].to_numpy())
        #print(d[1][d[1]['idx'] == 500])
        traces.append(
                #px.line(d[1],x='idx', y='value', color='sensor').update_traces(visible=True if i == 0 else False).data[0]
                px.line(d[1], x='idx', y='value').update_traces(visible=True if i == 0 else False).data[0]
        )

        buttons.append(dict(label=name,
                            method="update",
                            args=[{"visible": visible},
                                  {"title": f"{name}"}]))

    updatemenus = [{'active': 0, "buttons": buttons}]

    fig = go.Figure(data=traces,
                    layout=dict(updatemenus=updatemenus))
    fig.update_layout(title=first_title, title_x=0.5)
    fig.show()

# def InteractivePlots2():
#     from dash import Dash, dcc, html, Input, Output
#     from plotly.express import data
#     import pandas as pd
#
#     df = sensors_df[sensors_df['id'] == '77031_1']
#     print(df)
#     app = Dash(__name__)
#     app.layout = html.Div([
#         dcc.Checklist(options=list(df['limb'].unique()), value=['InfantRightArm'], id='pandas-dropdown-2'),
#         html.Div(children='Hello Dash',id='pandas-output-container-2')
#     ])
#
#     @app.callback(
#         Output('pandas-output-container-2', 'children'),
#         Input('pandas-dropdown-2', 'value')
#     )
#     def update_output(value):
#         fig1 = px.line(df[df['limb'] == value[0]],
#                        x='idx', y='value', color='limb')
#         return fig1
#
#     if __name__ == '__main__':
#         app.run_server(debug=True)
# #

a_file.close()
b_file.close()