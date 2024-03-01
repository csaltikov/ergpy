from dash import Dash, html, dcc, Output, Input, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Generates a random work for a trainer based on
# two sliders: one with a range of power and the other
# for setting a time range for each interval
# There is a 10 warm zone 1 warm at the start


app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])

# Different workout times
work_out_time_options = [{'label': f"{i} min", 'value': i} for i in [*range(30, 105, 15)]]

# Text used to generate the ERG files upon export
pre = """[COURSE HEADER]
VERSION = 2
UNITS = ENGLISH
DESCRIPTION =
FILE NAME = test.erg
FTP = 270
MINUTES WATTS
[END COURSE HEADER]
[COURSE DATA]\n"""

post = """[END COURSE DATA]
[COURSE TEXT]
[END COURSE TEXT]"""


def tss_calculator(minutes, watt, ftp):
    sec = minutes * 60
    intensity_factor = watt / ftp
    training_stress_score = (sec * watt * intensity_factor) / (ftp * 3600) * 100
    return round(training_stress_score)


def workout_maker(time, watts, total_time, ftp):
    """Random workout generator

    :param time: the range of times (minutes) for the intervals
    :param watts: range in watts
    :param total_time: how long the workout will be in minutes
    :param ftp: the ftp setting
    :return: a pandas dataframe of the workout
    """
    # set a counter for the first data point
    x = 0
    # keep track of the total time
    time_sum = 0
    # make a list of the time data points
    time_list = []
    # make a list of the watts data points
    watt_list = []
    # list of the percent ftp data points
    ftp_percent = []
    # TSS holder
    tss = 0

    # interval time/watts lists used to make the erg profile
    interval_times = list(np.random.randint(time[0], time[1], size=total_time))
    interval_watts = list(np.random.randint(watts[0], watts[1], size=total_time))

    # loop through the incoming time vs watts data from the fit file
    for t, w in zip(interval_times, interval_watts):
        # the first data needs to start with zero
        if x == 0:
            # warm up profile
            times = [0, 5, 5, 8, 8, 10]
            time_list.extend(times)
            watts = [120, 120, 140, 140, 160, 160]
            tss += tss_calculator(t, w, ftp)
            watt_list.extend(watts)
            ftp_percent.extend([x / ftp for x in watts])
            time_old = 0
            time_new = times[-1]
            time_sum = time_old + time_new
            x += 1
        if x > 0:
            # main workout
            tss += tss_calculator(t, w, ftp)
            time_list.append(time_sum)
            watt_list.append(w)
            ftp_percent.append(w / ftp)
            time_sum = time_sum + t
            time_list.append(time_sum)
            watt_list.append(w)
            ftp_percent.append(w / ftp)
        if time_sum > total_time:
            # Cooldown Profile
            time_list.extend([time_sum, time_sum + 5])
            watt_list.extend([ftp * .5, ftp * .4])
            ftp_percent.extend([.5, .4])
            break
    return pd.DataFrame({"time": time_list, "watts": watt_list, "ftp_percent": ftp_percent}), tss


def make_erg(profile, time):
    """Export the workout in ERG format"""
    global ergfile_name
    ergfile_name = f"erg_dash_workout_{time}.erg"
    ergfile = Path.joinpath(Path("static/"), Path(ergfile_name))
    with open(ergfile, "w") as file:
        for line in pre.splitlines():
            if line.rstrip().startswith("DESCRIPTION"):
                file.write(f"DESCRIPTION = work time is {time}\n")
            elif line.rstrip().startswith("FILE"):
                file.write(f"FILE NAME = {ergfile.name}\n")
            else:
                file.write(f"{line}\n")
    profile.loc[:, ["time", "watts"]].to_csv(ergfile, mode="a", index=False, header=False, sep="\t")
    with open(ergfile, "a") as file:
        file.write(post)
    print(f"ERG file was exported as {ergfile}")


app.layout = html.Div(children=[
    html.H2(children=["ERG Maker Dash App"]),
    html.Div(children=[dcc.RangeSlider(140, 300, 5, value=[160, 190], id='interval_input', vertical=True)],
             style={'display': 'inline-block', "width": "10%"}),
    html.Div(children=[dcc.RangeSlider(2, 10, 1, value=[2, 5], id='time_input', vertical=True)],
             style={'display': 'inline-block', "width": "10%"}),
    html.Div(children=[dcc.Graph(id="plot", figure={})], style={'display': 'inline-block', "width": "80%"}),
    html.Div(children=["Watts:", dcc.Input(id="ftp_input", value=270, type="text",
                                           debounce=True,
                                           placeholder="Enter watts",
                                           style={'marginRight': '10px', 'marginLeft': '10px', "width": "150px"})]),
    # Work time pulldown list, increments of 15 min
    html.Div(children=["Time (min):",
                       dcc.Dropdown(options=work_out_time_options,
                                    value='60', id='workout_time',
                                    style={"width": "50%", 'marginLeft': '10px', 'display': 'inline-block'})],
             style={"width": "50%", 'marginRight': '10px'}),
    html.Br(),
    html.Div(id='interval_output'),
    html.Div(id='ftp_output'),
    html.Div(id="tss"),
    html.Div(id="warning"),
    html.Button('Download ERG File', id='btn', n_clicks=0),
    dcc.Download(id="download"),
    # dcc.Store stores the intermediate value
    dcc.Store(id='meta_data')
], style={"height": "100%",
          "border": "1px solid grey",
          "padding": "5px",
          "margin": "30px"})


@app.callback(
    [Output('interval_output', 'children'),
     Output('ftp_output', 'children'),
     Output('plot', 'figure'),
     Output('tss', 'children'),
     Output('meta_data', 'data')],
    [Input('interval_input', 'value'),
     Input('ftp_input', 'value'),
     Input('time_input', 'value'),
     Input('workout_time', 'value'),
     Input("btn", "n_clicks")]
)
def update_output(interval_input, ftp_input, time_input, workout_time, n_clicks):
    work_out_time = int(workout_time) - 5
    ftp_out = f'FTP is : {ftp_input}'
    watt_low, watt_high = interval_input
    time_low, time_high = time_input
    watt_text = f"Watt range: {watt_low} to {watt_high}"
    ftp = int(ftp_input)
    wattinterval = [watt_low, watt_high]
    timeinterval = [time_low, time_high]

    # make the erg profile
    profile, tss = workout_maker(time=timeinterval, watts=wattinterval, total_time=work_out_time, ftp=ftp)
    max_time = profile["time"].max()

    # create the erg file name based on the workout time
    meta_data_dict = {"ergfile": f"erg_dash_workout_{workout_time}.erg",
                      "profile": profile.to_dict()}
    meta_data_json = json.dumps(meta_data_dict)

    # plot the profile
    fig = px.line(profile, x=profile["time"], y=profile["watts"])
    fig.update_yaxes(range=[0, 400])
    fig.add_hline(y=ftp, line_dash="dash", line_color="black", line_width=0.5)
    power_zones = dict(zone1=[0.55, "LightBlue"],
                       zone2=[0.75, "LightGreen"],
                       zone3=[0.9, "Orange"],
                       zone4=[1.06, "Red"],
                       zone5=[1.2, "Purple"])
    fig.add_shape(type="rect", x0=0, y0=0, x1=max_time, y1=ftp * power_zones['zone1'][0],
                  line=dict(color="Grey"), fillcolor=power_zones['zone1'][1])
    old_power = 0.55
    for zone, percent in power_zones.items():
        fig.add_shape(name=zone, type="rect", x0=0, y0=ftp * old_power, x1=max_time, y1=ftp * percent[0],
                      line=dict(color="Grey"), fillcolor=percent[1], opacity=0.1)
        old_power = percent[0]
    fig.update_layout(template="simple_white")
    # make_erg(profile, workout_time)
    tss_out = f'TSS is: {tss}'
    if ctx.triggered_id != "btn":
        return watt_text, ftp_out, fig, tss_out, meta_data_json
    else:
        raise PreventUpdate


@app.callback(Output("download", "data"),
              [Input("btn", "n_clicks"),
               Input("meta_data", "data"),
               Input("workout_time", "value")])
def generate_csv(n_nlicks, meta_data, workout_time):
    meta_data_json = json.loads(meta_data)
    ergfile_name = meta_data_json["ergfile"]
    profile_dict = meta_data_json["profile"]
    profile_df = pd.DataFrame(profile_dict)
    make_erg(profile_df, workout_time)
    if ctx.triggered_id == "btn":
        return dcc.send_file(path=f"static/{ergfile_name}",
                             filename=ergfile_name,
                             type='text/csv')
    else:
        raise PreventUpdate


if __name__ == '__main__':
    # Get the current directory of the Python script
    current_directory = Path(__file__).resolve().parent

    # Define the directory name
    directory_name = "static"

    # Create a Path object for the directory
    directory_path = current_directory / directory_name

    # Check if the directory exists
    if not directory_path.is_dir():
        # Create the directory if it doesn't exist
        directory_path.mkdir()

    app.run_server(debug=True, port=8056)
