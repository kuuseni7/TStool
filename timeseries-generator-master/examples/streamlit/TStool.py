import sys
from contextlib import contextmanager
from io import StringIO

import numpy as np
import pandas as pd
import datetime

import streamlit as st
from threading import current_thread
import altair as alt
import functools
import base64

from timeseries_generator import (
    Generator,
    RandomFeatureFactor,
    TemperatureFactor


)

sys.path.append("../..")

np.random.seed(42)

st.set_page_config(
    page_title="TStool", layout="wide", initial_sidebar_state="auto"
)


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield


# -----------------------------------
# Streamlit APP start from here

st.title("Iot Time Series Syntheic Data Generator")

st.sidebar.subheader("Select sensor")

sensor_dict = {}
sensor_flag = st.sidebar.multiselect("Select sensor", ["temperature", "level"])
# sensor_dict["sensor"] = sensors


# st.sidebar.subheader("Input a base value")
# base_amount = st.sidebar.number_input("", value=0, min_value=-50, max_value=50, format="%d")

if sensor_flag:

    if len(sensor_flag) > 0:

        for sensor in sensor_flag:

            if sensor == "temperature":

                temperature_dict = {}
                feature_raw_str = "Temperature"

                feature_list = feature_raw_str.split(",")

                for feat in feature_list:
                    default_val_l = [f"{feat}_{i}" for i in range(2)]

                    feat_val_l = st.sidebar.text_input(
                        f"Input amount of sensor(s) [{feat}] (must separate by comma)",
                        ",".join(default_val_l),
                    )
                    sensor_dict[feat] = feat_val_l.split(",")

            if sensor == "level":

                temperature_dict = {}
                feature_raw_str = "Level"

                feature_list = feature_raw_str.split(",")

                for feat in feature_list:
                    default_val_l = [f"{feat}_{i}" for i in range(2)]

                    feat_val_l = st.sidebar.text_input(
                        f"Input amount of sensor(s) [{feat}] (must separate by comma)",
                        ",".join(default_val_l),
                    )
                    sensor_dict[feat] = feat_val_l.split(",")

value_list = []


# -------------------------
# add feature related factors

st.sidebar.subheader("Select type for sensor")

type_factor_dict = {
    "random_factor": RandomFeatureFactor,
    "temperature_factor": TemperatureFactor,
    #"humidity_factor": HumidityFeatureFactor,
    #"pressure_factor": PressureFeatureFactor,
    #"proximity_factor": ProximityFeatureFactor,
    #"level_factor": LevelFeatureFactor
}

type_switch_dict = {}
for feat in sensor_dict.keys():

    sensor_switch = st.sidebar.checkbox(f"{feat}", key=f"sensor_switch_{feat}")

    if sensor_switch:

        sensor_options = st.sidebar.multiselect(
            f"select options for [{feat}]",
            ("Name","Range","Anomalies")
        )

        if len(sensor_options) > 0:

            for options in sensor_options:

                if options == "Name":

                    feature_raw_str = st.sidebar.text_input("Input sensor name")

                if options == "Range":

                    temperature_dict = {}

                    for feat_val in sensor_dict[feat]:
                        minTmp = st.sidebar.slider(
                            f"Min temperature {feat_val}",
                            value=0,
                            format="%f",
                            key="temperature_value_{feat}_min_{feat_val}",
                            min_value=-50,
                            max_value=50
                        )

                        maxTmp = st.sidebar.slider(
                            f"Max temperature{feat_val}",
                            value=0,
                            format="%f",
                            key="temperature_factor_{feat}_max_{feat_val}",
                            min_value=-50,
                            max_value=50
                        )

                        temperature_dict[feat_val] = {
                            "min_temperature_value": minTmp,
                            "max_temperature_value": maxTmp,

                        }
                    value_list.append(
                        TemperatureFactor(
                            feature=feat,
                            feature_values=sensor_dict[feat],
                            col_name=f"temperature_{feat}",
                            min_temperature_value=minTmp,
                            max_temperature_value=maxTmp
                        )
                    )
                if options == "Anomalies":
                    anomalies_l = []
                    st.sidebar.slider("Select amount of anomalies")
                    st.sidebar.slider("Select frequency of anomalies")
                    st.sidebar.text_input("Select type of anomaly")
                    st.sidebar.number_input("Set time of anomaly")
                    #posibilyt for specific datetime?




# add global factors



# ---------------------------
# select time period

st.subheader("Input start date and end date")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Start time", datetime.datetime(2022, 1, 1, 1, 1, 1)
    )
with col2:
    end_date = st.date_input(
        "End time", datetime.datetime(2022, 1, 1, 1, 1, 1)
    )

# generate time series
g: Generator = Generator(
    factors=set(value_list),
    features=sensor_dict,
    date_range=pd.date_range(start_date, end_date)
)
df_sensor = g.generate()


# ------------------------------------------------
# visualization

st.subheader("Generated time series data")

# get all sensors in sensor_dict
all_sensors = list(sensor_dict.keys())

vis_sensor_l = st.multiselect("Choose features to aggregate", all_sensors, all_sensors)
if len(vis_sensor_l) > 0:
    group_feat_l = vis_sensor_l.copy()
    group_feat_l.insert(0, "date")
    df_vis = df_sensor.groupby(group_feat_l)["value"].sum().reset_index()
else:
    df_vis = df_sensor.copy()


df_plot = df_vis[["date", "value"]]

if len(vis_sensor_l) > 0:
    color_col = "-".join(vis_sensor_l)
    df_plot[color_col] = functools.reduce(
        lambda x, y: x + "-" + y, (df_vis[feat] for feat in vis_sensor_l)
    )

    base = (
        alt.Chart(df_plot)
        .mark_line()
        .encode(x="date:T", y="value:Q", color=f"{color_col}:N")
    )

    selection = alt.selection_multi(fields=[color_col], bind="legend")

    chart = (
        base.mark_line()
        .encode(opacity=alt.condition(selection, alt.value(1), alt.value(0.2)))
        .add_selection(selection)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)
else:

    base = (
        alt.Chart(df_plot)
        .mark_line()
        .encode(
            x="date:T",
            y="value:Q",
        )
        .interactive()
    )

    st.altair_chart(base, use_container_width=True)


# --------------
# download dataframe
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href


st.markdown(get_table_download_link(df_vis), unsafe_allow_html=True)


# -------------
# show dataframe

col1, col2 = st.columns(2)
with col1:
    show_base_df = st.checkbox("Show dataframe")
with col2:
    topn = st.number_input("Top N rows", value=50, format="%d")
if show_base_df:
    show_col = ["date"] + vis_sensor_l + ["value"]
    st.dataframe(df_vis[show_col].head(topn))
