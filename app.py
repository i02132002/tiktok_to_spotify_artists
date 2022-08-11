import streamlit as st
from chartanalyzer import ChartAnalyzer
import plotly.express as px
from datetime import datetime, timedelta


def plot_matched_artists():
    return plot_trajectories(ca.matched, suffix='spotify')


def plot_nonmatched_artists():
    return plot_trajectories(ca.nonmatched, suffix='spotify')


def plot_trajectories(df, suffix='spotify'):
    if suffix:
        suffix = '_' + suffix
    fig = px.line(df, x="Chart Date" + suffix, y="Rank" + suffix, color='Artist', symbol="Artist",
                  hover_data=["Monthly Listeners Change",
                              "Release to Chart Time", "Track"],
                  labels={
                      "Chart Date" + suffix: "Date",
                      "Rank": "Rank",
                      "Artist": "Artist",
                      "Monthly Listeners Change": "Monthly Listeners",
                      "Release to Chart Time": "Release to Chart Time",
                      "Track": "Track"
                  })
    fig.update_traces(marker={'size': 8})
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[datetime.strptime(
        '2021-08-01', '%Y-%m-%d'), datetime.strptime('2022-08-15', '%Y-%m-%d')])
    fig.update_layout(dragmode='pan', width=700)
    return fig


ca = ChartAnalyzer()
st.title('Which Tiktokkers will Make it to Spotify?')
st.text("""To see which artists had their big breaks from Tiktok, I selected artists who charted on both Spotify and Tiktok.""")
st.image('tiktok_spotify.png')
print(ca.rising_tiktok_artists.Artist)
st.text("""Here is the list of 103 Tiktokkers who made it big on Spotify and their chart rank in time.""")
fig = plot_matched_artists()
st.plotly_chart(fig)
st.text("""Here you can see how any other artist compares in ranking in the same time.""")
fig = plot_nonmatched_artists()
st.plotly_chart(fig)
