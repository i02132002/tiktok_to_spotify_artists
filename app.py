import streamlit as st
from chartanalyzer import ChartAnalyzer
import plotly.express as px
from datetime import datetime, timedelta


@st.cache
def plot_matched_artists():
    return plot_trajectories(ca.matched, suffix='spotify')


@st.cache
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
                      "Release to Chart Time": "Time from Release to TikTok",
                      "Track": "Track"
                  })
    fig.update_traces(marker={'size': 8})
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[datetime.strptime(
        '2021-08-01', '%Y-%m-%d'), datetime.strptime('2022-08-15', '%Y-%m-%d')])
    fig.update_layout(dragmode='pan', width=700)
    return fig


ca = ChartAnalyzer()
st.title('Which Tiktokkers Made it to Spotify?')
st.write("""TikTok is an emerging source for discovering young musical talent, 
            but not every popular track on TikTok makes it into Spotify playlists.
            Which TikTokers are making it big on Spotify?""")
st.image('tiktok_spotify.png')
st.write("""To see which artists got their big breaks from TikTok, I found artists who both charted on 
            TikTok and Spotify. I then filtered out \"established artists\" by excluding artists who have 
            charted on Spotify first before TikTok, artists who have previously charted on Spotify, and artists
            whose top tracks were released 400 days or more ago before charting on TikTok. After filtration,
            159 artists were left matching these criteria. Some googling on these 159 artists showed
            that 103 of them became famous by releasing their songs on TikTok. These 103 TikTokers can be
            used as examples to build a model for predicting Spotify success. Data used here came from
            Chartmetric's weekly top TikTok tracks and weekly Spotify top Artists by monthly plays from
            Apr. 18, 2021 to Aug. 4, 2022.
        """)

st.subheader('Spotify Trajectories of Rising TikTok Artists')
st.write("""Shown here are Spotify ranking trajectories of the 103 Tiktokkers after their careers
            took off on Spotify. Most trajectories show a ballistic rise and fall. Drag to scroll.""")
fig = plot_matched_artists()
st.plotly_chart(fig)

st.subheader('Spotify Trajectories of Established Artists')
st.write("""Shown here are Spotify ranking trajectories of 56 artists who already had established
            careers before charting on TikTok, but were not eliminated by the filter.
            Most trajectories show a downward trend, possibly indicating a well-established fan base
            who follow the artist's newly released songs. Drag to scroll.""")
fig = plot_nonmatched_artists()
st.plotly_chart(fig)

st.subheader('Spotify Trajectories of All Artists')
st.write("""Here you can visualize the Spotify ranking trajectory of some of your favorite artists
            whose songs are also popular on TikTok, to see how they compare to rising TikTok artists.
            Drag to scroll.
""")

artists = ca.merged[~ca.merged.Artist.isin(ca.rising_tiktok_artists.Artist)]
artists = artists.Artist.sort_values().unique().tolist()
selection = st.selectbox('Select artist:', artists,
                         index=artists.index('ABBA'))
artist = ca.merged[ca.merged.Artist == selection]
fig = plot_trajectories(artist)
st.plotly_chart(fig)
st.write("""Made by Franklin Liou (i02132002b@gmail.com), Aug. 11, 2022.""")
