from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import glob
import re
from datetime import datetime, timedelta
import plotly.express as px
pd.set_option('display.max_columns', None)


class ChartAnalyzer:        
        
    def __init__(self, read_from_csv = True):
        if read_from_csv:
            self.matched = pd.read_csv('matched.csv')
            self.nonmatched = pd.read_csv('nonmatched.csv')
            self.merged = pd.read_csv('merged.csv')
            self.rising_tiktok_artists = pd.read_csv('rising_artists_filtered.csv')
        else:
            self.INITIAL_DATE = '2021-04-18'
            self.spotify = self.load_chart('spotify')
            self.tiktok = self.load_chart('tiktok')
            self._calculate_release_to_chart()
            self.first_tiktok = self.get_first_tiktok()
            self.merged = self.merge_spotify_tiktok()
            self.prespotify = self.get_prespotify_data(self.merged)
            self.rising_tiktok_artists, self.matched, self.nonmatched, self.matched_prespotify, self.nonmatched_prespotify = self.get_rising_tiktok_artists()
            self.collapsed = self.collapse_tiktok_rank()
            self.X, self.y = self.make_X()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)
    
    def get_first_tiktok(self):
        first = self.tiktok[self.tiktok['Chart Date'] == self.tiktok.groupby('Artist')['Chart Date'].transform(min)]
        first_only = first[first['Rank'] == first.groupby('Artist')['Rank'].transform(min)]
        first_only = first_only[first_only.groupby('Artist').cumcount() == 0]
        return first_only
    
    def _calculate_release_to_chart(self):
        min_chart_date = self.tiktok.groupby('Artist')['Chart Date'].transform(min)
        self.tiktok['Release to Chart Time'] = ((min_chart_date - self.tiktok['Release Date'])
                                                .astype('timedelta64[D]').astype(int))
    
    def plot_matched_artists(self):
        self.plot_trajectories(self.matched, suffix= 'spotify')
    
    def plot_nonmatched_artists(self):
        self.plot_trajectories(self.nonmatched, suffix= 'spotify')
    
    def plot_trajectories(self, df, suffix = 'spotify'):
        if suffix:
            suffix = '_' + suffix
        fig = px.line(df, x="Chart Date" + suffix, y="Rank" + suffix, color='Artist', symbol="Artist",
         hover_data=["Monthly Listeners Change", "Release to Chart Time","Track"],
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
        fig.update_xaxes(range=[datetime.strptime('2021-08-01','%Y-%m-%d'), datetime.strptime('2022-08-15','%Y-%m-%d')])
        fig.update_layout(dragmode='pan')
        fig.show()

    def collapse_tiktok_rank(self):
        idx = self.prespotify.groupby(['Artist', 'Chart Date_tiktok'])['Rank_tiktok'].transform(min) == self.prespotify['Rank_tiktok']
        collapsed = self.prespotify[idx]
        return collapsed[collapsed.groupby('Artist')['Rank_spotify'].transform('count') > 1]
        
    def make_X(self):        
        numeric_columns = ['Rank','Video Count', 'All-Time Video Count','All Time Rank','7-Day Velocity'] #,'All-Time Video Count','All Time Rank',,'7-Day Velocity','Peak Position_tiktok','All-Time Video Count','All Time Rank', 'Change_tiktok','Peak Date_tiktok','Release Date', 
        
        selector = ColumnTransformer([
            ('numeric', 'passthrough', numeric_columns),
        ])
        
        pipeline = Pipeline([
            ('selector', selector),
            ('scaler', StandardScaler())
        ])
        
        tss = []
        y_train = []
        X = pipeline.fit_transform(self.first_tiktok)
        y = self.first_tiktok.Artist.isin(self.rising_tiktok_artists.Artist).astype(int)
        
        return X, y
        
    
    def merge_spotify_tiktok(self):
        merged = (self.spotify.merge(self.tiktok, on=['Artist'], how='inner', suffixes = ('_spotify', '_tiktok'))
                       .sort_values(by=['Chart Date_spotify', 'Chart Date_tiktok', 'Artist']))
        merged = merged[merged['Chart Date_spotify'] > merged['Chart Date_tiktok']]
        return merged
    
    def get_prespotify_data(self, df):
        mask = (df.groupby('Artist')['Chart Date_spotify']
            .transform(lambda x: x == x.min())
            .astype(bool)) # Gets all rows before the artist's first spotify appearance
        return df[mask]
         
    def _filter_initial_artists(self, df):
        initial_artists = df[df['Chart Date'] == self.INITIAL_DATE].Artist.unique()
        return df[~df.Artist.isin(initial_artists)]
        
    def filter_out_famous_artists(self, df):
        return df[df['Total Weeks on Chart'] == df['Current Weekly Streak']]
    
    def get_rising_tiktok_artists(self, save_csv = False, load_csv = True):
        filtered = self.filter_out_famous_artists(self.merged)
        filtered = filtered[filtered['Release to Chart Time']<=400]
        
        matched = filtered[filtered['Chart Date_spotify'] > filtered['Chart Date_tiktok']]
        idx = matched.groupby('Chart Date_spotify')['Chart Date_tiktok'].transform(max) == matched['Chart Date_tiktok']
        matched = matched[idx]
        #idx = matched.groupby('Track')['Rank_tiktok'].transform(min) == matched['Rank_tiktok']
        #matched = matched[idx]
        
        artist_list = pd.DataFrame({"Artist": matched.Artist.unique()})
        if save_csv:
            artist_list.to_csv('rising_artists_filtered.csv')
        
        if load_csv:
            # Override rising_tiktok_artists with manually filtered artists
            artist_list = pd.read_csv('rising_artists_filtered.csv')
            nonmatched_artist_list = artist_list[artist_list.New == 0]
            artist_list = artist_list[artist_list.New == 1]
            nonmatched = matched[matched.Artist.isin(nonmatched_artist_list.Artist)] # manual selection of artists
            matched = matched[matched.Artist.isin(artist_list.Artist)] # manual selection of artists
            nonmatched_prespotify = self.get_prespotify_data(nonmatched)
        else:
            nonmatched = None
            nonmatched_prespotify = None
        matched_prespotify = self.get_prespotify_data(matched)
        return artist_list, matched, nonmatched, matched_prespotify, nonmatched_prespotify
        
    def load_chart(self, chartname):
        path = r'data/' + chartname + '/*.csv'
        dfs = []
        # Load and concat all the data
        for filename in glob.glob(path):
            df = pd.read_csv(filename)
            df['Chart Date'] = self._get_date_from_filename(filename)
            dfs.append(df)
        chart = pd.concat(dfs)
        
        if chartname == 'spotify':
            chart = self._process_spotify_df(chart)
        elif chartname == 'tiktok':
            chart = self._process_tiktok_df(chart)
        chart = self._filter_initial_artists(chart)
        return chart
    
    def _get_date_from_filename(self, filename):
        date = re.search(r'_(\d{4}-\d{2}-\d{2})\.csv',filename)[1]
        date = datetime.strptime(date,'%Y-%m-%d')
        return date
    
    def _process_tiktok_df(self, df):
        df['Artists'] = df['Artists'].apply(lambda x: [s.strip() for s in x.split(',')])
        df = (df.explode('Artists')
                .rename(columns={"Artists": "Artist"})
                .drop_duplicates())
        df = self._numstring_2_num(df)
        df['Release Date'] = df['Release Date'].apply(lambda x : datetime.strptime(x,'%b %d, %Y'))
        df['Peak Date'] = df['Peak Date'].apply(lambda x : datetime.strptime(x,'%b %d, %Y'))
        df['7-Day Velocity'] = df['7-Day Velocity'].fillna(0)
        return df
    
    def _process_spotify_df(self, df):
        df['Peak Date'] = df['Peak Date'].apply(lambda x : datetime.strptime(x,'%b %d, %Y'))
        return df
    
    def _numstring_2_num(self, df):
        df = df.copy()
        df['Video Count'] = df['Video Count'].replace({"K":"*1e3", "M":"*1e6"}, regex=True).map(pd.eval).astype(int)
        df['All-Time Video Count'] = df['All-Time Video Count'].replace({"K":"*1e3", "M":"*1e6"}, regex=True).map(pd.eval).astype(int)
        return df
