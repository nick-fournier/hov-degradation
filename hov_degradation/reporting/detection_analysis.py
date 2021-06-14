from plotnine import *
import os
import pandas as pd
import datetime
import numpy as np
from pandas.api.types import CategoricalDtype

class DetectionsPlot:
    def __init__(self, inpath, outpath):
        self.inpath, self.outpath = inpath, outpath

        # Defined colors
        self.colors = {'Data unavailable': '#bababa', 'Not misconfigured': '#7fc97f',
                       'Possible misconfiguration\n(Supervised only)': '#beaed4',
                       'Possible misconfiguration\n(Supervised and Unsupervised)': '#fdc086',
                       'Possible misconfiguration\n(Unsupervised only)': '#386cb0'}

        # Load data in
        self.df = self.load_data()


        # # Generate plots
        # self.freq_plot()
        # self.date_matrix_plot()
        #
        # # Frequency table by date
        # self.date_count()

    def load_data(self):
        dirlist = sorted(set(os.listdir(self.outpath)).intersection(os.listdir(self.inpath)))

        # Get latest meta data (this ensures latest sensor data is used)
        metafile = '/'.join(
            [self.inpath, dirlist[-1], [x for x in os.listdir(self.inpath + '/' + dirlist[-1]) if 'meta' in x][0]])
        meta = pd.read_csv(metafile, delimiter='\t')
        meta = meta.loc[meta.Type == 'HV']
        meta[['supervised', 'unsupervised', 'available']] = False  # Adding dummy columns

        # Load the detections
        df = pd.DataFrame()
        for dir in dirlist:
            # Load current predictions
            file = '/'.join([self.outpath, dir, [x for x in os.listdir(self.outpath + dir) if 'predictions' in x][0]])
            new_df = pd.read_csv(file)
            new_df = new_df[['Unnamed: 0', 'preds_classification', 'preds_unsupervised']]
            new_df = new_df.rename(columns={"Unnamed: 0": "ID", "preds_classification": "supervised",
                                            'preds_unsupervised': 'unsupervised'})
            new_df.supervised = new_df.supervised.astype(bool)
            new_df.unsupervised = new_df.unsupervised.astype(bool)

            # Load current meta data, add any missing sensors
            metafile = '/'.join([self.inpath, dir, [x for x in os.listdir(self.inpath + '/' + dir) if 'meta' in x][0]])
            new_meta = pd.read_csv(metafile, delimiter='\t')
            new_meta = new_meta.loc[new_meta.Type == 'HV']
            new_meta[['supervised', 'unsupervised', 'available']] = False  # Adding dummy columns
            meta = meta.append(new_meta.loc[new_meta.ID.isin(list(set(new_meta.ID).difference(meta.ID)))])

            # Add in meta data columns, this forms the available data for this run date
            meta_cols = [x for x in meta.columns if x not in ['supervised', 'unsupervised', 'available']]
            new_df['available'] = True
            new_df = new_df.merge(meta[meta_cols], on='ID')

            # Append the unavailable sensors
            new_df = new_df.append(meta.loc[meta.ID.isin(set(meta.ID).difference(new_df.ID))])
            new_df['run_id'] = dir

            # Add to master df
            df = df.append(new_df)

        df['Date'] = df.apply(lambda row: self.quarter_name(row.run_id), axis=1)
        df['Detection'] = df.apply(lambda row: self.detection_code(row), axis=1)
        df = df.sort_values('ID')

        return df

    def method_code(self, row):
        code = ''
        if row['supervised']:
            code += 'S'
        if row['unsupervised']:
            code += 'U'
        return code

    def detection_code(self, row):
        if not row.available:
            return 'Data unavailable'
        if row.supervised and row.unsupervised:
            return 'Possible misconfiguration\n(Supervised and Unsupervised)'
        if row.supervised and not row.unsupervised:
            return 'Possible misconfiguration\n(Supervised only)'
        if not row.supervised and row.unsupervised:
            return 'Possible misconfiguration\n(Unsupervised only)'
        if not row.supervised and not row.unsupervised and row.available:
            return 'Not misconfigured'

    def quarter_name(self, s):
        quarters = {'Q1': [1, 2, 3], 'Q2': [4, 5, 6], 'Q3': [7, 8, 9], 'Q4': [10, 11, 12]}
        year, month, days = s.split('_')
        days = '-'.join([str(int(x)) for x in days.split('-')])
        quarter = [x for x in quarters.keys() if int(month) in quarters[x]][0]

        datename = ''.join([datetime.datetime.strptime(month, "%m").strftime("%b"), '. ', days, ', ', year, ' (', quarter, ')'])
        # quartername = year + '-' + quarter
        return datename

    def recast_wide(self):
        df_wide = self.df.pivot(index='ID', columns='run_id', values='Detection')
        df_wide = df_wide.merge(right=self.df[['ID', 'User_ID_1']].drop_duplicates(), on='ID')
        df_wide.rename(columns={'User_ID_1': 'MS ID'}, inplace=True)

        # Get detection counts
        counts = self.df.groupby(['ID', 'Detection']).size().reset_index().rename(columns={0: 'Total detections'})
        counts = counts[~counts.Detection.isin(['Data unavailable', 'Not misconfigured'])].groupby('ID').sum()
        counts = counts.append(
            pd.DataFrame({'Total detections': 0},
                         index=[x for x in df_wide.ID.unique() if x not in counts.index.unique()])
        )
        counts.index.name = 'ID'

        # Merge the counts and sort
        df_wide = df_wide.merge(right=counts, how='left', on='ID').sort_values('Total detections', ascending=False)
        df_wide.to_excel(self.outpath + "/detection_matrix.xlsx", sheet_name='Sheet 1')

    def date_count(self):
        df_freq = self.df.groupby(['Date', 'Detection']).size().reset_index(name='count')
        df_freq = df_freq.pivot(index='Date', columns='Detection', values='count')
        df_freq = df_freq.merge(self.df.groupby(['Date']).size().reset_index(name='Total'), on='Date')
        df_freq['Sensors analyzed'] = df_freq['Total'] - df_freq['Data unavailable']
        df_freq['Total detections'] = df_freq[[x for x in df_freq.columns if 'Possible' in x]].apply(lambda row: row.sum(), axis=1)

        # Date row order
        date_cat = [self.quarter_name(x) for x in sorted(self.df.run_id.unique())]
        date_cat += ['Mean', 'Median', 'Standard Deviation']
        date_cat = CategoricalDtype(categories=date_cat, ordered=True)

        # Sort by date
        df_freq['Date'] = df_freq.Date.astype(date_cat)
        df_freq = df_freq.sort_values('Date')
        df_freq = df_freq.set_index('Date')

        cols = [x for x in df_freq.columns if x != 'Total']
        # 1. Calculate rate (%)
        for col in cols:
            df_freq[col] = df_freq[col].replace(np.nan, 0)
            if col in ['Sensors analyzed', 'Data unavailable']:
                df_freq[col + '_p'] = df_freq[col] / df_freq['Total']
            else:
                df_freq[col + '_p'] = df_freq[col] / df_freq['Sensors analyzed']

        # 2. Calculate stats for both
        stats = {'Mean': df_freq[df_freq.columns].mean(),
                 'Median': df_freq[df_freq.columns].median(),
                 'Standard Deviation': df_freq[df_freq.columns].std()}
        df_freq = df_freq.append(pd.DataFrame(stats).T)

        # 3. Format into string
        for col in cols:
            num = round(df_freq[col], 1).astype(str)
            perc = round(100 * df_freq[col + '_p'], 1).astype(str)
            df_freq[col + ' (rate)'] = num + " (" + perc + "%)"

        df_freq.to_excel(self.outpath + "/detection_summary.xlsx", sheet_name='Sheet 1')

    def freq_plot(self):

        detects = [x for x in self.colors.keys() if x not in ['Data unavailable', 'Not misconfigured']]
        sort_cols = ['detect_sum'] + detects

        # Get the frequencies
        df_freq = self.df.groupby(['ID', 'Detection']).size().reset_index(name='Count')
        # Pivot to wide and sort
        df_sort = df_freq.pivot(index='ID', columns='Detection', values='Count')
        df_sort['detect_sum'] = df_sort[detects].apply(lambda row: row.sum(skipna=True), axis=1)
        df_sort = df_sort.sort_values(sort_cols, ascending=[False]*len(sort_cols))
        _ids = df_sort.index

        self.full_freq = ggplot(data=df_freq[df_freq.ID.isin(_ids)],
                           mapping=aes(x='factor(ID)', y='Count', fill='Detection')) + \
                    geom_col() + \
                    scale_fill_manual(name='', values=self.colors) + \
                    scale_y_continuous(name='Frequency', breaks=range(0, df_freq.Count.max()), expand=(0, 0)) +\
                    scale_x_discrete(name='VDS ID', limits=_ids) +\
                    theme_bw() + theme(text=element_text(size=8),
                                       axis_ticks_length=0,
                                       axis_text_x=element_blank(),
                                       legend_background=element_blank(),
                                       legend_position='top')
        ggsave(plot=self.full_freq, filename=self.outpath + '/detection_frequency.png', height=4, width=12, dpi=300)


        # Truncated plot
        df_sort_cut = df_sort[df_sort.detect_sum > 2]
        df_sort_cut = df_sort_cut.sort_values(sort_cols, ascending=[False]*len(sort_cols))
        _ids_cut = df_sort_cut.index

        self.trunc_freq = ggplot(data=df_freq[df_freq.ID.isin(_ids_cut)],
                           mapping=aes(x='factor(ID)', y='Count', fill='Detection')) + \
                    geom_col() + \
                    scale_fill_manual(name='', values=self.colors) + \
                    scale_y_continuous(name='Frequency', breaks=range(0, df_freq.Count.max()), expand=(0, 0)) +\
                    scale_x_discrete(name='VDS ID', limits=_ids_cut) +\
                    theme_bw() + theme(text=element_text(size=8),
                                       axis_ticks_length=0,
                                       axis_text_x=element_text(angle=90, size=8),
                                       legend_background=element_blank(),
                                       legend_position='top')
        ggsave(plot=self.trunc_freq, filename=self.outpath + '/detection_frequency_trunc.png', height=4, width=12, dpi=300)

    def date_matrix_plot(self):
        chunk_size = 100
        for i in range(0, len(self.df.ID.unique()), chunk_size):
            _ids = self.df.ID.unique()[i:i + chunk_size]

            self.date_matrix = ggplot(data=self.df.loc[self.df.ID.isin(_ids), ['ID', 'Date', 'Detection']],
                                      mapping=aes(x='factor(ID)', y='Date', fill='Detection')) + \
                               ylab('VDS ID') + xlab('Analysis Date') + \
                               scale_fill_manual(self.colors) + \
                               geom_tile() + coord_flip() + theme_bw() +\
                               theme(axis_text_x=element_text(angle=45, hjust=1), text=element_text(size=6))

            outname = '/detection_matrix_{}-{}.png'.format(i, i+chunk_size)
            ggsave(plot=self.date_matrix, filename=self.outpath + outname, height=12, width=4, dpi=300)


if __name__ == "__main__":

    # line by line or as source
    if os.path.isdir('../../experiments/input/D7/5min'):
        analysis = DetectionsPlot(inpath='../../experiments/input/D7/5min', outpath='../../experiments/output/')
    else:
        analysis = DetectionsPlot(inpath='./experiments/input/D7/5min', outpath='./experiments/output/')

    # Generate plots
    analysis.freq_plot()
    analysis.date_matrix_plot()
    # Frequency table by date
    analysis.date_count()





