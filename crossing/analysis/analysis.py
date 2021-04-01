# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import json
import os
import subprocess
import io
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
from plotly import subplots

import crossing as cs

matplotlib.use('TkAgg')
logger = cs.CustomLogger(__name__)  # use custom logger


class Analysis:
    # folder for output
    folder = '/figures/'
    # resolution of keypress data
    res = 0

    def __init__(self, res: int):
        # set font to Times
        plt.rc('font', family='serif')
        # set template for plotly output
        self.template = cs.common.get_configs('plotly_template')
        # store resolution for keypress data
        self.res = res

    def corr_matrix(self, mapping, save_file=False):
        """
        Output correlation matrix.
        """
        logger.info('Creating correlation matrix.')
        # drop not needed columns
        columns_drop = ['id_segment', 'set', 'video', 'extra',
                        'alternative_frame', 'alternative_frame']
        mapping = mapping.drop(columns_drop, 1)
        # mapping.fillna(0, inplace=True)
        # create correlation matrix
        corr = mapping.corr()
        # create mask
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        # set larger font
        s_font = 12  # small
        m_font = 16  # medium
        l_font = 18  # large
        plt.rc('font', size=s_font)         # controls default text sizes
        plt.rc('axes', titlesize=s_font)    # fontsize of the axes title
        plt.rc('axes', labelsize=m_font)    # fontsize of the axes labels
        plt.rc('xtick', labelsize=s_font)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=s_font)   # fontsize of the tick labels
        plt.rc('legend', fontsize=s_font)   # fontsize of the legend
        plt.rc('figure', titlesize=l_font)  # fontsize of the figure title
        plt.rc('axes', titlesize=m_font)    # fontsize of the subplot title
        # create figure
        fig = plt.figure(figsize=(15, 8))
        g = sns.heatmap(corr,
                        annot=True,
                        mask=mask,
                        cmap='coolwarm',
                        fmt=".2f")
        # rotate ticks
        for item in g.get_xticklabels():
            item.set_rotation(45)
        # save image
        if save_file:
            self.save_fig('all',
                          fig,
                          self.folder,
                          '_corr_matrix.jpg',
                          pad_inches=0.05)
        # revert font
        self.reset_font()

    def hist_stim_duration(self, df, nbins=0, save_file=True):
        """
        Output distribution of stimulus durations.

        Args:
            df (dataframe): dataframe with data from heroku.
            nbins (int, optional): number of bins in histogram.
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating histogram of stimulus durations.')
        df = df[df.columns[df.columns.to_series().str.contains('-dur')]]
        # create figure
        if nbins:
            fig = px.histogram(df, nbins=nbins, marginal='rug')
        else:
            fig = px.histogram(df, marginal='rug')
        # ticks as numbers
        fig.update_layout(xaxis=dict(tickformat='digits'))
        # update layout
        fig.update_layout(template=self.template)
        # save file
        if save_file:
            self.save_plotly(fig, 'hist_stim_duration', self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def hist_time_participation(self, df, nbins=0, save_file=True):
        """
        Output histogram of time of participation.

        Args:
            df (dataframe): dataframe with data from heroku.
            nbins (int, optional): number of bins in histogram.
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating histogram of time of study.')
        # create figure
        if nbins:
            fig = px.histogram(df,
                               x='time',
                               nbins=nbins,
                               marginal='rug',
                               color='country')
        else:
            fig = px.histogram(df, x='time', marginal='rug', color='country')
        # ticks as numbers
        fig.update_layout(xaxis=dict(tickformat='digits'))
        # update layout
        fig.update_layout(template=self.template)
        # save file
        if save_file:
            self.save_plotly(fig, 'hist_time_participation', self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def hist_browser_dimensions(self, df, nbins=0, save_file=True):
        """
        Output distribution of browser dimensions.

        Args:
            df (dataframe): dataframe with data from heroku.
            nbins (int, optional): number of bins in histogram.
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating histogram of window dimensions.')
        df['window_area'] = df['window_height'] * df['window_width']
        # plotly
        fig = subplots.make_subplots(rows=1,
                                     cols=1,
                                     shared_xaxes=True)
        # variables to plot
        variables = ['window_height', 'window_width', 'window_area']
        data = df[variables]
        # plot each variable in data
        for i, variable in enumerate(variables):
            if nbins:
                fig.add_trace(go.Histogram(x=df[variable],
                                           nbinsx=nbins,
                                           name=variable),
                              row=1,
                              col=1)
            else:
                fig.add_trace(go.Histogram(x=df[variable],
                                           name=variable),
                              row=1,
                              col=1)
        buttons = list([dict(label='All',
                             method='update',
                             args=[{'visible': [True] * 3 * len(variables)},
                                   {'title': 'All',
                                    'showlegend': True}])])
        for i, label in enumerate(variables):
            visibility = [[i == j] for j in range(len(variables))]
            visibility = [item for sublist in visibility for item in sublist]
            button = dict(label=label,
                          method='update',
                          args=[{'visible': visibility},
                                {'title': label}])
            buttons.append(button)
        updatemenus = [dict(x=-0.15, buttons=buttons, showactive=True)]
        # ticks as numbers
        fig.update_layout(xaxis=dict(tickformat='digits'))
        # update layout
        fig['layout']['title'] = 'Title'
        # fig['layout']['showlegend'] = True
        fig['layout']['updatemenus'] = updatemenus
        fig.update_layout(template=self.template)
        # save file
        if save_file:
            self.save_plotly(fig, 'hist_browser_dimensions', self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def scatter_browser_dimensions(self, df, type_plot='scatter',
                                   save_file=True):
        """
        Output scatter plot of browser dimensions.

        Args:
            df (dataframe): dataframe with data from heroku.
            type_plot (str, optional): type of plot: scatter, density_heatmap.
            save_file (bool, optional): flag for saving an html file with plot.
        """
        logger.info('Creating plot of type_plot {} for browser dimensions.',
                    type_plot)
        # scatter plot with histograms
        if type_plot == 'scatter':
            fig = px.scatter(df,
                             x='window_width',
                             y='window_height',
                             marginal_x='violin',
                             marginal_y='violin',
                             color='browser_name')
        # density map with histograms
        elif type_plot == 'density_heatmap':
            fig = px.density_heatmap(df,
                                     x='window_width',
                                     y='window_height',
                                     marginal_x='violin',
                                     marginal_y='violin')
        # unsopported type
        else:
            logger.error('Wrong type of plot {} given.', type_plot)
            return -1

        # update layout
        fig.update_layout(template=self.template)
        # save file
        if save_file:
            self.save_plotly(fig, 'scatter_browser_dimensions', self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp(self, df, save_file=True,
                xaxis_title='Time (ms)',
                yaxis_title='Percentage of keypresses',):
        """Take in a variable with values which are optional
        Args:
            data (array of keypress data): Array containing data of all classes
                                           to plot.
            titles (array of strings): Array with the same length as data,
                                       which are the plot names.
        """
        logger.info('Creating visualisations of keypresses for all data.')
        # calculate times
        times = np.array(range(self.res, df['video_length'].max() + self.res, self.res)) / 1000  # noqa: E501
        # add all data together. Must be converted to np array to add together
        kp_data = np.array([0] * len(times))
        for i, data in enumerate(df['keypresses']):
            kp_data += np.array(data)
        kp_data = (kp_data / i)
        # plot keypresses
        fig = px.line(y=kp_data,
                      x=times,
                      title='Keypresses for all stimuli')
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)
        # save file
        if save_file:
            self.save_plotly(fig, 'kp', self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_video(self, df, stimulus, extention='mp4',
                      xaxis_title='Time (ms)',
                      yaxis_title='Percentage of keypresses',
                      save_file=True):
        """Plot keypresses with multiple variables as a filter.

        Args:
            df (TYPE): Description
            res (TYPE): Description
            save_file (bool, optional): Description
        """
        # name of video file
        video_file = stimulus + '.' + extention
        # extract video length
        video_len = df.loc[video_file]['video_length']
        # calculate times
        times = np.array(range(self.res, video_len + self.res, self.res)) / 1000  # noqa: E501
        # plot keypresses
        fig = px.line(y=df.loc[video_file]['keypresses'],
                      x=times,
                      title='Keypresses for stimulus ' + stimulus)
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)
        # save file
        if save_file:
            self.save_plotly(fig, 'kp_' + stimulus, self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_videos(self, df, xaxis_title='Time (ms)',
                       yaxis_title='Percentage of keypresses',
                       save_file=True):
        """Plot keypresses with multiple variables as a filter.

        Args:
            df (TYPE): Description
            res (TYPE): Description
            save_file (bool, optional): Description
        """
        # calculate times
        times = np.array(range(self.res, df['video_length'].max() + self.res, self.res)) / 1000  # noqa: E501

        # plotly
        fig = subplots.make_subplots(rows=1,
                                     cols=1,
                                     shared_xaxes=True)
        # plot for all videos
        for index, row in df.iterrows():
            values = row['keypresses']
            fig.add_trace(go.Scatter(y=values,
                                     mode='lines',
                                     x=times,
                                     name=os.path.splitext(index)[0]),
                          row=1,
                          col=1)
        buttons = list([dict(label='All',
                             method='update',
                             args=[{'visible': [True] * df.shape[0]},
                                   {'title': 'Keypresses for individual stimuli',  # noqa: E501
                                    'showlegend': True}])])
        # counter for traversing through stimuli
        counter_rows = 0
        for index, row in df.iterrows():
            visibility = [[counter_rows == j] for j in range(df.shape[0])]
            visibility = [item for sublist in visibility for item in sublist]
            button = dict(label=os.path.splitext(index)[0],
                          method='update',
                          args=[{'visible': visibility},
                                {'title': os.path.splitext(index)[0]}])
            buttons.append(button)
            counter_rows = counter_rows + 1
        updatemenus = [dict(x=-0.15, buttons=buttons, showactive=True)]
        fig['layout']['updatemenus'] = updatemenus
        # update layout
        fig['layout']['title'] = 'Keypresses for individual stimuli'
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)
        # save file
        if save_file:
            self.save_plotly(fig, 'kp_videos', self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_variable(self, df, variable, values=None,
                         xaxis_title='Time (ms)',
                         yaxis_title='Percentage of keypresses',
                         save_file=True):
        """Plot figures of individual videos with analysis.
        Args:
            df (TYPE): updated dataframe with keypress data.
            res (TYPE): Description
            variable (TYPE): Description
            values (None, optional): Description
            save_file (bool, optional): Description.
        """
        logger.info('Creating visualisation of keypresses based on values ' +
                    '{} of variable {} .', values, variable)
        # calculate times
        times = np.array(range(self.res, df['video_length'].max() + self.res, self.res)) / 1000  # noqa: E501
        # if no values specified, plot value
        if not values:
            values = df[variable].unique()
        print(values)
        # extract data for values
        extracted_data = []
        for data in values:
            keypress_data = np.array([0] * len(times))
            for index, row in df[df[variable] == data].iterrows():
                keypress_data = keypress_data + np.array(row['keypresses'])
            extracted_data.append({'value': data,
                                   'data': keypress_data / len(keypress_data)})
        # plotly figure
        fig = subplots.make_subplots(rows=1,
                                     cols=1,
                                     shared_xaxes=True)
        # plot each variable in data
        for data in extracted_data:
            fig.add_trace(go.Scatter(y=data['data'],  # noqa: E501
                                     mode='lines',
                                     x=times,
                                     name=data['value']),
                          row=1,
                          col=1)
        # create tabs
        buttons = list([dict(label='All',
                             method='update',
                             args=[{'visible': [True] * len(values)},
                                   {'title': 'All',
                                    'showlegend': True}])])
        # counter for traversing through stimuli
        counter_rows = 0
        for value in values:
            visibility = [[counter_rows == j] for j in range(len(values))]
            visibility = [item for sublist in visibility for item in sublist]  # noqa: E501
            button = dict(label=value,
                          method='update',
                          args=[{'visible': visibility},
                                {'title': value}])
            buttons.append(button)
            counter_rows = counter_rows + 1
        # add menu
        updatemenus = [dict(x=-0.15, buttons=buttons, showactive=True)]
        fig['layout']['updatemenus'] = updatemenus
        # update layout
        fig['layout']['title'] = 'Keypresses for ' + variable
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)
        # save file
        if save_file:
            self.save_plotly(fig,
                             'kp_' + variable + '-' + ','.join(values),
                             self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_variables_or(self, df, variables, xaxis_title='Time (ms)',
                             yaxis_title='Percentage of keypresses',
                             save_file=True):
        """Separate plots of keypresses with multiple variables as a filter.

        Args:
            df (TYPE): Description
            res (TYPE): Description
            variables (TYPE): Description
            save_file (bool, optional): Description
        """
        logger.info('Creating visualisation of keypresses based on ' +
                    'variables {} with OR filter.', variables)
        # build string with variables
        variables_str = ''
        for variable in variables:
            variables_str = variables_str + '_' + variable['variable'] + \
                '-' + variable['value']
        # calculate times
        times = np.array(range(self.res, df['video_length'].max() + self.res, self.res)) / 1000  # noqa: E501
        # extract data for values
        extracted_data = []
        for var in variables:
            keypress_data = np.array([0] * len(times))
            for index, row in df[df[var['variable']] == var['value']].iterrows():  # noqa: E501
                keypress_data = keypress_data + np.array(row['keypresses'])
            extracted_data.append({'value': var['variable'] + '-' + var['value'],  # noqa: E501
                                   'data': keypress_data / len(keypress_data)})
        # plotly figure
        fig = subplots.make_subplots(rows=1,
                                     cols=1,
                                     shared_xaxes=True)
        # plot each variable in data
        for data in extracted_data:
            fig.add_trace(go.Scatter(y=data['data'],  # noqa: E501
                                     mode='lines',
                                     x=times,
                                     name=data['value']),  # noqa: E501
                          row=1,
                          col=1)
        # create tabs
        buttons = list([dict(label='All',
                             method='update',
                             args=[{'visible': [True] * len(variables)},
                                   {'title': 'All',
                                    'showlegend': True}])])
        # counter for traversing through stimuli
        counter_rows = 0
        for data in extracted_data:
            visibility = [[counter_rows == j] for j in range(len(variables))]
            visibility = [item for sublist in visibility for item in sublist]
            button = dict(label=data['value'],
                          method='update',
                          args=[{'visible': visibility},
                                {'title': data['value']}])  # noqa: E501
            buttons.append(button)
            counter_rows = counter_rows + 1
        # add menu
        updatemenus = [dict(x=-0.15, buttons=buttons, showactive=True)]
        fig['layout']['updatemenus'] = updatemenus
        # update layout
        fig['layout']['title'] = 'Keypresses with OR filter'
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)
        # save file
        if save_file:
            self.save_plotly(fig, 'kp_or' + variables_str, self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def plot_kp_variables_and(self, df, variables,
                              xaxis_title='Time (ms)',
                              yaxis_title='Percentage of keypresses',
                              save_file=True):
        """Separate plots of keypresses with multiple variables as a filter.

        Args:
            df (TYPE): Description
            res (TYPE): Description
            variables (TYPE): Description
            save_file (bool, optional): Description
        """
        logger.info('Creating visualisation of keypresses based on ' +
                    'variables {} with AND filter.', variables)
        # build string with variables
        variables_str = ''
        for variable in variables:
            variables_str = variables_str + '_' + variable['variable']
        # calculate times
        times = np.array(range(self.res, df['video_length'].max() + self.res, self.res)) / 1000  # noqa: E501
        # filter df based on variables given
        for var in variables:
            df = df[df[var['variable']] == var['value']]
        # check if any data in df left
        if df.empty:
            logger.error('Provided variables yielded empty dataframe.')
            return
        # add all data together. Must be converted to np array to add together
        kp_data = np.array([0] * len(times))
        for i, data in enumerate(df['keypresses']):
            kp_data += np.array(data)
        kp_data = (kp_data / i)
        # plot keypresses
        fig = px.line(y=kp_data,
                      x=times,
                      title='Keypresses with AND filter')
        # update layout
        fig.update_layout(template=self.template,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)
        # save file
        if save_file:
            self.save_plotly(fig, 'kp_and' + variables_str, self.folder)
        # open it in localhost instead
        else:
            fig.show()

    def save_plotly(self, fig, name, output_subdir):
        """
        Helper function to save figure as file.
        """
        # build path
        path = cs.settings.output_dir + output_subdir
        if not os.path.exists(path):
            os.makedirs(path)
        file_plot = os.path.join(path + name + '.html')
        # save to file
        py.offline.plot(fig, filename=file_plot)

    def save_fig(self, image, fig, output_subdir, suffix, pad_inches=0):
        """
        Helper function to save figure as file.
        """
        # extract name of stimulus after last slash
        file_no_path = image.rsplit('/', 1)[-1]
        # remove extension
        file_no_path = os.path.splitext(file_no_path)[0]
        # create path
        path = cs.settings.output_dir + output_subdir
        if not os.path.exists(path):
            os.makedirs(path)
        # save file
        plt.savefig(path + file_no_path + suffix,
                    bbox_inches='tight',
                    pad_inches=pad_inches)
        # clear figure from memory
        plt.close(fig)

    def save_anim(self, image, anim, output_subdir, suffix):
        """
        Helper function to save figure as file.
        """
        # extract name of stimulus after last slash
        file_no_path = image.rsplit('/', 1)[-1]
        # remove extension
        file_no_path = os.path.splitext(file_no_path)[0]
        # create path
        path = cs.settings.output_dir + output_subdir
        if not os.path.exists(path):
            os.makedirs(path)
        # save file
        anim.save(path + file_no_path + suffix, writer='ffmpeg')
        # clear animation from memory
        plt.close(self.fig)

    def autolabel(self, ax, on_top=False, decimal=True):
        """
        Attach a text label above each bar in *rects*, displaying its height.
        """
        # todo: optimise to use the same method
        # on top of bar
        if on_top:
            for rect in ax.patches:
                height = rect.get_height()
                # show demical points
                if decimal:
                    label_text = f'{height:.2f}'
                else:
                    label_text = f'{height:.0f}'
                ax.annotate(label_text,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center',
                            va='bottom')
        # in the middle of the bar
        else:
            # based on https://stackoverflow.com/a/60895640/46687
            # .patches is everything inside of the chart
            for rect in ax.patches:
                # Find where everything is located
                height = rect.get_height()
                width = rect.get_width()
                x = rect.get_x()
                y = rect.get_y()
                # The height of the bar is the data value and can be used as
                # the label
                # show demical points
                if decimal:
                    label_text = f'{height:.2f}'
                else:
                    label_text = f'{height:.0f}'
                label_x = x + width / 2
                label_y = y + height / 2
                # plot only when height is greater than specified value
                if height > 0:
                    ax.text(label_x,
                            label_y,
                            label_text,
                            ha='center',
                            va='center')

    def reset_font(self):
        """
        Reset font to default size values. Info at
        https://matplotlib.org/tutorials/introductory/customizing.html
        """
        s_font = 8
        m_font = 10
        l_font = 12
        plt.rc('font', size=s_font)         # controls default text sizes
        plt.rc('axes', titlesize=s_font)    # fontsize of the axes title
        plt.rc('axes', labelsize=m_font)    # fontsize of the axes labels
        plt.rc('xtick', labelsize=s_font)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=s_font)   # fontsize of the tick labels
        plt.rc('legend', fontsize=s_font)   # legend fontsize
        plt.rc('figure', titlesize=l_font)  # fontsize of the figure title
