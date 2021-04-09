# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers

import crossing as cs

cs.logs(show_level='info', show_color=True)
logger = cs.CustomLogger(__name__)  # use custom logger

# Const
SAVE_P = True  # save pickle files with data
LOAD_P = False  # load pickle files with data
SAVE_CSV = True  # load csv files with data
FILTER_DATA = True  # filter Appen and heroku data
REJECT_CHEATERS = False  # reject cheaters on Appen
UPDATE_MAPPING = True  # update mapping with keypress data
SHOW_OUTPUT = True  # shoud figures
file_mapping = 'mapping.p'  # file to save lists with coordinates

if __name__ == '__main__':
    # todo: add descriptions for methods automatically with a sublime plugin
    # create object for working with heroku data
    files_heroku = cs.common.get_configs('files_heroku')
    heroku = cs.analysis.Heroku(files_data=files_heroku,
                                save_p=SAVE_P,
                                load_p=LOAD_P,
                                save_csv=SAVE_CSV)
    # read heroku data
    heroku_data = heroku.read_data(filter_data=FILTER_DATA)
    # create object for working with appen data
    file_appen = cs.common.get_configs('file_appen')
    appen = cs.analysis.Appen(file_data=file_appen,
                              save_p=SAVE_P,
                              load_p=LOAD_P,
                              save_csv=SAVE_CSV)
    # read appen data
    appen_data = appen.read_data(filter_data=FILTER_DATA)
    # get keys in data files
    heroku_data_keys = heroku_data.keys()
    appen_data_keys = appen_data.keys()
    # flag and reject cheaters
    if REJECT_CHEATERS:
        qa = cs.analysis.QA(file_cheaters=cs.common.get_configs('file_cheaters'),  # noqa: E501
                            job_id=cs.common.get_configs('appen_job'))
        qa.flag_users()
        qa.reject_users()
    # merge heroku and appen dataframes into one
    all_data = heroku_data.merge(appen_data,
                                 left_on='worker_code',
                                 right_on='worker_code')
    logger.info('Data from {} participants included in analysis.',
                all_data.shape[0])
    # update original data files
    heroku_data = all_data[all_data.columns.intersection(heroku_data_keys)]
    heroku_data = heroku_data.set_index('worker_code')
    heroku.set_data(heroku_data)  # update object with filtered data
    appen_data = all_data[all_data.columns.intersection(appen_data_keys)]
    appen_data = appen_data.set_index('worker_code')
    appen.set_data(appen_data)  # update object with filtered data
    appen.show_info()  # show info for filtered data
    # update mapping with keypress data
    if UPDATE_MAPPING:
        # read in mapping of stimuli
        mapping = heroku.read_mapping()
        # process keypresses and update mapping
        mapping = heroku.process_kp()
        # post-trial questions to process
        questions = [{'question': 'risky_slider',
                      'type': 'num'},
                     {'question': 'eye-contact',
                      'type': 'str',
                      'options': ['Yes',
                                  'Yes but too late',
                                  'No',
                                  'I don\'t know']}]
        # process post-trial questions and update mapping
        mapping = heroku.process_stimulus_questions(questions)
        # calculate mean of eye contact
        mapping['eye-contact-no'] = mapping['eye-contact-no'] * 1
        mapping['eye-contact-yes_but_too_late'] = mapping['eye-contact-yes_but_too_late'] * 2  # noqa: E501
        mapping['eye-contact-yes'] = mapping['eye-contact-yes'] * 3
        mapping['eye-contact_score'] = mapping[['eye-contact-yes',
                                                'eye-contact-yes_but_too_late',
                                                'eye-contact-no']].sum(axis=1)
        mapping['eye-contact_mean'] = mapping[['eye-contact-yes',
                                               'eye-contact-yes_but_too_late',
                                               'eye-contact-no']].mean(axis=1)
        # export to pickle
        cs.common.save_to_p(file_mapping,
                            mapping,
                            'mapping with keypress data')
    else:
        mapping = cs.common.load_from_p(file_mapping,
                                        'mapping of stimuli')
    if SHOW_OUTPUT:
        # Output
        analysis = cs.analysis.Analysis()
        logger.info('Creating figures.')
        # all keypresses
        analysis.plot_kp(mapping)
        # keypresses of an individual stimulus
        analysis.plot_kp_video(mapping, 'video_0')
        # keypresses of all videos individually
        analysis.plot_kp_videos(mapping)
        # 1 var, all values
        analysis.plot_kp_variable(mapping, 'cross_look')
        # 1 var, certain values
        analysis.plot_kp_variable(mapping, 'cross_look', ['C_L', 'nC_L'])
        # separate plots for multiple variables
        analysis.plot_kp_variables_or(mapping, [{'variable': 'cross_look', 'value': 'C_L'},  # noqa: E501
                                                {'variable': 'traffic_rules', 'value': 'traffic_lights'},  # noqa: E501
                                                {'variable': 'traffic_rules', 'value': 'ped_crossing'}])  # noqa: E501
        # multiple variables as a single filter
        analysis.plot_kp_variables_and(mapping, [{'variable': 'cross_look', 'value': 'C_L'},  # noqa: E501
                                                        {'variable': 'traffic_rules', 'value': 'traffic_lights'}])  # noqa: E501
        # columns to drop in correlation matrix and scatter matrix
        columns_drop = ['id_segment', 'set', 'video', 'extra',
                        'alternative_frame', 'alternative_frame.1', 'kp',
                        'video_length', 'min_dur', 'max_dur',
                        'eye-contact-yes', 'eye-contact-yes_but_too_late',
                        'eye-contact-no', 'eye-contact-i_don\'t_know',
                        'eye-contact_mean']
        # set nan to -1
        df = mapping
        df = df.fillna(-1)
        # create correlation matrix
        analysis.corr_matrix(df,
                             columns_drop=columns_drop,
                             save_file=True)
        # create correlation matrix
        analysis.scatter_matrix(df,
                                columns_drop=columns_drop,
                                color='cross_look',
                                symbol='cross_look',
                                diagonal_visible=True,
                                save_file=True)
        # stimulus duration
        analysis.hist(heroku_data,
                  x=heroku_data.columns[heroku_data.columns.to_series().str.contains('-dur')],  # noqa: E501
                  nbins=100,
                  pretty_text=True,
                  save_file=True)
        # browser window dimensions
        analysis.scatter(heroku_data,
                         x='window_width',
                         y='window_height',
                         color='browser_name',
                         pretty_text=True,
                         save_file=True)
        analysis.heatmap(heroku_data,
                         x='window_width',
                         y='window_height',
                         pretty_text=True,
                         save_file=True)
        # time of participation
        analysis.hist(appen_data,
                      x=['time'],
                      color='country',
                      pretty_text=True,
                      save_file=True)
        # eye contact of driver and pedestrian
        analysis.scatter(appen_data,
                         x='ec_driver',
                         y='ec_pedestrian',
                         color='year_license',
                         pretty_text=True,
                         save_file=True)
        # barchart of communication data
        post_qs = ['Importance of eye contact to pedestrian',
                   'Importance of hand gestures to pedestrian',
                   'Importance of eye contact to driver',
                   'Importance of light signaling to driver',
                   'Importance of waiting for car slow down']
        analysis.communication(all_data,
                               pre_q='communication_importance',
                               post_qs=post_qs,
                               save_file=True)
        # histogram for driving frequency
        analysis.hist(appen_data,
                      x=['driving_freq'],
                      pretty_text=True,
                      save_file=True)
        # grouped barchart of DBQ data
        analysis.hist(appen_data,
                      x=['dbq1_anger',
                         'dbq2_speed_motorway',
                         'dbq3_speed_residential',
                         'dbq4_headway',
                         'dbq5_traffic_lights',
                         'dbq6_horn',
                         'dbq7_mobile'],
                      marginal='violin',
                      pretty_text=True,
                      save_file=True)
        # post-trial questions. level of danger
        analysis.bar(mapping,
                     x=mapping.index,
                     y=['risky_slider'],
                     show_all_xticks=True,
                     xaxis_title='Video ID',
                     yaxis_title='Score',
                     save_file=True)
        # post-trial questions. bar chart for eye contact
        analysis.bar(mapping,
                     x=mapping.index,
                     y=['eye-contact-yes',
                        'eye-contact-yes_but_too_late',
                        'eye-contact-no',
                        'eye-contact-i_don\'t_know'],
                     stacked=True,
                     show_all_xticks=True,
                     xaxis_title='Video ID',
                     yaxis_title='Count',
                     pretty_text=True,
                     save_file=True)
        # post-trial questions. hist for eye contact
        analysis.hist(df,
                      x=['eye-contact_score'],
                      pretty_text=True,
                      xaxis_title='Whether pedestiran made eye contact',
                      yaxis_title='Count',
                      save_file=True)
        # scatter plot of risk score / eye contact
        analysis.scatter(df,
                         x='risky_slider',
                         y='eye-contact_score',
                         color='cross_look',
                         hover_data=['risky_slider',
                                     'eye-contact_score',
                                     'eye-contact_mean',
                                     'eye-contact-yes',
                                     'eye-contact-yes_but_too_late',
                                     'eye-contact-no',
                                     'eye-contact-i_don\'t_know',
                                     'cross_look',
                                     'traffic_rules'],
                         # pretty_text=True,
                         xaxis_title='The riskiness of behaviour in video '
                                     + '(0-100)',
                         yaxis_title='Whether pedestiran made eye contact '
                                     + '(No=1, Yes but too late=2, Yes=3)',
                         # xaxis_range=[-10, 100],
                         # yaxis_range=[-1, 20],
                         save_file=True)
        # check if any figures are to be rendered
        figures = [manager.canvas.figure
                   for manager in
                   matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        # show figures, if any
        if figures:
            plt.show()
