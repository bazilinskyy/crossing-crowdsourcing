# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers

import crossing as cs

cs.logs(show_level='info', show_color=True)
logger = cs.CustomLogger(__name__)  # use custom logger

# Const
# SAVE_P = True  # save pickle files with data
# LOAD_P = False  # load pickle files with data
# SAVE_CSV = True  # load csv files with data
# FILTER_DATA = True  # filter Appen and heroku data
# CLEAN_DATA = True  # clean Appen data
# REJECT_CHEATERS = True  # reject cheaters on Appen
# UPDATE_MAPPING = True  # update mapping with keypress data
# SHOW_OUTPUT = True  # should figures be plotted

# for debugging, skip processing
SAVE_P = False  # save pickle files with data
LOAD_P = True  # load pickle files with data
SAVE_CSV = True  # load csv files with data
FILTER_DATA = False  # filter Appen and heroku data
CLEAN_DATA = False  # clean Appen data
REJECT_CHEATERS = False  # reject cheaters on Appen
UPDATE_MAPPING = False  # update mapping with keypress data
SHOW_OUTPUT = True  # should figures be plotted

file_mapping = 'mapping.p'  # file to save updated mapping

if __name__ == '__main__':
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
    appen_data = appen.read_data(filter_data=FILTER_DATA,
                                 clean_data=CLEAN_DATA)
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
    # generate country-specific data
    countries_data = appen.process_countries()
    # update mapping with keypress data
    if UPDATE_MAPPING:
        # read in mapping of stimuli
        mapping = heroku.read_mapping()
        # process keypresses and update mapping
        mapping = heroku.process_kp()
        # add distance to pedestrian at different time intervals
        mapping = heroku.check_dist_type(mapping, 'dist_at_9')
        mapping = heroku.check_dist_type(mapping, 'dist_at_10')
        mapping = heroku.check_dist_type(mapping, 'dist_at_11')
        # add count of specific object (district of city, pedestrian, vehicle)
        mapping = heroku.add_district_data(mapping)
        mapping = heroku.add_object_count(mapping, 'person')
        mapping = heroku.add_object_count(mapping, 'car')
        # check is velocity data is processed correctly to match kp
        mapping = heroku.evaluate_bins(mapping, 'vehicle_velocity_GPS')
        mapping = heroku.evaluate_bins(mapping, 'vehicle_velocity_OBD')
        mapping = heroku.evaluate_bins(mapping, 'dist_to_ped')
        mapping = heroku.evaluate_bins(mapping, 'object_count')
        mapping = heroku.evaluate_bins(mapping, 'object_surface')
        mapping = heroku.evaluate_bins(mapping, 'person_count')
        mapping = heroku.evaluate_bins(mapping, 'car_count')
        # add binary data to mapping without specific keys
        mapping = heroku.add_binary_data(mapping, True, 'traffic_rules',
                                         'none', 'Traffic_rules')
        mapping = heroku.add_binary_data(mapping, True, 'cross_look',
                                         'notCrossing', 'Crossing')
        mapping = heroku.add_binary_data(mapping, True, 'cross_look',
                                         'notLooking', 'Looking')
        mapping = heroku.add_binary_data(mapping, False, 'district',
                                         'suburbs', 'Suburbs')
        mapping = heroku.add_binary_data(mapping, False, 'district',
                                         'urbs', 'Urbs')
        mapping = heroku.add_binary_data(mapping, False, 'district',
                                         'outskirt', 'Outskirts')
        mapping = heroku.add_binary_data(mapping, False, 'traffic_rules',
                                         'ped_crossing', 'pedestrian crossing')
        mapping = heroku.add_binary_data(mapping, False, 'traffic_rules',
                                         'stop_sign', 'stop sign')
        mapping = heroku.add_binary_data(mapping, False, 'traffic_rules',
                                         'traffic_lights', 'traffic lights')
        # add data at specific times
        mapping = heroku.add_data_at_time(mapping, 'kp',
                                          [7, 8, 9, 10, 11, 12, 13])
        mapping = heroku.add_data_at_time(mapping, 'dist_to_ped',
                                          [7, 8, 9, 10, 11, 12, 13])
        mapping = heroku.add_data_at_time(mapping, 'object_count',
                                          [7, 8, 9, 10, 11, 12, 13])
        mapping = heroku.add_data_at_time(mapping, 'object_surface',
                                          [7, 8, 9, 10, 11, 12, 13])
        mapping = heroku.add_data_at_time(mapping, 'car_count',
                                          [7, 8, 9, 10, 11, 12, 13])
        mapping = heroku.add_data_at_time(mapping, 'person_count',
                                          [7, 8, 9, 10, 11, 12, 13])
        # add quantification of danger of velocity for each video
        mapping = heroku.add_avg(mapping, 'kp', 'kp')
        mapping = heroku.add_avg(mapping, 'vehicle_velocity_GPS', 'velocity')
        mapping = heroku.add_avg(mapping, 'object_count', 'object')
        mapping = heroku.add_avg(mapping, 'person_count', 'person')
        mapping = heroku.add_avg(mapping, 'car_count', 'car')
        mapping = heroku.add_avg(mapping, 'object_surface', 'obj_surface')
        mapping = heroku.add_cols_avg(mapping,
                                      ['dist_at_9', 'dist_at_10', 'dist_at_11'],  # noqa: E501
                                      'dist')
        # add quantification of danger of velocity for each video
        mapping = heroku.process_velocity_risk(mapping)
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
        mapping.rename(columns={'eye-contact-yes_but_too_late': 'EC-yes_but_too_late',  # noqa: E501
                                'eye-contact-yes': 'EC-yes',
                                'eye-contact-i_don\'t_know': 'EC-i_don\'t_know',  # noqa: E501
                                'eye-contact-no': 'EC-no'},
                       inplace=True)
        # add percentage of participants who wrongly indicated looking data
        mapping = heroku.verify_looking(mapping)
        # calculate mean of eye contact
        mapping['EC-no-score'] = mapping['EC-no'] * 0
        mapping['EC-yes_but_too_late-score'] = mapping['EC-yes_but_too_late'] * 0.25  # noqa: E501
        mapping['EC-yes-score'] = mapping['EC-yes'] * 1
        mapping['EC_score'] = mapping[['EC-yes-score',
                                       'EC-yes_but_too_late-score',
                                       'EC-no-score']].sum(axis=1)
        mapping['EC_mean'] = mapping[['EC-yes-score',
                                      'EC-yes_but_too_late-score',
                                      'EC-no-score']].mean(axis=1)
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
        # all keypresses with confidence interval
        analysis.plot_kp(mapping, conf_interval=0.95)
        # keypresses of an individual stimulus
        analysis.plot_kp_video(mapping, 'video_0', conf_interval=0.95)
        # keypresses of all videos individually
        analysis.plot_kp_videos(mapping)
        # 1 var, all values
        analysis.plot_kp_variable(mapping, 'traffic_rules')
        # 1 var, certain values
        analysis.plot_kp_variable(mapping,
                                  'traffic_rules',
                                  ['ped_crossing', 'stop_sign'])
        # plot of multiple combined AND variables
        analysis.plot_video_data(mapping, 'video_5',
                                 ['vehicle_velocity_GPS', 'dist_to_ped'],
                                 yaxis_title='Distance & velocity data',
                                 conf_interval=0.95)
        analysis.plot_video_data(mapping, 'video_50', ['vehicle_velocity_GPS', 'dist_to_ped'],  # noqa: E501
                                 yaxis_title='Distance & velocity data', conf_interval=0.95)  # noqa: E501
        analysis.plot_kp_variables_and(mapping,
                                       plot_names=['traffic rules',
                                                   'no traffic rules'],
                                       variables_list=[[{'variable': 'traffic_rules',  # noqa: E501
                                                         'value': 'stop_sign'},        # noqa: E501
                                                        {'variable': 'traffic_rules',  # noqa: E501
                                                         'value': 'traffic_lights'},   # noqa: E501
                                                        {'variable': 'traffic_rules',  # noqa: E501
                                                         'value': 'ped_crossing'}],    # noqa: E501
                                                       [{'variable': 'traffic_rules',  # noqa: E501
                                                         'value': 'none'}]])
        # plot of separate variables
        analysis.plot_kp_variables_or(mapping,
                                      variables=[{'variable': 'cross_look',
                                                  'value': 'Crossing_Looking'},     # noqa: E501
                                                 {'variable': 'cross_look',
                                                  'value': 'notCrossing_Looking'},  # noqa: E501
                                                 {'variable': 'cross_look',
                                                  'value': 'Crossing_notLooking'},  # noqa: E501
                                                 {'variable': 'cross_look',
                                                  'value': 'nonspecific'}])

        # columns to drop in correlation matrix and scatter matrix
        columns_drop = ['id_segment', 'set', 'video', 'extra',
                        'alternative_frame', 'alternative_frame.1',
                        'video_length', 'min_dur', 'max_dur', 'start',
                        'danger_b', 'danger-p', 'look_moment', 'cross_moment',
                        'time_before_interaction', 'gesture', 'kp',
                        'look_frame_ms', 'cross_frame_ms', 'interaction',
                        'vehicle_velocity_OBD', 'vehicle_velocity_GPS',
                        'EC-yes-score', 'EC-no-score', 'dist_to_ped',
                        'EC-yes_but_too_late-score', 'object_count',
                        'object_surface', 'object_entities', 'person_count',
                        'car_count']
        # set nan to -1
        df = mapping[(mapping['dist_to_ped_at_7.0'] != 'no data found')]
        df = df.fillna(-1)
        # create correlation matrix
        analysis.corr_matrix(df,
                             columns_drop=columns_drop,
                             save_file=True)
        # create correlation matrix
        analysis.scatter_matrix(df,
                                columns_drop=columns_drop,
                                color='traffic_rules',
                                symbol='traffic_rules',
                                diagonal_visible=False,
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
        df = appen_data
        df['country'] = df['country'].fillna('NaN')
        df['time'] = df['time'] / 60.0  # convert to min
        analysis.hist(df,
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
        # histogram for the year of license
        analysis.hist(appen_data,
                      x=['year_license'],
                      pretty_text=True,
                      save_file=True)
        # histogram for the mode of transportation
        analysis.hist(appen_data,
                      x=['mode_transportation'],
                      pretty_text=True,
                      save_file=True)
        # histogram for the input device
        analysis.hist(appen_data,
                      x=['device'],
                      pretty_text=True,
                      save_file=True)
        # histogram for milage
        analysis.hist(appen_data,
                      x=['milage'],
                      pretty_text=True,
                      save_file=True)
        # histogram for the place of participant
        analysis.hist(appen_data,
                      x=['place'],
                      pretty_text=True,
                      save_file=True)
        # histogram for the eye contact of driver
        analysis.hist(appen_data,
                      x=['ec_driver'],
                      pretty_text=True,
                      save_file=True)
        # histogram for the eye contact of pedestrian
        analysis.hist(appen_data,
                      x=['ec_pedestrian'],
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
                     y=['looking_fails'],
                     show_all_xticks=True,
                     xaxis_title='Video ID',
                     yaxis_title='% participants that wrongly' +
                                 ' indicated looking behaviour',
                     save_file=True)
        # post-trial questions. bar chart for eye contact
        analysis.bar(mapping,
                     y=['risky_slider'],
                     show_all_xticks=True,
                     xaxis_title='Video ID',
                     yaxis_title='Risk score',
                     save_file=True)
        # post-trial questions. bar chart for eye contact
        analysis.bar(mapping,
                     y=['EC-yes',
                        'EC-yes_but_too_late',
                        'EC-no',
                        'EC-i_don\'t_know'],
                     stacked=True,
                     show_all_xticks=True,
                     xaxis_title='Video ID',
                     yaxis_title='Eye contact score ' +
                                 '(No=0, Yes but too late=0.25, Yes=1)',
                     pretty_text=True,
                     save_file=True)
        # post-trial questions. hist for eye contact
        analysis.bar(mapping,
                     y=['EC_score'],
                     stacked=True,
                     show_all_xticks=True,
                     xaxis_title='Video ID',
                     yaxis_title='Eye contact score ' +
                                 '(No=0, Yes but too late=0.25, Yes=1)',
                     pretty_text=True,
                     save_file=True)
        # scatter plot of risk / eye contact without traffic rules involved
        analysis.scatter(mapping[(mapping['cross_look'] != 'notCrossing_Looking') &  # noqa: E501
                                 (mapping['cross_look'] != 'notCrossing_notLooking') &  # noqa: E501
                                 (mapping['velocity_risk'] != 'No velocity data found')],  # noqa: E501
                         x='EC_score',
                         y='risky_slider',
                         # color='traffic_rules',
                         trendline='ols',
                         hover_data=['risky_slider',
                                     'EC_score',
                                     'EC_mean',
                                     'EC-yes',
                                     'EC-yes_but_too_late',
                                     'EC-no',
                                     'EC-i_don\'t_know',
                                     'cross_look',
                                     'traffic_rules'],
                         # pretty_text=True,
                         xaxis_title='Eye contact score '
                                     + '(No=0, Yes but too late=0.25, Yes=1)',  # noqa: E501
                         yaxis_title='The riskiness of behaviour in video'
                                     + ' (0-100)',
                         # xaxis_range=[-10, 100],
                         # yaxis_range=[-1, 20],
                         save_file=True)
        # scatter of velocity vs eye contact
        analysis.scatter(mapping[(mapping['cross_look'] != 'notCrossing_Looking') &  # noqa: E501
                                 (mapping['cross_look'] != 'notCrossing_notLooking') &  # noqa: E501
                                 (mapping['velocity_risk'] != 'No velocity data found')],  # noqa: E501
                         x='velocity_risk',
                         y='risky_slider',
                         # color='traffic_rules',
                         trendline='ols',
                         hover_data=['risky_slider',
                                     'EC_score',
                                     'EC_mean',
                                     'EC-yes',
                                     'EC-yes_but_too_late',
                                     'EC-no',
                                     'EC-i_don\'t_know',
                                     'cross_look',
                                     'traffic_rules'],
                         # pretty_text=True,
                         xaxis_title='Velocity (avg) at keypresses',
                         yaxis_title='The riskiness of behaviour in video'
                                     + ' (0-100)',
                         # xaxis_range=[-10, 100],
                         # yaxis_range=[-1, 20],
                         save_file=True)
        # scatter plot of risk and eye contact without traffic rules involved
        analysis.scatter_mult(mapping,
                              x=['EC-yes',
                                 'EC-yes_but_too_late',
                                 'EC-no',
                                 'EC-i_don\'t_know'],
                              y='risky_slider',
                              trendline='ols',
                              hover_data=['risky_slider',
                                          'EC-yes',
                                          'EC-yes_but_too_late',
                                          'EC-no',
                                          'EC-i_don\'t_know',
                                          'cross_look',
                                          'traffic_rules'],
                              xaxis_title='Subjective eye contact (n)',
                              yaxis_title='Mean risk slider (0-100)',
                              marginal_x='rug',
                              marginal_y=None,
                              save_file=True)
        # scatter plot of risk and percentage of participants indicating eye
        # contact
        analysis.scatter_mult(mapping,
                              x=['EC-yes',
                                  'EC-yes_but_too_late',
                                  'EC-no',
                                  'EC-i_don\'t_know'],
                              y='avg_kp',
                              trendline='ols',
                              xaxis_title='Percentage of participants indicating eye contact (%)',  # noqa: E501
                              yaxis_title='Mean keypresses (%)',
                              marginal_y=None,
                              marginal_x='rug',
                              save_file=True)
        # todo: add comment
        analysis.scatter(mapping[mapping['avg_dist'] != ''],  # noqa: E501
                         x='avg_dist',
                         y='risky_slider',
                         trendline='ols',
                         xaxis_title='Mean distance to pedestrian (m)',
                         yaxis_title='Mean risk slider (0-100)',
                         marginal_x='rug',
                         save_file=True)
        # todo: add comment
        analysis.scatter(mapping[mapping['avg_dist'] != ''],  # noqa: E501
                         x='avg_velocity',
                         y='avg_kp',
                         trendline='ols',
                         xaxis_title='Mean speed of the vehicle (km/h)',
                         yaxis_title='Mean risk slider (0-100)',
                         marginal_x='rug',
                         save_file=True)
        # todo: add comment
        analysis.scatter_mult(mapping[mapping['avg_person'] != ''],     # noqa: E501
                              x=['avg_object', 'avg_person', 'avg_car'],
                              y='risky_slider',
                              trendline='ols',
                              xaxis_title='Object count',
                              yaxis_title='Mean risk slider (0-100)',
                              marginal_y=None,
                              marginal_x='rug',
                              save_file=True)
        # todo: add comment
        analysis.scatter_mult(mapping[mapping['avg_person'] != ''],     # noqa: E501
                              x=['avg_object', 'avg_person', 'avg_car'],
                              y='avg_kp',
                              trendline='ols',
                              xaxis_title='Object count',
                              yaxis_title='Mean keypresses (%)',
                              marginal_y=None,
                              marginal_x='rug',
                              save_file=True)
        # todo: add comment
        analysis.scatter(mapping[mapping['avg_obj_surface'] != ''],    # noqa: E501
                         x='avg_obj_surface',
                         y='avg_kp',
                         trendline='ols',
                         xaxis_title='Mean object surface (0-1)',
                         yaxis_title='Mean keypresses (%)',
                         marginal_x='rug',
                         save_file=True)
        # todo: add comment
        analysis.scatter(mapping[mapping['avg_obj_surface'] != ''],    # noqa: E501
                         x='avg_obj_surface',
                         y='risky_slider',
                         trendline='ols',
                         xaxis_title='Mean object surface (0-1)',
                         yaxis_title='Mean risk slider (0-100)',
                         marginal_x='rug',
                         save_file=True)
        # map of participants
        analysis.map(countries_data, color='counts', save_file=True)
        # map of mean age per country
        analysis.map(countries_data, color='age', save_file=True)
        # map of gender per country
        analysis.map(countries_data, color='gender', save_file=True)
        # map of year of obtaining license per country
        analysis.map(countries_data, color='year_license', save_file=True)
        # map of year of automated driving per country
        analysis.map(countries_data, color='year_ad', save_file=True)
        # check if any figures are to be rendered
        figures = [manager.canvas.figure
                   for manager in
                   matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        # show figures, if any
        if figures:
            plt.show()
