# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers

import crossing as cs

cs.logs(show_level='debug', show_color=True)
logger = cs.CustomLogger(__name__)  # use custom logger

# Const
SAVE_P = True  # save pickle files with data
LOAD_P = False  # load pickle files with data
SAVE_CSV = True  # load csv files with data
REJECT_CHEATERS = False  # reject cheaters on Appen
UPDATE_MAPPING = True  # update mapping with keypress data
file_coords = 'coords.p'  # file to save lists with coordinates
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
    heroku_data = heroku.read_data()
    # create object for working with appen data
    file_appen = cs.common.get_configs('file_appen')
    appen = cs.analysis.Appen(file_data=file_appen,
                              save_p=SAVE_P,
                              load_p=LOAD_P,
                              save_csv=SAVE_CSV)
    # read appen data
    appen_data = appen.read_data()
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
        stimuli_mapped = heroku.read_mapping()
        # process keypresses and update mapping
        stimuli_mapped = heroku.process_kp()
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
        stimuli_mapping = heroku.process_post_stimulus_questions(questions)
        # export to pickle
        cs.common.save_to_p(file_mapping,
                            stimuli_mapped,
                            'mapping with keypress data')
    else:
        stimuli_mapped = cs.common.load_from_p(file_mapping,
                                               'mapping of stimuli')
    # Output
    analysis = cs.analysis.Analysis()
    logger.info('Creating figures.')
    # all keypresses
    analysis.plot_kp(stimuli_mapped)
    # keypresses of an individual stimulus
    analysis.plot_kp_video(stimuli_mapped, 'video_0')
    # keypresses of all videos individually
    analysis.plot_kp_videos(stimuli_mapped)
    # 1 var, all values
    analysis.plot_kp_variable(stimuli_mapped, 'cross_look')
    # 1 var, certain values
    analysis.plot_kp_variable(stimuli_mapped, 'cross_look', ['C_L', 'nC_L'])
    # separate plots for multiple variables
    analysis.plot_kp_variables_or(stimuli_mapped, [{'variable': 'cross_look', 'value': 'C_L'},  # noqa: E501
                                                   {'variable': 'traffic_rules', 'value': 'traffic_lights'},  # noqa: E501
                                                   {'variable': 'traffic_rules', 'value': 'ped_crossing'}])  # noqa: E501
    # multiple variables as a single filter
    analysis.plot_kp_variables_and(stimuli_mapped, [{'variable': 'cross_look', 'value': 'C_L'},  # noqa: E501
                                                    {'variable': 'traffic_rules', 'value': 'traffic_lights'}])  # noqa: E501
    # create correlation matrix
    analysis.corr_matrix(stimuli_mapped, save_file=True)
    # stimulus duration
    analysis.hist(heroku_data,
              x=heroku_data.columns[heroku_data.columns.to_series().str.contains('-dur')],  # noqa: E501
              nbins=100,
              pretty_ticks=True,
              save_file=True)
    # browser window dimensions
    analysis.scatter(heroku_data,
                     x='window_width',
                     y='window_height',
                     color='browser_name',
                     pretty_ticks=True,
                     save_file=True)
    analysis.heatmap(heroku_data,
                     x='window_width',
                     y='window_height',
                     pretty_ticks=True,
                     save_file=True)
    # time of participation
    analysis.hist(appen_data,
                  x=['time'],
                  color='country',
                  pretty_ticks=True,
                  save_file=True)
    # eye contact of driver and pedestrian
    analysis.scatter(appen_data,
                     x='ec_driver',
                     y='ec_pedestrian',
                     color='year_license',
                     pretty_ticks=True,
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
                  pretty_ticks=True,
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
                  pretty_ticks=True,
                  save_file=True)
    # post-trial questions. level of danger
    analysis.bar(stimuli_mapped,
                 y=['risky_slider'],
                 show_all_xticks=True,
                 xaxis_title='Video ID',
                 yaxis_title='Score',
                 save_file=True)
    # post-trial questions. eye contact
    analysis.bar(stimuli_mapped,
                 y=['eye-contact-yes',
                    'eye-contact-yes_but_too_late',
                    'eye-contact-no',
                    'eye-contact-i_don\'t_know'],
                 stacked=True,
                 show_all_xticks=True,
                 xaxis_title='Video ID',
                 yaxis_title='Count',
                 save_file=True)
    # check if any figures are to be rendered
    figures = [manager.canvas.figure
               for manager in
               matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    # show figures, if any
    if figures:
        plt.show()
