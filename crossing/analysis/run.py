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
    # read in mapping of stimuli
    stimuli_mapped = heroku.read_mapping()

    mapping_updated = heroku.keypresses_td()
    #mapping_updated.to_csv('mapping_updated.csv')
    
    # Outputs
    analysis = cs.analysis.Analysis()

    #analysis.keypress_plot(mapping_updated)
    #analysis.plot_variable(mapping_updated)  # all vars

    #Filter data to inputs of choice and append to array
    data = []
    data.append(analysis.filter_data(mapping_updated))
    data.append(analysis.filter_data(mapping_updated, ['cross_look','traffic_rules'],['nC_L', 'none']))
    data.append(analysis.filter_data(mapping_updated, ['cross_look','traffic_rules'],['C_L', 'none']))
    data.append(analysis.filter_data(mapping_updated, ['traffic_rules'],['stop_sign']))
    data.append(analysis.filter_data(mapping_updated, ['traffic_rules'],['ped_crossing']))
    titles = ['All data', 'Not crossing & Looking', 'Crossing & Looking', 'Stop-signs', 'Pedestrian_crossing']
    analysis.plot_variables(data, titles)

    #analysis.keypress_plot_indiv(mapping_updated)

    # number of stimuli to process
    num_stimuli = cs.common.get_configs('num_stimuli')
    logger.info('Creating figures for {} stimuli.', num_stimuli)
    # create correlation matrix
    analysis.corr_matrix(stimuli_mapped, save_file=True)
    # stimulus duration
    analysis.hist_stim_duration(heroku_data, nbins=100, save_file=True)
    # browser window dimensions
    # analysis.hist_browser_dimensions(heroku_data, nbins=100, save_file=True)
    analysis.scatter_browser_dimensions(heroku_data,
                                        type_plot='scatter',
                                        save_file=True)
    # time of participation
    analysis.hist_time_participation(appen_data, save_file=True)
    # check if any figures are to be rendered
    figures = [manager.canvas.figure
               for manager in
               matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    # show figures, if any
    if figures:
        plt.show()
