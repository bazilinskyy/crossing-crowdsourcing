# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers

import cshf

cshf.logs(show_level='info', show_color=True)
logger = cshf.CustomLogger(__name__)  # use custom logger

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
    files_heroku = cshf.common.get_configs('files_heroku')
    heroku = cshf.analysis.Heroku(files_data=files_heroku,
                                  save_p=SAVE_P,
                                  load_p=LOAD_P,
                                  save_csv=SAVE_CSV)
    # read heroku data
    heroku_data = heroku.read_data()
    # create object for working with appen data
    file_appen = cshf.common.get_configs('file_appen')
    appen = cshf.analysis.Appen(file_data=file_appen,
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
        qa = cshf.analysis.QA(file_cheaters=cshf.common.get_configs('file_cheaters'),  # noqa: E501
                            job_id=cshf.common.get_configs('appen_job'))
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
    heroku.show_info()  # show info for filtered data
    appen_data = all_data[all_data.columns.intersection(appen_data_keys)]
    appen_data = appen_data.set_index('worker_code')
    appen.set_data(appen_data)  # update object with filtered data
    appen.show_info()  # show info for filtered data
    # read in mapping of stimuli
    stimuli_mapped = heroku.read_mapping()
    # Output
    analysis = cshf.analysis.Analysis()
    # number of stimuli to process
    num_stimuli = cshf.common.get_configs('num_stimuli')
    logger.info('Creating figures for {} stimuli.', num_stimuli)
    # todo: add analysis code here
    # show figures, if any
    if figures:
        plt.show()
