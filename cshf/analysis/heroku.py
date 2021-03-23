# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from collections import Counter

import cshf

# warning about partial assignment
pd.options.mode.chained_assignment = None  # default='warn'

logger = cshf.CustomLogger(__name__)  # use custom logger


class Heroku:
    # todo: parse browser interactions
    files_data = []  # list of files with heroku data
    heroku_data = pd.DataFrame()  # pandas dataframe with extracted data
    save_p = False  # save data as pickle file
    load_p = False  # load data as pickle file
    save_csv = False  # save data as csv file
    # pickle file for saving data
    file_p = 'heroku_data.p'
    # csv file for saving data
    file_data_csv = 'heroku_data'
    # csv file for mapping of stimuli
    file_mapping_csv = 'mapping'
    # keys with meta information
    meta_keys = ['worker_code',
                 'browser_user_agent',
                 'browser_app_name',
                 'browser_major_version',
                 'browser_full_version',
                 'browser_name',
                 'group_choice',
                 'image_ids']
    # prefixes used for files in node.js implementation
    prefixes = {'training': 'training_',
                'stimulus': 'image_',
                'codeblock': 'cb_',
                'sentinel': 'sentinel_',
                'sentinel_cb': 'sentinel_cb_'}
    # stimulus duration
    default_dur = 0

    def __init__(self,
                 files_data: list,
                 save_p: bool,
                 load_p: bool,
                 save_csv: bool):
        self.files_data = files_data
        self.save_p = save_p
        self.load_p = load_p
        self.save_csv = save_csv
        self.num_stimuli = cshf.common.get_configs('num_stimuli')

    def set_data(self, heroku_data):
        """
        Setter for the data object.
        """
        old_shape = self.heroku_data.shape  # store old shape for logging
        self.heroku_data = heroku_data
        logger.info('Updated heroku_data. Old shape: {}. New shape: {}.',
                    old_shape,
                    self.heroku_data.shape)

    def read_data(self):
        """
        Read data into an attribute.
        """
        # todo: read heroku data
        # load data
        if self.load_p:
            df = cshf.common.load_from_p(self.file_p,
                                         'heroku data')
        # process data
        else:
            # read files with heroku data one by one
            data_list = []
            data_dict = {}  # dictionary with data
            for file in self.files_data:
                logger.info('Reading heroku data from {}.', file)
                f = open(file, 'r')
                # add data from the file to the dictionary
                data_list += f.readlines()
                f.close()
            # read rows in data
            for row in tqdm(data_list):  # tqdm adds progress bar
                # use dict to store data
                dict_row = {}
                # load data from a single row into a list
                list_row = json.loads(row)
                # flag that training image was detected
                train_found = False
                # last found training image
                train_name = ''
                # flag that stimulus was detected
                stim_found = False
                # last found stimulus
                stim_name = ''
                # duratoin of last found stimulus
                stim_dur = -1
                # flag that sentinel image was detected
                sent_found = False
                # last found sentinel image
                sent_name = ''
                # last time_elapsed
                time_elapsed_last = -1
                # go over cells in the row with data
                for data_cell in list_row['data']:
                    # extract meta info form the call
                    for key in self.meta_keys:
                        if key in data_cell.keys():
                            # piece of meta data found, update dictionary
                            dict_row[key] = data_cell[key]
                            if key == 'worker_code':
                                logger.debug('{}: working with row with data.',
                                             data_cell['worker_code'])
                    # check if stimulus data is present
                    if 'stimulus' in data_cell.keys():
                        # extract name of stimulus after last slash
                        stim_no_path = data_cell['stimulus'].rsplit('/', 1)[-1]
                        # remove extension
                        stim_no_path = os.path.splitext(stim_no_path)[0]
                        # Check if it is a block with stimulus and not an
                        # instructions block
                        if (cshf.common.search_dict(self.prefixes, stim_no_path)  # noqa: E501
                           is not None):
                            # training is found
                            logger.debug('Found stimulus {}.', stim_no_path)
                            if self.prefixes['training'] in stim_no_path:
                                # Record that stimulus was detected for the
                                # cells to follow
                                train_found = True
                                train_name = stim_no_path
                            # stimulus image is found
                            elif self.prefixes['stimulus'] in stim_no_path:
                                # Record that stimulus was detected for the
                                # cells to follow
                                stim_found = True
                                stim_name = stim_no_path
                                # stimulus duration
                                # todo: uncomment for next study with correct data recording
                                # if 'stimulus_duration' in data_cell.keys():
                                #     stim_dur = data_cell['stimulus_duration']  # noqa: E501
                                if time_elapsed_last > -1:
                                    stim_dur = data_cell['time_elapsed'] - time_elapsed_last  # noqa: E501
                                    # find closest value in the list of
                                    # durations
                                    stim_dur = min(self.durations,
                                                   key=lambda x: abs(x - stim_dur))  # noqa: E501
                                else:  # assign default duration
                                    stim_dur = self.default_dur
                            # codeblock for sentinel image is found
                            elif self.prefixes['sentinel_cb'] in stim_no_path:
                                # record codeblock name for last stimulus
                                if sent_name != '':
                                    # extract ID of codeblock
                                    num_found = re.findall(r'\d+',
                                                           stim_no_path)
                                    # Check if codeblocks were recorded previously  # noqa: E501
                                    if sent_name + '-cb' not in dict_row.keys():  # noqa: E501
                                        # first value
                                        dict_row[sent_name + '-cb'] = [num_found[0]]  # noqa: E501
                                    else:
                                        # previous values found
                                        dict_row[sent_name + '-cb'].append(num_found[0])  # noqa: E501
                            # codeblock image is found
                            elif self.prefixes['codeblock'] in stim_no_path:
                                # record codeblock name for last stimulus or
                                # training image
                                if train_name != '':
                                    # extract ID of codeblock
                                    num_found = re.findall(r'\d+',
                                                           stim_no_path)
                                    # Check if codeblocks were recorded previously  # noqa: E501
                                    if train_name + '-cb' not in dict_row.keys():  # noqa: E501
                                        # first value
                                        dict_row[train_name + '-cb'] = [num_found[0]]  # noqa: E501
                                    else:
                                        # previous values found
                                        dict_row[train_name + '-cb'].append(num_found[0])  # noqa: E501
                                elif stim_name != '':
                                    # extract ID of codeblock
                                    num_found = re.findall(r'\d+',
                                                           stim_no_path)
                                    # Check if codeblocks were recorded previously  # noqa: E501
                                    if stim_name + '-' + str(stim_dur) + '-cb' not in dict_row.keys():  # noqa: E501
                                        # first value
                                        dict_row[stim_name + '-' + str(stim_dur) + '-cb'] = [num_found[0]]  # noqa: E501
                                    else:
                                        # previous values found
                                        dict_row[stim_name + '-' + str(stim_dur) + '-cb'].append(num_found[0])  # noqa: E501
                            # sentinel image is found
                            elif self.prefixes['sentinel'] in stim_no_path:
                                # Record that stimulus was detected for the
                                # cells to follow
                                sent_found = True
                                sent_name = stim_no_path
                    # data entry following a codechart found
                    elif 'responses' in data_cell.keys():
                        # record given input
                        responses = json.loads(data_cell['responses'])
                        logger.debug('Found input {}.',
                                     responses['input-codeblock'])
                        if train_name != '':
                            # turn input to upper case
                            str_in = responses['input-codeblock'].upper()
                            # Check if inputted values were recorded previously
                            if train_name + '-in' not in dict_row.keys():
                                # first value
                                dict_row[train_name + '-in'] = [str_in]
                            else:
                                # previous values found
                                dict_row[train_name + '-in'].append(str_in)
                            # Check if time spent values were recorded previously  # noqa: E501
                            if train_name + '-rt' not in dict_row.keys():
                                # first value
                                dict_row[train_name + '-rt'] = [data_cell['rt']]  # noqa: E501
                            else:
                                # previous values found
                                dict_row[train_name + '-rt'].append(data_cell['rt'])  # noqa: E501
                            # reset flags for found stimulus
                            train_found = False
                            train_name = ''
                        if stim_name != '':
                            # turn input to upper case
                            str_in = responses['input-codeblock'].upper()
                            # Check if inputted values were recorded previously  # noqa: E501
                            if stim_name + '-' + str(stim_dur) + '-in' not in dict_row.keys():  # noqa: E501
                                # first value
                                dict_row[stim_name + '-' + str(stim_dur) + '-in'] = [str_in]  # noqa: E501
                            else:
                                # previous values found
                                dict_row[stim_name + '-' + str(stim_dur) + '-in'].append(str_in)  # noqa: E501
                            # Check if time spent values were recorded previously  # noqa: E501
                            if stim_name + '-' + str(stim_dur) + '-rt' not in dict_row.keys():  # noqa: E501
                                # first value
                                dict_row[stim_name + '-' + str(stim_dur) + '-rt'] = [data_cell['rt']]  # noqa: E501
                            else:
                                # previous values found
                                dict_row[stim_name + '-' + str(stim_dur) + '-rt'].append(data_cell['rt'])  # noqa: E501
                            # reset flags for found stimulus
                            stim_found = False
                            stim_name = ''
                        elif sent_name != '':
                            # turn input to upper case
                            str_in = responses['input-codeblock'].upper()
                            # Check if inputted values were recorded previously
                            if sent_name + '-in' not in dict_row.keys():
                                # first value
                                dict_row[sent_name + '-in'] = [str_in]
                            else:
                                # previous values found
                                dict_row[sent_name + '-in'].append(str_in)
                            # Check if time spent values were recorded previously  # noqa: E501
                            if sent_name + '-rt' not in dict_row.keys():
                                # first value
                                dict_row[sent_name + '-rt'] = [data_cell['rt']]
                            else:
                                # previous values found
                                dict_row[sent_name + '-rt'].append(data_cell['rt'])  # noqa: E501
                            # reset flags for found sentinel image
                            sent_found = False
                            sent_name = ''
                    # record time_elapsed
                    if 'time_elapsed' in data_cell.keys():
                        time_elapsed_last = data_cell['time_elapsed']
                # worker_code was ecnountered before
                if dict_row['worker_code'] in data_dict.keys():
                    # iterate over items in the data dictionary
                    for key, value in dict_row.items():
                        # new value
                        if key not in data_dict[dict_row['worker_code']].keys():  # noqa: E501
                            data_dict[dict_row['worker_code']][key] = value
                        # update old value
                        else:
                            # udpate only if it is a list
                            if isinstance(data_dict[dict_row['worker_code']][key], list):  # noqa: E501
                                data_dict[dict_row['worker_code']][key].extend(value)  # noqa: E501
                # worker_code is ecnountered for the first time
                else:
                    data_dict[dict_row['worker_code']] = dict_row
            # turn into panda's dataframe
            df = pd.DataFrame(data_dict)
            df = df.transpose()
            # report people that attempted study
            unique_worker_codes = df['worker_code'].drop_duplicates()
            logger.info('People who attempted to participate: {}',
                        unique_worker_codes.shape[0])
            # filter data
            df = self.filter_data(df)
        # save to pickle
        if self.save_p:
            cshf.common.save_to_p(self.file_p,  df, 'heroku data')
        # save to csv
        if self.save_csv:
            df.to_csv(gz.settings.output_dir + '/' + self.file_data_csv +
                      '.csv')
            logger.info('Saved heroku data to csv file {}',
                        self.file_data_csv + '.csv')
        # update attribute
        self.heroku_data = df
        # return df with data
        return df

    def read_mapping(self):
        """
        Read mapping.
        """
        # read mapping from a csv file
        mapping = pd.read_csv(cshf.common.get_configs('mapping_stimuli'))
        # set index as stimulus_id
        mapping.set_index('video_id', inplace=True)
        # return mapping as a dataframe
        return mapping

    def populate_mapping(self, df, points_duration, mapping):
        """
        Populate dataframe with mapping of stimuli with counts of detected
        coords for each stimulus duration.
        """
        # todo: add analysis info to mapping matrix. for example mean keypresses
        logger.info('Populating coordinates in mapping of stimuli')
        # # read mapping of polygons from a csv file
        # polygons = pd.read_csv(cshf.common.get_configs('vehicles_polygons'))
        # # set index as stimulus_id
        # polygons.set_index('image_id', inplace=True)
        # # loop over stimuli
        # for stim_id in tqdm(range(1, self.num_stimuli + 1)):
        #     # polygon of vehicle
        #     coords = np.array(polygons.at[stim_id, 'coords'].split(','),
        #                       dtype=int).reshape(-1, 2)
        #     polygon = Polygon(coords)
        #     # loop over durations of stimulus
        #     for duration in range(len(self.durations)):
        #         # loop over coord in the list of coords
        #         for point in points_duration[duration][stim_id]:
        #             # convert to point object
        #             point = Point(point[0], point[1])
        #             # check if point is within polygon of vehicle
        #             if polygon.contains(point):
        #                 # check if nan is in the cell
        #                 if pd.isna(mapping.at[stim_id,
        #                                       self.durations[duration]]):
        #                     mapping.at[stim_id, self.durations[duration]] = 1
        #                 # not nan
        #                 else:
        #                     mapping.at[stim_id, self.durations[duration]] += 1
        #         # count number of participants per duration
        #         name_cell = 'image_' + \
        #                     str(stim_id) + \
        #                     '-' + str(self.durations[duration]) + \
        #                     '-cb'
        #         count = int(self.heroku_data[name_cell].count())
        #         mapping.at[stim_id,
        #                    str(self.durations[duration]) + '_count'] = count
        #     # add area of vehicle polygon
        #     mapping.at[stim_id, 'veh_area'] = polygon.area
        # # add mean value of counts
        # mapping['gazes_mean'] = mapping[self.durations].mean(axis=1)
        # # convert counts of participants to integers
        # for duration in range(len(self.durations)):
        #     column = str(self.durations[duration]) + '_count'
        #     mapping[column] = mapping[column].astype(int)
        # save to csv
        if self.save_csv:
            # save to csv
            mapping.to_csv(gz.settings.output_dir + '/' +
                           self.file_mapping_csv + '.csv')
        # return mapping
        return mapping

    def filter_data(self, df):
        """
        Filter data based on the folllowing criteria:
            1. People who entered incorrect codes for sentinel images more than
               cshf.common.get_configs('allowed_mistakes_sent') times.
        """
        # more than allowed number of mistake with codes for sentinel images
        # load mapping of codes and coordinates
        logger.info('Filtering heroku data.')
        logger.info('Filter-h1. People who made mistakes with sentinel '
                    + 'images.')
        with open(cshf.common.get_configs('mapping_sentinel_cb')) as f:
            mapping = json.load(f)
        allowed_mistakes = cshf.common.get_configs('allowed_mistakes_sent')
        # number of sentinel images in trainig
        training_total = cshf.common.get_configs('training_sent')
        # df to store data to filter out
        df_1 = pd.DataFrame()
        # loop over rows in data
        # tqdm adds progress bar
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            # fill nans with empty lists
            empty = pd.Series([[] for _ in range(len(row.index))],
                              index=row.index)
            row = row.fillna(empty)
            # counter mistakes
            mistakes_counter = 0
            # counter sentinel images found in training
            training_counter = 0
            # loop over values in the row
            for index_r, value_r in row.iteritems():
                # check if input is given
                if (value_r == []):
                    # if no data present, move to the next cell
                    continue
                # sentinel image
                if 'sentinel_' in index_r and '-in' in index_r:
                    # sentinel image in training found
                    if training_counter < training_total:
                        # increase counter of sentinel images
                        training_counter = training_counter + 1
                        # skip since we are still in training data
                        continue
                    # sentinel image not in training found
                    else:
                        # increase counter of sentinel images
                        training_counter = training_counter + 1
                        sent_found = True
                        # extract ID of image
                        num_found = re.findall(r'\d+',
                                               index_r)
                        sent_name = num_found[0]
                        # check if input is in list of correct codes
                        mapping_cb = '../public/img/sentinel/sentinel_' + \
                                     str(sent_name) + \
                                     '.jpg'
                        if (value_r[0] not in mapping[mapping_cb]['correct_codes']):  # noqa: E501
                            # mistake found
                            mistakes_counter = mistakes_counter + 1
                            # check if limit was reached
                            if mistakes_counter > allowed_mistakes:
                                logger.debug('{}: found {} mistakes for '
                                             + 'sentinel images.',
                                             row['worker_code'],
                                             mistakes_counter)
                                # add to df with data to filter out
                                df_1 = df_1.append(row)
                                break
        logger.info('Filter-h1. People who made more than {} mistakes with '
                    + 'sentinel images: {}',
                    allowed_mistakes,
                    df_1.shape[0])
        # concatanate dfs with filtered data
        old_size = df.shape[0]
        # one filter employed
        df_filtered = df_1
        # drop rows with filtered data
        unique_worker_codes = df_filtered['worker_code'].drop_duplicates()
        df = df[~df['worker_code'].isin(unique_worker_codes)]
        logger.info('Filtered in total in heroku data: {}',
                    old_size - df.shape[0])
        return df

    def show_info(self):
        """
        Output info for data in object.
        """
        # info on age
        # info on gender
        count = Counter(self.heroku_data['group_choice'])
        logger.info('Groups: {}', count.most_common())
