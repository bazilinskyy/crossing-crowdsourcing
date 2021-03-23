# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import json
import os
import pandas as pd
from collections import Counter
from tqdm import tqdm

import cshf

logger = cshf.CustomLogger(__name__)  # use custom logger


class Appen:
    file_data = []  # list of files with appen data
    appen_data = pd.DataFrame()  # pandas dataframe with extracted data
    save_p = False  # save data as pickle file
    load_p = False  # load data as pickle file
    save_csv = False  # save data as csv file
    file_p = 'appen_data.p'  # pickle file for saving data
    file_csv = 'appen_data.csv'  # csv file for saving data
    file_cheaters_csv = 'cheaters.csv'  # csv file for saving list of cheaters
    # mapping between appen column names and readable names
    columns_mapping = {'_started_at': 'start',
                       '_created_at': 'end',
                       'about_how_many_kilometers_miles_did_you_drive_in_the_last_12_months': 'milage',  # noqa: E501
                       'at_which_age_did_you_obtain_your_first_license_for_driving_a_car_or_motorcycle': 'year_license',  # noqa: E501
                       'have_you_read_and_understood_the_above_instructions': 'instructions',  # noqa: E501
                       'how_many_accidents_were_you_involved_in_when_driving_a_car_in_the_last_3_years_please_include_all_accidents_regardless_of_how_they_were_caused_how_slight_they_were_or_where_they_happened': 'accidents',  # noqa: E501
                       'how_often_do_you_do_the_following_becoming_angered_by_a_particular_type_of_driver_and_indicate_your_hostility_by_whatever_means_you_can': 'dbq1_anger',  # noqa: E501
                       'how_often_do_you_do_the_following_disregarding_the_speed_limit_on_a_motorway': 'dbq2_speed_motorway',  # noqa: E501
                       'how_often_do_you_do_the_following_disregarding_the_speed_limit_on_a_residential_road': 'dbq3_speed_residential',  # noqa: E501
                       'how_often_do_you_do_the_following_driving_so_close_to_the_car_in_front_that_it_would_be_difficult_to_stop_in_an_emergency': 'dbq4_headway',  # noqa: E501
                       'how_often_do_you_do_the_following_racing_away_from_traffic_lights_with_the_intention_of_beating_the_driver_next_to_you': 'dbq5_traffic_lights',  # noqa: E501
                       'how_often_do_you_do_the_following_sounding_your_horn_to_indicate_your_annoyance_with_another_road_user': 'dbq6_horn',  # noqa: E501
                       'how_often_do_you_do_the_following_using_a_mobile_phone_without_a_hands_free_kit': 'dbq7_mobile',  # noqa: E501
                       'if_you_answered_other_in_the_previous_question_please_decribe_the_place_where_you_located_now_below': 'place_other',  # noqa: E501
                       'if_you_answered_other_in_the_previous_question_please_decribe_your_input_device_below': 'device_other',  # noqa: E501
                       'in_which_type_of_place_are_you_located_now': 'place',
                       'in_which_year_do_you_think_that_most_cars_will_be_able_to_drive_fully_automatically_in_your_country_of_residence': 'year_ad',  # noqa: E501
                       'on_average_how_often_did_you_drive_a_vehicle_in_the_last_12_months': 'driving_freq',  # noqa: E501
                       'please_provide_any_suggestions_that_could_help_engineers_to_build_safe_and_enjoyable_automated_cars': 'suggestions_ad',  # noqa: E501
                       'type_the_code_that_you_received_at_the_end_of_the_experiment': 'worker_code',  # noqa: E501
                       'what_is_your_age': 'age',
                       'what_is_your_gender': 'gender',
                       'what_is_your_primary_mode_of_transportation': 'mode_transportation',  # noqa: E501
                       'which_input_device_are_you_using_now': 'device'}

    def __init__(self,
                 file_data: list,
                 save_p: bool,
                 load_p: bool,
                 save_csv: bool):
        self.file_data = file_data
        self.save_p = save_p
        self.load_p = load_p
        self.save_csv = save_csv

    def set_data(self, appen_data):
        """
        Setter for the data object
        """
        old_shape = self.appen_data.shape  # store old shape for logging
        self.appen_data = appen_data
        logger.info('Updated appen_data. Old shape: {}. New shape: {}.',
                    old_shape,
                    self.appen_data.shape)

    def read_data(self):
        # load data
        if self.load_p:
            df = cshf.common.load_from_p(self.file_p,
                                       'appen data')
        # process data
        else:
            # load from csv
            df = pd.read_csv(self.file_data)
            # drop legcy worker code column
            df = df.drop('worker_code', axis=1)
            # drop _gold columns
            df = df.drop((x for x in df.columns.tolist() if '_gold' in x),
                         axis=1)
            # replace linebreaks
            df = df.replace('\n', '', regex=True)
            # rename columns to readable names
            df.rename(columns=self.columns_mapping, inplace=True)
            # convert to time
            df['start'] = pd.to_datetime(df['start'])
            df['end'] = pd.to_datetime(df['end'])
            df['time'] = (df['end'] - df['start']) / pd.Timedelta(seconds=1)
            # filter data
            df = self.filter_data(df)
            # mask IDs and IPs
            df = self.mask_ips_ids(df)
        # save to pickle
        if self.save_p:
            cshf.common.save_to_p(self.file_p,  df, 'appen data')
        # save to csv
        if self.save_csv:
            df.to_csv(gz.settings.output_dir + '/' + self.file_csv)
            logger.info('Saved appen data to csv file {}', self.file_csv)
        # assign to attribute
        self.appen_data = df
        # return df with data
        return df

    def filter_data(self, df):
        """
        Filter data based on the folllowing criteria:
            1. People who did not read instructions.
            2. People that are under 18 years of age.
            3. People who completed the study in under 5 min.
            4. People who completed the study from the same IP more than once
               (the 1st data entry is retained).
            5. People who used the same `worker_code` multiple times.
        """
        # todo: export csv file with cheaters
        logger.info('Filtering appen data.')
        # people that did not read instructions
        df_1 = df.loc[df['instructions'] == 'no']
        logger.info('Filter-a1. People who did not read instructions: {}',
                    df_1.shape[0])
        # people that are underages
        df_2 = df.loc[df['age'] < 18]
        logger.info('Filter-a2. People that are under 18 years of age: {}',
                    df_2.shape[0])
        # People that took less than cshf.common.get_configs('allowed_min_time')
        # minutes to complete the study
        df_3 = df.loc[df['time'] < cshf.common.get_configs('allowed_min_time')]
        logger.info('Filter-a3. People who completed the study in under ' +
                    str(cshf.common.get_configs('allowed_min_time')) +
                    ' sec: {}',
                    df_3.shape[0])
        # people that completed the study from the same IP address
        df_4 = df[df['_ip'].duplicated(keep='first')]
        logger.info('Filter-a4. People who completed the study from the ' +
                    'same IP: {}',
                    df_4.shape[0])
        # people that entered the same worker_code more than once
        df_5 = df[df['worker_code'].duplicated(keep='first')]
        logger.info('Filter-a5. People who used the same worker_code: {}',
                    df_5.shape[0])
        # save to csv
        if self.save_csv:
            df_5 = df_5.reset_index()
            df_5.to_csv(gz.settings.output_dir + '/' + self.file_cheaters_csv)
            logger.info('Filter-a5. Saved list of cheaters to csv file {}',
                        self.file_cheaters_csv)
        # concatanate dfs with filtered data
        old_size = df.shape[0]
        df_filtered = pd.concat([df_1, df_2, df_3, df_4, df_5])
        # drop rows with filtered data
        unique_worker_codes = df_filtered['worker_code'].drop_duplicates()
        df = df[~df['worker_code'].isin(unique_worker_codes)]
        # reset index in dataframe
        df = df.reset_index()
        logger.info('Filtered in total in appen data: {}',
                    old_size - df.shape[0])
        return df

    def mask_ips_ids(self, df, mask_ip=True, mask_id=True):
        """
        Anonymyse IPs and IDs. IDs are anonymised by subtracting the
        given ID from cshf.common.get_configs('mask_id').
        """
        # loop through rows of the file
        if mask_ip:
            proc_ips = []  # store masked IP's here
            logger.info('Replacing IPs in appen data.')
        if mask_id:
            proc_ids = []  # store masked ID's here
            logger.info('Replacing IDs in appen data.')
        for i in range(len(df['_ip'])):  # loop through ips
            # anonymise IPs
            if mask_ip:
                # IP address
                # new IP
                if not any(d['o'] == df['_ip'][i] for d in proc_ips):
                    # mask in format 0.0.0.ID
                    masked_ip = '0.0.0.' + str(len(proc_ips))
                    # record IP as already replaced
                    # o=original; m=masked
                    proc_ips.append({'o': df['_ip'][i], 'm': masked_ip})
                    df.at[i, '_ip'] = masked_ip
                    logger.debug('{}: replaced IP {} with {}.',
                                 df['worker_code'][i],
                                 proc_ips[-1]['o'],
                                 proc_ips[-1]['m'])
                else:  # already replaced
                    for item in proc_ips:
                        if item['o'] == df['_ip'][i]:

                            # fetch previously used mask for the IP
                            df.at[i, '_ip'] = item['m']
                            logger.debug('{}: replaced repeating IP {} with ' +
                                         '{}.',
                                         df['worker_code'][i],
                                         item['o'],
                                         item['m'])
            # anonymise worker IDs
            if mask_id:
                # new worker ID
                if not any(d['o'] == df['_worker_id'][i] for d in proc_ids):
                    # mask in format random_int - worker_id
                    masked_id = (str(cshf.common.get_configs('mask_id') -
                                 df['_worker_id'][i]))
                    # record IP as already replaced
                    proc_ids.append({'o': df['_worker_id'][i],
                                     'm': masked_id})
                    df.at[i, '_worker_id'] = masked_id
                    logger.debug('{}: replaced ID {} with {}.',
                                 df['worker_code'][i],
                                 proc_ids[-1]['o'],
                                 proc_ids[-1]['m'])
                # already replaced
                else:
                    for item in proc_ids:
                        if item['o'] == df['_worker_id'][i]:
                            # fetch previously used mask for the ID
                            df.at[i, '_worker_id'] = item['m']
                            logger.debug('{}: replaced repeating ID {} '
                                         + 'with {}.',
                                         df['worker_code'][i],
                                         item['o'],
                                         item['m'])
        # output for checking
        if mask_ip:
            logger.info('Finished replacement of IPs in appen data.')
            logger.info('Unique IPs detected: {}', str(len(proc_ips)))
        if mask_id:
            logger.info('Finished replacement of IDs in appen data.')
            logger.info('Unique IDs detected: {}', str(len(proc_ids)))
        # return dataframe with replaced values
        return df

    def show_info(self):
        """
        Output info for data in object.
        """
        # info on age
        logger.info('Age: mean={:,.2f}, std={:,.2f}',
                    self.appen_data['age'].mean(),
                    self.appen_data['age'].std())
        # info on gender
        count = Counter(self.appen_data['gender'])
        logger.info('Gender: {}', count.most_common())
        # info on most represted countries in minutes
        count = Counter(self.appen_data['_country'])
        logger.info('Countires: {}', count.most_common())
        # info on duration in minutes
        logger.info('Time of participation: mean={:,.2f} min, '
                    + 'median={:,.2f} min, std={:,.2f} min.',
                    self.appen_data['time'].mean()/60,
                    self.appen_data['time'].median()/60,
                    self.appen_data['time'].std()/60)
        logger.info('oldest timestamp={}, newest timestamp={}.',
                    self.appen_data['start'].min(),
                    self.appen_data['start'].max())
