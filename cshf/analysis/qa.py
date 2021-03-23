# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import cshf

logger = cshf.CustomLogger(__name__)  # use custom logger


class QA:
    file_cheaters = ''  # csv file with cheaters
    job_id = -1  # appen job ID

    def __init__(self,
                 file_cheaters: str,
                 job_id: int):
        self.file_cheaters = file_cheaters
        self.job_id = job_id

    def flag_users(self):
        """
        Flag users descibed in csv file self.file_cheaters from job
        self.job_id.
        """
        # import csv file
        df = pd.read_csv(self.file_cheaters)
        # check if there are users to flag
        if df.shape[0] == 0:
            return
        logger.info('Flagging {} users.', df.shape[0])
        # count flagged users
        flagged_counter = 0
        # loop over users in the job
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            # make a PUT request for flagging
            cmd_put = 'https://api.appen.com/v1/jobs/' + \
                      str(self.job_id) + \
                      '/workers/' + \
                      str(row['_worker_id']) + \
                      '.json'
            if not pd.isna(row['worker_code']):
                flag_text = 'User repeatidly ignored our instructions and ' \
                            + 'joined job from different accounts/IP ' \
                            + 'addresses. The same code ' \
                            + str(row['worker_code']) \
                            + ' used internally in the job was reused.'
            else:
                flag_text = 'User repeatidly ignored our instructions and ' \
                            + 'joined job from different accounts/IP ' \
                            + 'addresses. No worker code used internally  ' \
                            + 'was inputted (html regex validator was ' \
                            + 'bypassed).'
            params = {'flag': flag_text,
                      'key': cshf.common.get_secrets('appen_api_key')}
            # send PUT request
            r = requests.put(cmd_put,
                             data=params)
            # code 200 means success
            code = r.status_code
            msg = r.content.decode()
            if (code == 200
               and msg != 'Contributor has already been flagged'):
                flagged_counter += 1
            logger.debug('Flagged user {} with message \'{}\' .Returned '
                         + 'code {}: {}',
                         str(row['_worker_id']),
                         flag_text,
                         str(code),
                         r.content)
        logger.info('Flagged {} users successfully (users not flagged '
                    + 'previously).',
                    str(flagged_counter))

    def reject_users(self):
        """
        Reject users descibed in csv file self.file_cheaters from job
        self.job_id.
        """
        # import csv file
        df = pd.read_csv(self.file_cheaters)
        # check if there are users to reject
        if df.shape[0] == 0:
            return
        logger.info('Rejecting {} users.', df.shape[0])
        # count rejected users
        rejected_counter = 0
        # loop over users in the job
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            # make a PUT request for flagging
            cmd_put = 'https://api.appen.com/v1/jobs/' + \
                      str(self.job_id) + \
                      '/workers/' + \
                      str(row['_worker_id']) + \
                      '/reject.json'
            if not pd.isna(row['worker_code']):
                reason_text = 'User repeatidly ignored our instructions and ' \
                            + 'joined job from different accounts/IP ' \
                            + 'addresses. The same code ' \
                            + str(row['worker_code']) \
                            + ' used internally in the job was reused.'
            else:
                reason_text = 'User repeatidly ignored our instructions and ' \
                            + 'joined job from different accounts/IP ' \
                            + 'addresses. No worker code used internally  ' \
                            + 'was inputted (html regex validator was ' \
                            + 'bypassed).'
            params = {'reason': reason_text,
                      'manual': 'true',
                      'key': cshf.common.get_secrets('appen_api_key')}
            # send PUT request
            r = requests.put(cmd_put,
                             data=params)
            # code 200 means success
            code = r.status_code
            msg = r.content.decode()
            if code == 200:
                rejected_counter += 1
            logger.debug('Rejected user {} with message \'{}\' .Returned '
                         + 'code {}: {}',
                         str(row['_worker_id']),
                         reason_text,
                         str(code),
                         msg)
        logger.info('Rejected {} users successfully (users not rejected '
                    + 'previously).',
                    str(rejected_counter))
