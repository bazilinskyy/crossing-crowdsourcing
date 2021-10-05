Analysing perceived risk in pedestrian-vehicle interaction by means of crowdsourcing
=======
This project defines a framework for the analysis of perceived risk in the interaction between pedestrian and vehicle, from the perspective of the driver using a crowdsourcing approach. The jsPsych framework is used to for the frontend. In the description below, it is assumed that the repo is stored in the folder crossing-crowdsourcing. Terminal commands lower assume macOS.

## Setup
Tested with Python 3.8.5. To setup the environment run these two commands in a parent folder of the downloaded repository (replace `/` with `\` and possibly add `--user` if on Windows:
- `pip install -e crossing-crowdsourcing` will setup the project as a package accessible in the environment.
- `pip install -r crossing-crowdsourcing/requirements.txt` will install required packages.

## Implementation on heroku
We use [Heroku](https://www.heroku.com/) to host the node.js implementation. The demo of the implementation may be viewed [here](https://crossing-crowdsourced.herokuapp.com/?debug=1&save_data=0). Implementation supports images and/or videos as stimuli.

## Measuring perceived risk
In this crowdsourcing survey, participants are watching 35 out of a total of 86 videos, that include interactions of a vehicle with pedestrian, from the perspective of the driver. During these videos, the participants are tasked with pressing the F key on their keyboard when they feel a situation could become risky. 
![Example of video](https://github.com/bazilinskyy/gazes-crowdsourced/blob/main/public/img/interaction_1.jpg?raw=true)
An example of one of the videos included in the crowdsourcing survey:

### Filtering of heroku data
Data from heroku is filtered based on the following criteria:
1. People who made more than `config.allowed_mistakes_signs mistakes` in the meaning of traffic signs.
2. People of which the results had deviating video lengths in more than `config.allowed_stimuli_wrong_duration` of the time.

## Crowdsourcing job on appen
We use [appen](http://appen.com) to run a crowdsourcing job. You need to create a client account to be able to create a launch crowdsourcing job. Preview of the appen job used in this experiment is available [here](https://view.appen.io/channels/cf_internal/jobs/1730370/editor_preview?token=22UH3xH4x1hHZy2yVHntEg).

### Filtering of appen data
Data from appen is filtered based on the following criteria:
1. People who did not read instructions.
2. People who are younger than 18 years of age.
3. People who completed the study in under `config.allowed_min_time` min.::vpanel
4. People who completed the study from the same IP more than once (the 1st data entry is retained).
5. People who used the same `worker_code` multiple times. One of the disadvantages of crowdsourcing is having to deal with workers that accept and do crowdsourcing jobs just for money (i.e., `cheaters`). The framework offers filtering mechanisms to remove data from such people from the dataset used for the analysis. Cheaters can be reported from the `gazes.qa.QA` class. It also rejects rows of data from cheaters in appen data and triggers appen to acquire more data to replace the filtered rows.

### Anonymisation of data
Data from appen is anonymised in the following way:
1. IP addresses are assigned to a mask starting from `0.0.0.0` and incrementing by 1 for each unique IP address (e.g., the 257th IP address would be masked as `0.0.0.256`).
2. IDs are anonymised by subtracting the given ID from `config.mask_id`.
