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

![Example of video](https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/interaction_1.gif?raw=true).

An example of one of the videos included in the crowdsourcing survey

### Filtering of heroku data
Data from heroku is filtered based on the following criteria:
1. People who made more than `config.allowed_mistakes_signs mistakes` in the meaning of traffic signs.
2. People of which the results had deviating video lengths in more than `config.allowed_stimuli_wrong_duration` of the time, or who had data on less than `config.num_stimuli_participant` videos available.

## Crowdsourcing job on appen
We use [appen](http://appen.com) to run a crowdsourcing job. You need to create a client account to be able to create a launch crowdsourcing job. Preview of the appen job used in this experiment is available [here](https://view.appen.io/channels/cf_internal/jobs/1730370/editor_preview?token=22UH3xH4x1hHZy2yVHntEg).

### Filtering of appen data
Data from appen is filtered based on the following criteria:
1. People who did not read instructions.
2. People who are younger than 18 years of age.
3. People who completed the study in under `config.allowed_min_time` min.::vpanel
4. People who completed the study from the same IP more than once (the 1st data entry is retained).
5. People who used the same `worker_code` multiple times. One of the disadvantages of crowdsourcing is having to deal with workers that accept and do crowdsourcing jobs just for money (i.e., `cheaters`). The framework offers filtering mechanisms to remove data from such people from the dataset used for the analysis. Cheaters can be reported from the `crossing.qa.QA` class. It also rejects rows of data from cheaters in appen data and triggers appen to acquire more data to replace the filtered rows.

### Anonymisation of data
Data from appen is anonymised in the following way:
1. IP addresses are assigned to a mask starting from `0.0.0.0` and incrementing by 1 for each unique IP address (e.g., the 257th IP address would be masked as `0.0.0.256`).
2. IDs are anonymised by subtracting the given ID from `config.mask_id`.

## Analysis
Analysis can be started by running `python crossing-crowdsourcing/gazes/analysis/run.py`. A number of csv files used data processing are saved in `crossing-crowdsourcing/_output`. Visualisations of all data were saved in `crossing-crowdsourcing/_output/figures/`.

### Visualisation


Visualisations of how dynamic variables (keypresses, objects, velocity) change over time. An example for a single video is presented below in terms of keypress data, object data and vehicle speed:

<p float="left">
  <img src="https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/interaction_1.gif?raw=true" width="480" />
  <img src="https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/lineplot_keypresses.gif?raw=true" width="480" /> 
</p>

<p float="left">
  <img src="https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/barchart_objects.gif?raw=true" width="480" />
  <img src="https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/lineplot_vehicle_speed.gif?raw=true" width="480" /> 
</p>

### Regression and correlation analysis
After running `python crossing-crowdsourcing/gazes/analysis/run.py`, a number of plots should show up, including the results of the linear regression. The result of the correlation matrix, and linear regressions for eye contact, vehicle speed, vehicle distance, and object data are shown below:

![Correlation matrix](https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/public/img/all_corr_matrix.png?raw=true)
The correlation matrix that was created using the experimental parameters.

<p float="left">
  <img src="https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/ec-vs-kp.png?raw=true"/>
  <img src="https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/ec-vs-slider.png?raw=true"/> 
</p>
Regression results for eye contact data

<p float="left">
  <img src="https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/objects-vs-kp.png?raw=true"/>
  <img src="https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/objects-vs-slider.png?raw=true"/> 
</p>

<p float="left">
  <img src="https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/objsurface-vs-kp.png?raw=true"/>
  <img src="https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/objsurface-vs-slider.png?raw=true"/> 
</p>
Regression results for object data

<p float="left">
  <img src="https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/speed-vs-kp.png?raw=true"/>
  <img src="https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/speed-vs-slider.png?raw=true"/> 
</p>
Regression results for vehicle speed

<p float="left">
  <img src="https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/distance-vs-kp.png?raw=true"/>
  <img src="https://github.com/bazilinskyy/crossing-crowdsourcing/blob/main/figures/distance-vs-slider.png?raw=true"/> 
</p>
Regression results for distance to the pedestrian

### Configuration of analysis
Configuration of analysis needs to be defined in `gazes-crowdsourced/gazes/analysis/config`. Please use the `default.config` file for the required structure of the file. If no custom config file is provided, `default.config` is used. The config file has the following parameters:
* `appen_job`: ID of the appen job.
* `num_stimuli`: number of videos in the study.
* `num_stimuli_participant`: amount of videos each participant watched
* `stimulus_width`: width of the videos.
* `stimulus_height`: height of the videos.
* `allowed_min_time`: the cut-off for minimal time of participation for filtering.
* `files_heroku`: files with data from heroku.
* `file_appen`: file with data from appen.
* `file_cheaters`: csv file with cheaters for flagging.
* `path_stimuli`: path consisting of all videos included in the survey.
* `mapping_stimuli`: csv file that contains all data found in the videos
* `heroku_output`: csv file that contains all heroku data
* `plotly_template`: template used to make graphs in the analysis
* `kp_resolution`: bin size in ms in which data is stored.
* `video_length`: length of all videos
* `num_repeat`: The amount of times the a video was repeated
* `min_stimulus_duration`: minimal time it took for the participant to watch a single video, for inclusion in analysis.
* `min_stimulus_duration`: maximal time it took for the participant to watch a single video, for inclusion in analysis.
* `allowed_stimulus_wrong_duration`: if the percentage of videos with abnormal length is above this value, exclude participant from analysis
* `allowed_mistakes_signs`: number of allowed mistakes in the questions about traffic signs.
* `sign_answers`: answers to the questions on traffic signs

## Troubleshooting
### Unknown file extension .mp4
If you receive the `ValueError: unknown file extension: .mp4` from `PIL`, install FFMPEG from https://www.ffmpeg.org/download.html. This problem was reported on Windows.
