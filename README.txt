12 June 2017

- Implemented and tested in Python 3
- Required Python libraries
	- numpy
	- shutil
	- os
	- sys
	- csv
	- matplotlib
	- scipy
	- pywt
	- sklearn
	- tensorflow

***** Must extract data in root directory first! *****

***** Main entry to run classifer: tennisai_main.py *****
- model to train and predict with is specified via the "model_type" variable
- k-fold cross validation is down by setting "do_kf" to True and "k" to a desired value. Else, test_train_split is used from sklearn
- predict on test sequence by setting "test_on_vid" to True
- specify respective function parameters by changing:
	- classes
	- sensors
	- use_feats
	- folder_entry

Code Breakdown:
-- annotate_test_vid.m
	- MATLAB script for annotating test sequence video with predicted labels from csv files
	- required videos and csv files not provided due to storage considerations

-- tennisai_main.py
	- model_implement: implement chosen model. returns model, training accuracy, and validation accuracy

-- neural_net.py
	- nn_train: specifies neural net hyperparameters and create appropriate tensorflow objects and session to run
	- deep_net: creates a deep neural net based on specified size and hyperparameters in nn_train

-- pan_split_seq.py
	- pan_seq_test: returns data after applying sliding window to a test sequence
	- pan_seq_neutral: calls pan_seq_test with file paths for neutral training sequences

-- process_data.py
	- get_and_store: gets temporal and spatial data from specified training csv files. returns time, x, y, and z data. "up" and "lo" are values from 0-1 of where to crop data for start and end, respectively
	- features: specifies which features to extract from data and outputs a feature and corresponding labels list
	- fft_data, psd_data, wavelet_data, norm_data: extracts respective features from data

-- visualize_data.py
*** Run for data visualization ONLY ***
	- methods have same general functionality as in process_data.py, however, in a somewhat different format
	- plot_data: plots any three dimensional signal data passed through

-- get_watch_data.py
*** Warning: this script uses Linux specific shell commands and was implemented in Ubuntu 14.04. Functionality issues may rise on other platforms. ***
