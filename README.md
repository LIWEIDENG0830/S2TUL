# S2TUL

## Requirements
We depict the environments of our experiments as follows.
* Ubuntu OS
* Python >= 3.6 (Anaconda is recommended)
* Pytorch >= 1.5.0 (Tested on 1.5.0)

## File Descriptions
* data : the directory for saving data, in which foursquare (#User=108) dataset is included. 
        The other datasets cannot be uploaded since the oversize of the files, and will 
        be released once this paper is accepted.
* batched_main.py : training the **S2TUL-R** model.
* batched_main_with_spatioinfo.py : training the **S2TUL-HRS** and **S2TUL-HRS-G** models.
* batched_main_with_spatioinfo_hm.py : training the **S2TUL-RS** model.
* batched_main_with_spatiotemporalinfo.py : training the **S2TUL-HRST** and **S2TUL-HRST-G** models.
* batched_main_with_spatiotemporalinfo_hm.py : training the **S2TUL-RST** model.
* batched_main_withLSTM.py : training the **S2TUL-HRSTS** model.
* config.py : storing the configurations of the models.
* dataset.py : defining the I/O of the files.
* models.py : defining the classes of the models.
* utils.py : containing lots of tool functions.

## How to reproduce the results in the paper?
Firstly, a user should modify the "datadir" in the config.py and set suitable hyper-parameters in the config
 according to paper descriptions.

Then, taking **S2TUL-R** as an example, just run with the following command.
```shell script
python batched_main.py
```
