# NEED: Automatic Detection of Gaze Events in Natural and Dynamic Viewing

## Introduction
we introduce a natural viewing eye-movement event detection method that takes head and body motion into account only relying on the scene camera and gaze signal which are provided by a mobile eye-trackers.

![architecture](./figs/arch.png)

The natural eye-movement event detection (NEED) method computes the motion in gaze and head in conjunction with similarity of content in the central visual field and feeds them as input to a classification algorithm to detect gaze fixation, gaze following, gaze pursuit, and gaze shift.

![events](./figs/events.jpg)

The a
## Execution
The code has been executed and tested on UBUNTU 18.06. 
For running the code you need:

1. PyTorch
2. OpenCV
3. Python Libraries: numpy, matplotlib, scipy 
4. 2ch2stream Network (included in repo)
5. (OPTIONAL) MATLAB for some data inspection scripts

To execute the code, edit the `config.py` file and select your dataset and input directory and the rest of params. The exectute 'main.py'.

## Citing Us


## Remark and Acknowledgement
This code has been developed at the department of Research and Improvement of Care, Royal Dutch Visio by Ashkan Nejad.

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No 955590.

Parts of this repository are extracted from other repositories provided publicly by their authors. We would like to thank the all of researchers who helped us by providing the impelementation of their work to the public:
1. https://github.com/elmadjian/OEMC
2. https://github.com/Huangying-Zhan/DF-VO
3. https://github.com/szagoruyko/cvpr15deepcompare
