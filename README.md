# Color-Segmentation-using-GMM-and-EM

# Set of training images:
The folders “green_train”, “yellow_train”, “orange_train”  

# Files included:
1.	takeimage.py : Code to cut out buoys from frames to generate train set.
2.	1D_gauss.py : Code to generate random gaussian data samples and find means and standard deviations using EM.
3.	1D_gauss_green.py: Code to detect green buoy using 1D gaussian on green channel.
4.	1D_gauss_yellow.py: Code to detect yellow buoy using 1D gaussian on green and red channels.
5.	1D_gauss_orange.py: Code to detect orange buoy using 1D gaussian on red channel.
6.	3D_gauss_green.py :  Code to detect green buoy using 3D gaussian on all RGB channels
7.	3D_gauss_yellow.py :  Code to detect yellow buoy using 3D gaussian on all RGB channels
8.	3D_gauss_orange.py :  Code to detect orange buoy using 3D gaussian on all RGB channels
9.	3D_gauss_all.py :  Code to detect all buoys using 3D gaussian on all RGB channels 

# Videos included:
1.	detectbuoy.avi :The video we work with
2.	1D_gauss_green.avi : 1D_gauss_green output
3.	1D_gauss_yellow.avi : 1D_gauss_yellow output
4.	1D_gauss_orange.avi : 1D_gauss_orange output
5.	3D_gauss_green.avi : 3D_gauss_green output
6.	3D_gauss_yellow.avi : 3D_gauss_yellow output
7.	3D_gauss_orange.avi : 3D_gauss_orange output
8.	3D_gauss_all.avi : 3D_gauss_all output 

The python files must be kept in the same folder as the folders containing the training images. Each file can then be run from the command line.
