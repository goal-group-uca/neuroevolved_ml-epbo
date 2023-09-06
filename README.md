# The Machine Learning Efficient PHE Bus Operation Model, ML-EPBO

Source code and data to use the Neuroevolved Bi-Directional LSTM Applied to Zero Emissions Zones Management in Urban Transport
  
## Usage
### Binary
Binary files can be downloaded from the following link:

https://ucaes-my.sharepoint.com/:u:/g/personal/josemiguel_aragon_uca_es/Eezfabc_u0dEpwNkJ1BCekwB-jIFLOVEzNabSRsYYzmzxA?e=1uhL1K

There are three different precompiled software versions:

  * **ubuntu**. Precompiled software version for Ubuntu-like Operating Systems.
  * **windows**. Precompiled software version for Windows Operating System.
  * **archlinux**. Precompiled software version for ArchLinux-like Operating Systems.

Running the file will bring up a user interface where you can edit different parameters of the LSTM training. The training will start when the main button is pressed, saving the results of the genetics in a file called results.data at the end.

### Bash scripts
There are two bash scripts in the base directory (requires Python 3 previously installed):

  * ***launch_experiment.sh***. Launch 30 GA runs to train the LSTM.
  * ***launch_one_exec.sh***. Launch only one GA run to train the LSTM.
    
### Manual usage and installation
First of all, you need to install Python 3 and the project dependencies using pip package manager. You can employ the following command:

  ``pip install -r requirements.txt``

Then, you can launch the GA used to train the LSTM running the following command inside the GA directory:

``python3 LSTMGAMain.py``

Furthermore, you can modify the some parameters of the GA configuration inside the LSTMGAMain.py file such as the population size or the minimum number of epochs among others.
