# The Machine Learning Efficient PHE Bus Operation Model, ML-EPBO

Source code and data to use the Neuroevolved Bi-Directional LSTM Applied to Zero Emissions Zones Management in Urban Transport
  
## Usage
### Binary files
There are two binary files to execute the training software inside the GA directory:

  * **main**. Unix type system binary file.
  * **main.exe**. Windows type system executable file.

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

## Contact

Any problems or doubts with the source code contact me at **josemiguel.aragon@uca.es**.
