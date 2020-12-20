FILE SETUP
----------
To run the file download the Training.zip file from this link:
https://drive.google.com/file/d/1O2qtFslP2SZ385SCn8BraasXOec05PlV/view?usp=sharing

Extract the file in the project root.

The project root should contain the following:
Plots/
Training/
covid_classifier.py
model
README.md




RUNNING THE CODE
----------------
If you want the program to run using the existing model saved at ./model
then set the "LOAD" variable on line 18 to "true".

If you want the program to train a new model and run against that new model
then set the "LOAD" variable on line 18 to "false". Note that this new model
will be saved and will overwrite any existing model saved at ./model

Run the code by running covid_classifier.py with the proper modules installed.