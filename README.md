# MachineLearning with SKLearn

Project using the scikit-learn machine learning libraries to achieve static analysis and determination on a raw dataset provided by UNB.
Application uses both Decision Trees and Neural Network models to determine results. Results go through a confussion matrix and showcases the prediction accuracy aswell.
Pre-processing of the data has been applied on the raw datasets so that valid data passes through (data consisting integer values).
Memory optimization has also been applied so that the file can be read and used using the lowest data types possible.

## Static_ML file Contents (.py files)

- **Dict.py** contains all dictionary arrays that where used for pre-processing to convert non-integer values to an integer base number
- **functions.py** contains all methods used to perform machine learning, from reading the file, performing optimization, predictions, accuracy and analysis is done here.
- **main.py** as the name says it, this is the main file where the program is ran from. Asking user questions and retrieving/printing the results come from this class.

## Data File

- Provided data comes from the **DataFiles** folder that hosts all *.csv* files that have been merged into one *combinedData.csv* file which is pre-processed and used for the static Machine Learning Analysis
