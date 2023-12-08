# Team NPC - Traffic car predictions

Welcome to Team NPC's Traffic Deep Learning Model project! In this project, we aim to predict and analyze traffic incidents using deep learning techniques. Our model utilizes data from the Car Crashes in Chicago dataset, which you can find [here](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if/explore).

## Project Components

### 1. Preprocessing (preprocess.py)
The `preprocess.py` script is responsible for handling the preprocessing of the input data. It takes the Car Crashes in Chicago dataset as input thanks to an API and prepares it for training the deep learning model. This includes data cleaning, feature engineering, and any necessary transformations to ensure the data is in a suitable format for training.

### 2. Ensemble Model (ensembleModel.py)
The `ensembleModel.py` script implements the ensemble learning approach using Voting Classifiers. We leverage MLP model, GBM model and lasso models to create a strong, robust prediction model. The ensemble model combines the predictions of individual models, enhancing the overall performance and accuracy of the traffic incident prediction.

### 3. Output and Visualization (output.py)
The `output.py` script handles the final steps of our model pipeline. It interfaces with the weather APIs to retrieve real-time weather data. Additionally, the script generates a visual map illustrating the predicted traffic incidents based on the model's output. The map display the real-time prediction.

## How to Run the Project

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/traffic-deep-learning-model.git
   cd traffic-deep-learning-model
   
2. Install the required dependencies:

    ```bash
   pip install -r requirements.txt

3. Get the current prediction in real-time as a map:
    ```bash
    python output.py

4. You can get the accuracy with 10k data and the top 3 predictions counted as good result:

    ```bash
    python ensembleModel.py