# Hearth Failure prediction. Azure ML Example

This project is part of the Udacity Azure ML Nanodegree. Here, we use both the Hyperdrive and Automl API from Azureml to build a Classification Model using Kaggle's Heart Failure Prediction and finally deploy the best model as a Webservice.

## Project Set Up and Installation

The starter files that you need to run this project are the following:
- **automl.ipynb**: Jupyter Notebook to run the autoML experiment
- **hyperparameter_tuning.ipynb**: Jupyter Notebook to run the Hyperdrive experiment
- **train.py**. Script used in Hyperdrive
- **score.py**. Script used to deploy the model
- **heart_failure_clinical_records_dataset.csv**. The dataset

## Dataset

### Overview

This dataset can be found in https://www.kaggle.com/andrewmvd/heart-failure-clinical-data and, according to this page, Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.

### Task

The dataset contains 12 features that can be used to predict mortality by heart failure. People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

### Access

We use 2 ways to show how access the data in the workspace:

1. In AutoML we use read_csv() Pandas function to get file locally.

2. For Hyperdrive, we use Dataset.Tabular.from_delimited_files() in the train script to get the file with URL.

## Automated ML

Following there are the setting and configuration used for this experiment:

```

automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'AUC_weighted'    
}

automl_config = AutoMLConfig(
    
    compute_target = compute_target,
    task = "classification",
    training_data = train_data,
    label_column_name = "DEATH_EVENT",   
    path = project_folder,
    enable_early_stopping= True,
    featurization= 'auto',
    debug_log = "automl_errors.log",
    **automl_settings 
)

```
We can observe task is **Classification**, with **AUC_weighted** as primary metric and the target we want to find is the column **DEATH_EVENT**. The train_data needs to be TabularDataset type. We set featurization as **auto**  to do this step automatically. Finally, we enable **early stopping** to avoid overfitting. 

### Results

The best model was VotingEnsemble with 0.9252.

![Run Details1](https://github.com/Ulizeuz/nd00333-capstone/blob/main/ScreenShots/capstone_01_AutoML1.png)
![Run Details2](https://github.com/Ulizeuz/nd00333-capstone/blob/main/ScreenShots/capstone_01_AutoML2.png)
![Run Details3](https://github.com/Ulizeuz/nd00333-capstone/blob/main/ScreenShots/capstone_01_AutoML3.png)
![Run Details4](https://github.com/Ulizeuz/nd00333-capstone/blob/main/ScreenShots/capstone_01_AutoML4.png)
![Run Details5](https://github.com/Ulizeuz/nd00333-capstone/blob/main/ScreenShots/capstone_01_AutoML5.png)
![Run Details6](https://github.com/Ulizeuz/nd00333-capstone/blob/main/ScreenShots/capstone_01_AutoML6.png)
![Run Details7](https://github.com/Ulizeuz/nd00333-capstone/blob/main/ScreenShots/capstone_01_AutoML7.png)
![Run Details8](https://github.com/Ulizeuz/nd00333-capstone/blob/main/ScreenShots/capstone_01_AutoML8.png)

And the following are the parameters of the best model trained:

![Parameters](https://github.com/Ulizeuz/nd00333-capstone/blob/main/ScreenShots/capstone_02_AutoML1.png)
![Parameters](https://github.com/Ulizeuz/nd00333-capstone/blob/main/ScreenShots/capstone_02_AutoML2.png)


## Hyperparameter Tuning

On the other experiment, we use Logistic Regression from SciKit Learn because it's a well-known model to use for hyperparameter Tuning. The configuration was shown below:

```
early_termination_policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

param_sampling = RandomParameterSampling({
    '--C': choice(0.5, 1.5),
    'max_iter': choice( 50, 150)
})

src = ScriptRunConfig(source_directory='./',
                      script='train.py',
                      compute_target = compute_target,
                      environment=sklearn_env)

hyperdrive_run_config = HyperDriveConfig(
    run_config=src,
    hyperparameter_sampling=param_sampling,
    policy=early_termination_policy,
    primary_metric_name = "Accuracy",
    primary_metric_goal = PrimaryMetricGoal.MAXIMIZE,
    max_total_runs = 100,
    max_concurrent_runs = 2)
    
```

We define a **Early Policy**, based on slack factor and evaluation interval. We use this policy to terminate runs if the primary metric is non within the slack factor contrast with best run. We choose a 0.1 slack factor, but if we want to have more aggresive savings it could be smaller. In the same way, we could opt to small evaluation_interval, but we decide to choose the value 2 to evaluate every 2 runs.

For **Sampling** we use RandomParameterSampling in spite of grid sweep is exhaustive but consumes more time, whereas random sweep can get a good results without taking as much time, reducing computation cost. The parameters values are chosen with the function choice and we select values around the default values for C (Inverse of regularization strength) and max_iter (Maximum number of iterations to converge)

Finally, apply Scikit-Learn model to fit the training data and compute the accuracy.

### Results

At the end of the run, we get the score of 0.85 for our primary metric, with 0.5 value for --C and --150 for max_iter as you can see in the next Screenshots:

![Run_details_HD](https://github.com/Ulizeuz/nd00333-capstone/blob/main/ScreenShots/capstone_03_HD1.png)

![Params_HD](https://github.com/Ulizeuz/nd00333-capstone/blob/main/ScreenShots/capstone_03_HD2.png)



## Model Deployment

We decided deloy the best model we get with AutoML, it was VotingEnsemble, for that after saving the model we need to:

- Create a scoring script in order to answer the request to the web service. This script must have the init() and run(input_data) functions.

- Define the inference and the deployment configuration

- And finally create the enviroment for the deployment where Azure Machine Learning can install the necessary packages

## Screen Recording

You can find in the next video how to consume this model in a simple example
https://youtu.be/ZYDZ9tBgIZk

## Standout Suggestions

There are some improvement that I want to do as a next version

- Convert the model to ONNX format. 
- Deploy the model to the Edge using Azure IoT Edge. 
- Enable logging in the deployed web app.
