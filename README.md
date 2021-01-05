# Hearth Failure prediction. Azure ML Example

This project is part of the Udacity Azure ML Nanodegree. Here, we use both the Hyperdrive and Automl API from Azureml to build a Classification Model using Kaggle's Heart Failure Prediction and finally deploy the best model as a Webservice:

![Diagram](https://github.com/Ulizeuz/nd00333-capstone/blob/main/ScreenShots/capstone_04_Diagram.png) 

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
We can observe task is **Classification**, with **AUC_weighted** as primary metric and the target we want to find is the column **DEATH_EVENT**. The train_data needs to be TabularDataset type. We set **featurization** as auto to do this step automatically. Finally, we enable **early stopping** to avoid overfitting. 

Another important configuration settings that impact the training process are **experiment_timeout_minutes** and **max_concurrent_iterations**. The first define the maximum amount of time in minutes that all iterations combined can take before the experiment terminates and the second represents the maximun number of iterations that would be executed in parallel.

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

### Consume model deployed

Once deploying the model as a web service a REST API endpoint is created. We can send data to this endpoint and receive the prediction returned by the model. This example demonstrates how to use Python to call the web service created:

```
import requests
import json

scoring_uri = '<Update web service URI>'
headers = {'Content-Type':'application/json'}

test_data = json.dumps({'data':[{
    'age':75,
    'anaemia':0,
    'creatinine_phosphokinase':582,
    'diabetes':0,
    'ejection_fraction':20,
    'high_blood_pressure':1,
    'platelets':265000,
    'serum_creatinine':1.9,
    'serum_sodium':130,
    'sex':1,
    'smoking':0,
    'time':4}
    ]
        })

response = requests.post(scoring_uri, data=test_data, headers=headers)

print("Result:",response.text)

```
The result returned is similar to the following:

```
Result: [0]
```

Where **[0]** is the negative prediction of Death Event and the positive prediction is **[1]** 

## Screen Recording

You can find in the next video how to consume this model in a simple example
https://youtu.be/ZYDZ9tBgIZk

## Standout Suggestions

There are some improvement that I want to do as a next version

- Convert the model to ONNX format. 
- Deploy the model to the Edge using Azure IoT Edge. 
- Enable logging in the deployed web app.
- Deploy Swagger server

On the other hand, I'd like to improve the model in the future: firstly, trying changing the primary metric like death probability, becaming a continuous model. Another thing is add more data, the Kaggle dataset is good to practice, but has few data. Finally, I want to improve the hyperdrive experiment using another models, it's a good way to get experience tunning models and in some cases with better results than AutoML

