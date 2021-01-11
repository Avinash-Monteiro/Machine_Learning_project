# Building a Machine Learning Classifier Model for Diabetes

## Based on medical diagnostic measurements

![Image for post](https://miro.medium.com/max/952/1*W3MzYYTx8FUlxkJZ19rXKw.png)

The objective of this project is to build a predictive machine learning model to predict based on diagnostic measurements whether a patient has diabetes. This is a **binary (2-class)** classification project with supervised learning.


![Image for post](https://miro.medium.com/max/1033/1*wWRGcsBwoXw-WNJi5UpoOw.png)


## Step 1: Import relevant libraries

Standard libraries of Pandas and Numpy are imported, along with visualisation libraries of Matplotlib and Seaborn. There are also a host of models and measurement metrics imported from Scikit-Learn library.

## Step 2: Read in data, perform Exploratory Data Analysis (EDA)

Use Pandas to read the csv file “diabetes.csv”. There are 768 observations with 8 medical predictor features (input) and 1 target variable (output 0 for ”no diabetes” or 1 for ”yes”). Let’s check the target variable distribution:

![Image for post](https://miro.medium.com/max/434/1*iJNuWPTyzS-VySBgqARo4Q.png)

35% of the patients have diabetes (Output class 1)

The 8 medical predictor features are:  
·  **Pregnancies**: Number of times pregnant  
·  **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test  
·  **BloodPressure**: Diastolic blood pressure (mm Hg)  
·  **SkinThickness**: Triceps skin fold thickness (mm)  
·  **Insulin**: 2-Hour serum insulin (mu U/ml)  
·  **BMI**: Body Mass Index (weight in kg/(height in m)²)  
·  **DiabetesPedigreeFunction**: Diabetes pedigree function on genetic influence and hereditary risk  
·  **Age**: Age (years)  
Let’s visualise the distribution of these 8 features.

![Image for post](https://miro.medium.com/max/901/1*v_DrZeEpD4XWiQj2uAHhxg.png)

There are some zeros in the data set, which will affect the training accuracy. I have chosen to replace them with the median value.

![Image for post](https://miro.medium.com/max/675/1*jH8LU-nV4LVkExBma99tbg.png)

Zeros replaced with the median value

> Tip: Replacing zeros with median is a two-step process: first find out the mean of the feature when the class is true and also when it is False.
> replace zero by mean of the particular class.

After that we can modify the codes to visualise the relative positions of the 2 classes.



![download image 1.png](https://github.com/Avinash-Monteiro/Machine_Learning_project/blob/main/Diabetes_prediction/download%20image%201.png?raw=true)

The distribution of Target Variable is as under;
![TARGETVARIABLE.png](https://github.com/Avinash-Monteiro/Machine_Learning_project/blob/main/Diabetes_prediction/TARGETVARIABLE.png?raw=true)
After developing lot of Machine Learning Models from Basic Models, Hyper-parameter models, Linear Discriminant Analysis and also Ensemble Techniques, we finally found that Stacking Model gave the best results for predicting diabetes only from Diagnostic Measures.

![fianl comparision.png](https://github.com/Avinash-Monteiro/Machine_Learning_project/blob/main/Diabetes_prediction/fianl%20comparision.png?raw=true)

The confusion Matrix from Stacking Classifier is as follows;


![Confusion Matrix.png](https://github.com/Avinash-Monteiro/Machine_Learning_project/blob/main/Diabetes_prediction/Confusion%20Matrix.png?raw=true)



