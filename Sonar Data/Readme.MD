
# Logistic Regression in Python

![Logistic Regression in Machine Learning](https://www.mindsmapped.com/wp-content/uploads/2020/06/Logistic-Regression-in-Python.png)

### What is Classification?
**Classification**  is among the most important areas of machine learning, and  **logistic regression**  is one of its basic methods. 

## Classification[](https://realpython.com/logistic-regression-python/#classification "Permanent link")

[Classification](https://en.wikipedia.org/wiki/Statistical_classification)  is a very important area of  [supervised machine learning](https://en.wikipedia.org/wiki/Supervised_learning). A large number of important machine learning problems fall within this area. There are many classification methods, and logistic regression is one of them.


### What Is Classification?

Supervised machine learning algorithms define models that capture relationships among data.  **Classification**  is an area of supervised machine learning that tries to predict which class or category some entity belongs to, based on its features.

For example, you might analyze the employees of some company and try to establish a dependence on the  **features**  or  **variables**, such as the level of education, number of years in a current position, age, salary, odds for being promoted, and so on. The set of data related to a single employee is one  **observation**. The features or  variables can take one of two forms:

1.  **Independent variables**, also called inputs or predictors, don’t depend on other features of interest (or at least you assume so for the purpose of the analysis).
2.  **Dependent variables**, also called outputs or responses, depend on the independent variables.

In the above example where you’re analyzing employees, you might presume the level of education, time in a current position, and age as being mutually independent, and consider them as the inputs. The salary and the odds for promotion could be the outputs that depend on the inputs.

The nature of the dependent variables differentiates regressionand classification problems.  **Regression**  problems have continuous and usually unbounded outputs. An example is when you’re estimating the salary as a function of experience and education level. On the other hand,  **classification**  problems have discrete and finite outputs called  **classes**  or  **categories**. For example, predicting if an employee is going to be promoted or not (true or false) is a classification problem.

There are two main types of classification problems:

1.  **Binary**  or  **binomial classification:**  exactly two classes to choose between (usually 0 and 1, true and false, or positive and negative)
2.  **Multiclass**  or  **multinomial classification:**  three or more classes of the outputs to choose from

If there’s only one input variable, then it’s usually denoted with 𝑥. For more than one input, you’ll commonly see the vector notation 𝐱 = (𝑥₁, …, 𝑥ᵣ), where 𝑟 is the number of the predictors (or independent features). The output variable is often denoted with 𝑦 and takes the values 0 or 1.

### When Do You Need Classification?
You can apply classification in many fields of science and technology. For example, text classification algorithms are used to separate legitimate and spam emails, as well as positive and negative comments. Other examples involve medical applications, biological classification, credit scoring, and more.

Image recognition tasks are often represented as classification problems. For example, you might ask if an image is depicting a human face or not, or if it’s a mouse or an elephant, or which digit from zero to nine it represents, and so on. 

## Logistic Regression Overview[](https://realpython.com/logistic-regression-python/#logistic-regression-overview "Permanent link")

Logistic regression is a fundamental classification technique. It belongs to the group of  **linear classifiers**  and is somewhat similar to polynomial and  **linear regression**. Logistic regression is fast and relatively uncomplicated, and it’s convenient for you to interpret the results. Although it’s essentially a method for binary classification, it can also be applied to multiclass problems.

### Math Prerequisites[](https://realpython.com/logistic-regression-python/#logistic-regression-overview "Permanent link")


You’ll need an understanding of the  sigmoid function and the  natural logarithm function to understand what logistic regression is and how it works.

This image shows the sigmoid function (or S-shaped curve) of some variable 𝑥:

[![Sigmoid Function](https://files.realpython.com/media/log-reg-1.e32deaa7cbac.png)](https://files.realpython.com/media/log-reg-1.e32deaa7cbac.png)

The sigmoid function has values very close to either 0 or 1 across most of its domain. This fact makes it suitable for application in classification methods.

### Problem Formulation

In this tutorial, you’ll see an explanation for the common case of logistic regression applied to binary classification. When you’re implementing the logistic regression of some dependent variable 𝑦 on the set of independent variables 𝐱 = (𝑥₁, …, 𝑥ᵣ), where 𝑟 is the number of predictors ( or inputs), you start with the known values of the predictors 𝐱ᵢ and the corresponding actual response (or output) 𝑦ᵢ for each observation 𝑖 = 1, …, 𝑛.

Your goal is to find the  **logistic regression function**  𝑝(𝐱) such that the  **predicted responses**  𝑝(𝐱ᵢ) are as close as possible to the  **actual response**  𝑦ᵢ for each observation 𝑖 = 1, …, 𝑛. Remember that the actual response can be only 0 or 1 in binary classification problems! This means that each 𝑝(𝐱ᵢ) should be close to either 0 or 1. That’s why it’s convenient to use the sigmoid function.

Once you have the logistic regression function 𝑝(𝐱), you can use it to predict the outputs for new and unseen inputs, assuming that the underlying mathematical dependence is unchanged.

### Methodology
Logistic regression is a linear classifier, so you’ll use a linear function 𝑓(𝐱) = 𝑏₀ + 𝑏₁𝑥₁ + ⋯ + 𝑏ᵣ𝑥ᵣ, also called the  **logit**. The variables 𝑏₀, 𝑏₁, …, 𝑏ᵣ are the  **estimators**  of the regression coefficients, which are also called the   **coefficients**.

The logistic regression function 𝑝(𝐱) is the sigmoid function of 𝑓(𝐱): 𝑝(𝐱) = 1 / (1 + exp(−𝑓(𝐱)). As such, it’s often close to either 0 or 1. The function 𝑝(𝐱) is often interpreted as the predicted probability that the output for a given 𝐱 is equal to 1. Therefore, 1 − 𝑝(𝑥) is the probability that the output is 0.

Logistic regression determines the best predicted weights 𝑏₀, 𝑏₁, …, 𝑏ᵣ such that the function 𝑝(𝐱) is as close as possible to all actual responses 𝑦ᵢ, 𝑖 = 1, …, 𝑛, where 𝑛 is the number of observations. The process of calculating the best weights using available observations is called  **model training**  or  **fitting**.

To get the best weights, you usually maximize the  **log-likelihood function (LLF)**  for all observations 𝑖 = 1, …, 𝑛. This method is called the  **maximum likelihood estimation**  and is represented by the equation LLF = Σᵢ(𝑦ᵢ log(𝑝(𝐱ᵢ)) + (1 − 𝑦ᵢ) log(1 − 𝑝(𝐱ᵢ))).


### Classification Performance

Binary classification has four possible  types of results:

1.  **True negatives:**  correctly predicted negatives (zeros)
2.  **True positives:**  correctly predicted positives (ones)
3.  **False negatives:**  incorrectly predicted negatives (zeros)
4.  **False positives:**  incorrectly predicted positives (ones)

You usually evaluate the performance of your classifier by comparing the actual and predicted outputs and counting the correct and incorrect predictions.

The most straightforward indicator of  **classification accuracy**  is the ratio of the number of correct predictions to the total number of predictions (or observations). Other indicators of binary classifiers include the following:

-   **The  positive predictive value**  is the ratio of the number of true positives to the sum of the numbers of true and false positives.
-   **The  negative predictive value)**  is the ratio of the number of true negatives to the sum of the numbers of true and false negatives.
-   **The  sensitivity**  (also known as recall or true positive rate) is the ratio of the number of true positives to the number of actual positives.
-   **The  specificity**  (or true negative rate) is the ratio of the number of true negatives to the number of actual negatives.

The most suitable indicator depends on the problem of interest. 
