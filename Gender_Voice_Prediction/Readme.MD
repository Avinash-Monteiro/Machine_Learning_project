# Gender Voice Prediction
In this project we see how to develop Various Machine Learning model which could predict the gender of a person, either male or female.

Basically this is a Classification dataset with 30 numerical independent features and one target output.
For easy model building and analysis, we have given the rank 1 for male and rank 0 for female voices.
After developing various models and analyzing them through the concept of **AUC-ROC** we came to know that Support Vector Machines Algorithm is the best algorithm which sits this dataset.
Here is a brief about Support Vector Machines;

### The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.
To separate the two classes of data points, there are many possible hyperplanes that could be chosen.
![Image for post](https://miro.medium.com/max/300/0*9jEWNXTAao7phK-5.png)

Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.
![Image for post](https://miro.medium.com/max/300/0*0o8xIA4k3gXUDCFU.png)

Hyperplanes are decision boundaries that help classify the data points. Data points falling on either side of the hyperplane can be attributed to different classes. Also, the dimension of the hyperplane depends upon the number of features. If the number of input features is 2, then the hyperplane is just a line. If the number of input features is 3, then the hyperplane becomes a two-dimensional plane. It becomes difficult to imagine when the number of features exceeds 3.
![Image for post](https://miro.medium.com/max/1418/1*ZpkLQf2FNfzfH4HXeMw4MQ.png)
Support vectors are data points that are closer to the hyperplane and influence the position and orientation of the hyperplane. Using these support vectors, we maximize the margin of the classifier. Deleting the support vectors will change the position of the hyperplane. These are the points that help us build our SVM.
![Image for post](https://miro.medium.com/max/755/0*ecA4Ls8kBYSM5nza.jpg)

# Model Evaluation using AUC-ROC technique:
In Machine Learning, performance measurement is an essential task. So when it comes to a classification problem, we can count on an AUC - ROC Curve. When we need to check or visualize the performance of the multi - class classification problem, we use AUC (**Area Under The Curve**) ROC (**Receiver Operating Characteristics**) curve. It is one of the most important evaluation metrics for checking any classification model’s performance. It is also written as AUROC (**Area Under the**  **Receiver Operating Characteristics**)

# What is AUC - ROC Curve?

AUC - ROC curve is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents degree or measure of separability.  It tells how much model is capable of distinguishing between classes.  Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. By analogy, Higher the AUC, better the model is at distinguishing between patients with disease and no disease.

The ROC curve is plotted with TPR against the FPR where TPR is on y-axis and FPR is on the x-axis.

![Image for post](https://miro.medium.com/max/361/1*pk05QGzoWhCgRiiFbz-oKQ.png)
# **Defining terms used in AUC and ROC Curve.**

## **TPR (True Positive Rate) / Recall /Sensitivity**


![Image for post](https://miro.medium.com/max/355/1*HgxNKuUwXk9JHYBCt_KZNw.png)

## **Specificity**

![Image for post](https://miro.medium.com/max/246/1*f7NmMcQtfes1ng7jtjNtHQ.png)


## False Positive Rate, FPR

![Image for post](https://miro.medium.com/max/245/1*3GhDfiuhvINF5-9eL8g6Pw.png)


# **How to speculate the performance of the model?**

An excellent model has AUC near to the 1 which means it has good measure of separability. A poor model has AUC near to the 0 which means it has worst measure of separability. 
In fact it means it is reciprocating the result. It is predicting 0s as 1s and 1s as 0s. And when AUC is 0.5, it means model has no class separation capacity whatsoever.

Let’s interpret above statements.

As we know, ROC is a curve of probability. So lets plot the distributions of those probabilities:

Note: Red distribution curve is of the positive class (patients with disease) and green distribution curve is of negative class(patients with no disease).


![Image for post](https://miro.medium.com/max/528/1*Uu-t4pOotRQFoyrfqEvIEg.png)

![Image for post](https://miro.medium.com/max/365/1*HmVIhSKznoW8tFsCLeQjRw.png)

This is an ideal situation. When two curves don’t overlap at all means model has an ideal measure of separability. It is perfectly able to distinguish between positive class and negative class.

![Image for post](https://miro.medium.com/max/507/1*yF8hvKR9eNfqqej2JnVKzg.png)

![Image for post](https://miro.medium.com/max/365/1*-tPXUvvNIZDbqXP0qqYNuQ.png)

When two distributions overlap, we introduce type 1 and type 2 error. Depending upon the threshold, we can minimize or maximize them. When AUC is 0.7, it means there is 70% chance that model will be able to distinguish between positive class and negative class.

![Image for post](https://miro.medium.com/max/430/1*iLW_BrJZRI0UZSflfMrmZQ.png)
![Image for post](https://miro.medium.com/max/365/1*k_MPO2Q9bLNH9k4Wlk6v_g.png)

This is the worst situation. When AUC is approximately 0.5, model has no discrimination capacity to distinguish between positive class and negative class.

![Image for post](https://miro.medium.com/max/556/1*aUZ7H-Lw74KSucoLlj1pgw.png)
![Image for post](https://miro.medium.com/max/365/1*H7JGQbaa06BUab6tvGNZKg.png)

When AUC is approximately 0, model is actually reciprocating the classes. It means, model is predicting negative class as a positive class and vice versa.

# **Relation between Sensitivity, Specificity, FPR and Threshold.**

Sensitivity and Specificity are inversely proportional to each other. So when we increase Sensitivity, Specificity decreases and vice versa.

> Sensitivity⬆️, Specificity⬇️ and Sensitivity⬇️, Specificity⬆️

When we decrease the threshold, we get more positive values thus it increases the sensitivity and decreasing the specificity.

Similarly, when we increase the threshold, we get more negative values thus we get higher specificity and lower sensitivity.

As we know FPR is 1 - specificity. So when we increase TPR, FPR also increases and vice versa.

> TPR⬆️, FPR⬆️ and TPR⬇️, FPR⬇️

# **How to use AUC ROC curve for multi-class model?**

In multi-class model, we can plot N number of AUC ROC Curves for N number classes using One vs ALL methodology. So for Example, If you have  **three**  classes named  **X, Y** and **Z**, you will have one ROC for X classified against Y and Z, another ROC for Y classified against X and Z, and a third one of Z classified against Y and X.

Thanks for Reading
