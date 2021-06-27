# Type 2 Diabetes Risk Prediction
## Problem Statment
Type 2 diabetes is a chronic condition that affects the way the body metabolizes sugar (glucose). With type 2 diabetes, the body either resists the effects of insulin (a hormone that regulates the movement of sugar into cells) or it doesn't produce enough insulin to maintain normal glucose levels. Type 2 diabetes occurs more commonly in middle-aged and elderly people. Uncontrolled it can cause all sorts of very bad things: infections, damaged kidneys, vision loss and blindness, amputations and many more. So, there is no question that type 2 diabetes needs to be taken seriously and treated. Type 2 diabetes is usually diagnosed using the glycated hemoglobin (A1C) test. This blood test indicates the average blood sugar level for the past two to three months. Normal levels are below 5.7 percent, and a result between 5.7 and 6.4 percent is considered prediabetes. An A1C level of 6.5 percent or higher on two separate tests means you have diabetes.  People who have diabetes need this test regularly to see if their levels are staying within range and if they need to adjust their diabetes medicines. To treat type 2 diabetes lifestyle changes are very effective, and the side effects of eating more healthfully and staying more active are positive ones. In this project I will try to predict A1C levels: no-diabetes, pre-diabetes and diabetes. I will transform the dataset from a regression task (A1C) into a multi-class classification task (3 A1C levels).

<span style="display:block;text-align:center">![png](A1c_normal_to_high_ranges.png)</span>

## Challenges

I was facing two challenegs with my dataset, the relatively small number of observations, and the imbalanced classes (A1C levels). To overcome the issues with imbalanced data, I use the following techniques:
-   F1 macro averaged score for performance metric
-   Cost-sensitive learning (penalize algorithms)
-   SMOTE - Synthetic Minority Over-sampling Technique

My goal is to predict the risk with the highest accuracy possible with the relatively small data set we have, for that I train and compare the accuray of the following machine learning models:
- L_1-regularized Logistic Regression
- L_2-regularized Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- AdaBoost
All together, In this project I have trained 22 models.

## Result Reproduction

**Diabetes Dataset**

These data are courtesy of Dr John Schorling, Department of Medicine, University of Virginia School of Medicine which can be obtained from http://biostat.mc.vanderbilt.edu/DataSets.

The data consist of 19 variables on 403 subjects from 1046 subjects who were interviewed in a study to understand the prevalence of obesity, diabetes, and other cardiovascular risk factors in central Virginia for African Americans. According to Dr John Hong, Diabetes Mellitus Type II (adult onset diabetes) is associated most strongly with obesity. The waist/hip ratio may be a predictor in diabetes and heart disease. Type 2 Diabetes is also associated with hypertension - they may both be part of Metabolic Syndrome.

**Metabolic syndrome** is a collection of risk factors that includes high blood pressure, high blood sugar, excess body fat around the waist, and abnormal cholesterol levels. The syndrome increases the chance of developing heart disease, stroke, and diabetes. Aside from a large waist circumference, most of the disorders associated with metabolic syndrome have no symptoms. Losing weight, exercise, and dietary changes can help prevent or reverse metabolic syndrome. According to a national health survey, more than 1 in 5 Americans has metabolic syndrome. The number of people with metabolic syndrome increases with age, affecting more than 40% of people in their 60s and 70s.

The 403 subjects were the ones who were actually screened for diabetes. Glycosolated hemoglobin (A1C) > 7.0 is usually taken as a positive diagnosis of diabetes.

**Code and article** can be found on this github repo and on https://mohcinemadkour.github.io/posts/2019/07/Machine%20Learning,%20July%202019,%20Risk%20prediction/

## Interpretation of Performance Measures

As a classification problem, you can get the following from the confusion matrix: TR,FN, FP, TN. From these measures you can calculate Accuracy, Precision, Recall and F1 score:
  - **Accuracy** - Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations (TP+TN/TP+FP+FN+TN). Accuracy is a great measure but only when you have symmetric datasets where values of false positive and false negatives are almost same. 
  - **Precision** - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations (TP/TP+FP). The question that this metric answer is of all patients that labeled as diabetics, how many are actually diabetics? High precision relates to the low false positive rate.
  - **Recall (Sensitivity)** - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes (TP/TP+FN). The question recall answers is: Of all the patients that are truly diabetic.
  - **F1 score** - F1 Score is the weighted average of Precision and Recall (2*(Recall * Precision) / (Recall + Precision)). Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution which is our case. F1 Score works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall. 

## Insights

My dataset was relatively small and imbalanced and I had to employ several techniques for handling imbalanced classes:

- f1 macro averaged score for performance metri
- cost-sensitive learning (penalize algorithms)
- SMOTE - Synthetic Minority Over-sampling Technique

I have used six machine learning algorithms: $L_1$ and $L_2$ regularized Logistic Regressions, SVM and three tree ensembles, Random Forest, Gradient Boost and AdaBoost.

In total, I have trained 22 models.

  - Plain models, without any of the above listed techniques, did prety bad with predicting minority classes. They mostly predicted the majority class. Because of that, their accuracy score was high, but f1-macro score was low. As expected, tree ensembles models, were performed slightly better.
  - All three techniques listed above, made a positive difference. Again, tree ensemble models produced better performance.
  - I could not find one single health condition that could alone increase the risk of being diagnosed with type 2 diabetes.
  - It looks that they are working differently for different people.
  - From my limited sample, I could conclude that the most contributing factors were age, cholesterol ratio and waist cirumference.

The **Random Forest** model with cost-sensitive learning have produced the best performance:

  - prety good on the majority class - recall=0.83
  - on the diabetes class not so bad - recall=0.58
  - not so good on the smallest, pre-diabetes, class - recall=0.29

It is interesting to note the following regarding **predicting pre-diabetes**:

  - there were only 7 labels in the test set
  - recall=0.29 means 2 successfully predicted labels and in addition to my winning model, only 3 models had this score: Gradient Boost, Random Forest and SVM all with SMOTE
  - only 2 models succeded in 3 positive prediction, recall=0.43. Surprisingly, that was $L_2$-regularized Logistic Regression with SMOTE and cost-sensitive learning.

