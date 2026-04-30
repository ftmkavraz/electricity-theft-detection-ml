\# Electricity Theft Detection Using Machine Learning



<p align="center">

&#x20; <img src="images/scatter\_plot.png" width="700"/>

</p>



\## 1. Introduction

Electricity theft is a significant issue in power distribution systems, leading to substantial financial losses and operational inefficiencies. Traditional detection methods are often insufficient due to the complexity and variability of consumption patterns.



This project proposes a machine learning-based approach to detect electricity theft by analyzing consumption data and identifying anomalous patterns.



\---



\## 2. Objective

The main objective of this project is to develop a predictive model capable of distinguishing between normal and fraudulent electricity consumption behaviors.



Specifically:

\- To preprocess and analyze electricity consumption data  

\- To extract meaningful features  

\- To train and evaluate machine learning models  

\- To detect anomalies indicating potential electricity theft  



\---



\## 3. Dataset

The dataset used in this project consists of electricity consumption records. Each record represents usage patterns over time.



Data preprocessing steps include:

\- Handling missing values  

\- Normalization and scaling  

\- Feature selection  



(Note: Large datasets are not included due to GitHub file size limitations.)



\---



\## 4. Methodology



\### 4.1 Data Preprocessing

Raw data is cleaned and transformed into a suitable format for model training. This includes normalization and removal of outliers.



\### 4.2 Feature Engineering

Relevant features are extracted to improve model performance. These may include statistical measures such as mean, variance, and temporal consumption trends.



\### 4.3 Model Development

Machine learning algorithms are applied to classify consumption patterns. Models used:

\- Logistic Regression  

\- Random Forest  



\### 4.4 Evaluation Metrics

Model performance is evaluated using:

\- Accuracy  

\- Precision  

\- Recall  

\- F1-score  



\---



\## 5. Experimental Results and Visualization



\### Correlation Matrix

<p align="center">

&#x20; <img src="images/correlation\_matrix.png" width="600"/>

</p>

The correlation matrix illustrates relationships between features and helps identify dependencies in the dataset.



\---



\### Scatter Plot Analysis

<p align="center">

&#x20; <img src="images/scatter\_plot.png" width="600"/>

</p>

Scatter plots provide insight into the distribution of data and potential separability between classes.



\---



\### Log Scale Visualization

<p align="center">

&#x20; <img src="images/log\_scale.png" width="600"/>

</p>

Log scaling improves visualization of skewed data distributions.



\---



\### Confusion Matrix (General Model)

<p align="center">

&#x20; <img src="images/confusion\_matrix.png" width="500"/>

</p>

The confusion matrix shows classification performance across predicted and actual labels.



\---



\### Confusion Matrix (Logistic Regression)

<p align="center">

&#x20; <img src="images/confusion\_matrix\_logistic\_regression.png" width="500"/>

</p>

This matrix highlights the effectiveness of the Logistic Regression model in detecting electricity theft.



\---



\## 6. System Architecture

The system follows a pipeline structure:



1\. Data Collection  

2\. Data Preprocessing  

3\. Feature Extraction  

4\. Model Training  

5\. Prediction and Classification  



\---



\## 7. Conclusion

This project demonstrates that machine learning provides an effective and scalable solution for detecting electricity theft. The models successfully identify abnormal consumption patterns and can be further improved with advanced techniques.



Future work may include:

\- Deep learning models  

\- Real-time monitoring systems  

\- Deployment as a web-based application  



\---



\## 8. Technologies Used

\- Python  

\- Pandas  

\- NumPy  

\- Scikit-learn  

\- Matplotlib  

\- Seaborn  



\---



\## 9. Author

Fatıma Kavraz  

Electrical and Electronics Engineering Student  



\---



\## 10. Notes

This project is developed for educational and research purposes.

