# Wine_Quality
Group project by: Morgan Gere, Dan Burke, Natali Newman, and Nick Tinsley
### Overview
Exploration of the wine quality data set found on the University of Irvine's Machine Learning Repository.  THe goal is to predict the quality of wine and identify best varaiables for high quality wine.  This project was mostly done in R with two models being done in python.  The data itself contains multiple columns off empirical data such as pH, acidity, residual sugar levels, alcohol % and more along with the rating giving at this winery.
### EDA
Some Distributions to have a better understanding of the variables.

![image](https://user-images.githubusercontent.com/118774600/213280735-e5b93111-bf04-4eff-b9a2-88a9847002ac.png)

![image](https://user-images.githubusercontent.com/118774600/213280845-ad62e7a2-ddb7-4bdb-b1d1-4ec635e5cc92.png)

![image](https://user-images.githubusercontent.com/118774600/213280871-bc5828f3-d934-4a4d-8021-3485e0fce011.png)

Correlation plot to understand how the variables posibly relate to one another.
R

![image](https://user-images.githubusercontent.com/118774600/213281054-35324192-f8fa-4e1d-8c21-8454717fe498.png)

Python

![image](https://user-images.githubusercontent.com/118774600/213281815-685a0c91-23b8-4798-82ab-f1ff1fd4e69a.png)

### Decision Tree
The data was split on a 70/30 train/test using quality as a factor for the predictor. The accuracy was the lowest as to be expected, 51.64% the tree can be seen below indicating the most important variables to be alcohol amount, volatility, residual sugars and free sulur dioxide.

![image](https://user-images.githubusercontent.com/118774600/213282894-0b25afad-9b49-4e38-8b64-cca3af243f4a.png)

Confusion Matrix and Statistics

          Reference
Prediction   3   4   5   6   7   8   9
         3   0   0   0   0   0   0   0
         4   0   0   0   0   0   0   0
         5   4  49 403 254  13   3   0
         6   4  29 219 571 264  53   1
         7   0   1   6  25  32  17   0
         8   0   0   0   0   0   0   0
         9   0   0   0   0   0   0   0

Overall Statistics
                                         
               Accuracy : 0.5164         
                 95% CI : (0.494, 0.5388)
    No Information Rate : 0.4363         
    P-Value [Acc > NIR] : 7.659e-13      
                                         
                  Kappa : 0.2171         
                                         
 Mcnemar's Test P-Value : NA             

Statistics by Class:

                     Class: 3 Class: 4 Class: 5 Class: 6 Class: 7
Sensitivity          0.000000  0.00000   0.6417   0.6718  0.10356
Specificity          1.000000  1.00000   0.7553   0.4809  0.97010
Pos Pred Value            NaN      NaN   0.5551   0.5004  0.39506
Neg Pred Value       0.995893  0.95945   0.8159   0.6543  0.85163
Prevalence           0.004107  0.04055   0.3224   0.4363  0.15862
Detection Rate       0.000000  0.00000   0.2069   0.2931  0.01643
Detection Prevalence 0.000000  0.00000   0.3727   0.5857  0.04158
Balanced Accuracy    0.500000  0.50000   0.6985   0.5763  0.53683
                     Class: 8  Class: 9
Sensitivity           0.00000 0.0000000
Specificity           1.00000 1.0000000
Pos Pred Value            NaN       NaN
Neg Pred Value        0.96253 0.9994867
Prevalence            0.03747 0.0005133
Detection Rate        0.00000 0.0000000
Detection Prevalence  0.00000 0.0000000
Balanced Accuracy     0.50000 0.5000000

### Linear SVM
The linear SVM was run in the same test split with manual tuning.  The accuracy of prediction was found to be 53.59%

Confusion Matrix and Statistics

          Reference
Prediction   3   4   5   6   7   8   9
         3   0   0   0   0   0   0   0
         4   0   0   0   0   0   0   0
         5   5  50 388 194  21   6   0
         6   3  29 240 656 288  67   1
         7   0   0   0   0   0   0   0
         8   0   0   0   0   0   0   0
         9   0   0   0   0   0   0   0

Overall Statistics
                                          
               Accuracy : 0.5359          
                 95% CI : (0.5135, 0.5583)
    No Information Rate : 0.4363          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.2298          
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: 3 Class: 4 Class: 5 Class: 6 Class: 7
Sensitivity          0.000000  0.00000   0.6178   0.7718   0.0000
Specificity          1.000000  1.00000   0.7909   0.4281   1.0000
Pos Pred Value            NaN      NaN   0.5843   0.5109      NaN
Neg Pred Value       0.995893  0.95945   0.8131   0.7078   0.8414
Prevalence           0.004107  0.04055   0.3224   0.4363   0.1586
Detection Rate       0.000000  0.00000   0.1992   0.3368   0.0000
Detection Prevalence 0.000000  0.00000   0.3409   0.6591   0.0000
Balanced Accuracy    0.500000  0.50000   0.7044   0.5999   0.5000
                     Class: 8  Class: 9
Sensitivity           0.00000 0.0000000
Specificity           1.00000 1.0000000
Pos Pred Value            NaN       NaN
Neg Pred Value        0.96253 0.9994867
Prevalence            0.03747 0.0005133
Detection Rate        0.00000 0.0000000
Detection Prevalence  0.00000 0.0000000
Balanced Accuracy     0.50000 0.5000000

### SVM RBF Kernel
The SVM with RBF Kernel was run in the same test split with manual tuning. The accuracy of prediction was found to be 62.42%

Confusion Matrix and Statistics

          Reference
Prediction   3   4   5   6   7   8   9
         3   0   0   0   0   0   0   0
         4   0   2   0   0   0   0   0
         5   2  25 392 131  13   1   0
         6   4  37 246 679 162  40   3
         7   0   0   7  39 132  19   1
         8   0   0   0   1   1  11   0
         9   0   0   0   0   0   0   0

Overall Statistics
                                          
               Accuracy : 0.6242          
                 95% CI : (0.6023, 0.6458)
    No Information Rate : 0.4363          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.3992          
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: 3 Class: 4 Class: 5 Class: 6 Class: 7
Sensitivity           0.00000 0.031250   0.6078   0.7988  0.42857
Specificity           1.00000 1.000000   0.8680   0.5519  0.95976
Pos Pred Value            NaN 1.000000   0.6950   0.5798  0.66667
Neg Pred Value        0.99692 0.968140   0.8172   0.7799  0.89943
Prevalence            0.00308 0.032854   0.3311   0.4363  0.15811
Detection Rate        0.00000 0.001027   0.2012   0.3486  0.06776
Detection Prevalence  0.00000 0.001027   0.2895   0.6011  0.10164
Balanced Accuracy     0.50000 0.515625   0.7379   0.6754  0.69416
                     Class: 8 Class: 9
Sensitivity          0.154930 0.000000
Specificity          0.998934 1.000000
Pos Pred Value       0.846154      NaN
Neg Pred Value       0.968992 0.997947
Prevalence           0.036448 0.002053
Detection Rate       0.005647 0.000000
Detection Prevalence 0.006674 0.000000
Balanced Accuracy    0.576932 0.500000

### Discretized  Models
The Wine qualities were discretized  into low, medium, and high quality wines.  This would increase greatly the accuracy and allow focus on making sure the prediction would give high results for the wine quality.  The SVM Model using RBF kernal was run on the discritized data giving an accuracy of 77.82%

Confusion Matrix and Statistics

          Reference
Prediction  low  med high
      low   418  131    0
      med   297 1098    4
      high    0    0    0

Overall Statistics
                                          
               Accuracy : 0.7782          
                 95% CI : (0.7591, 0.7965)
    No Information Rate : 0.6309          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.4999          
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: low Class: med Class: high
Sensitivity              0.5846     0.8934    0.000000
Specificity              0.8938     0.5814    1.000000
Pos Pred Value           0.7614     0.7848         NaN
Neg Pred Value           0.7877     0.7614    0.997947
Prevalence               0.3670     0.6309    0.002053
Detection Rate           0.2146     0.5637    0.000000
Detection Prevalence     0.2818     0.7182    0.000000
Balanced Accuracy        0.7392     0.7374    0.500000

A comparison plot of the different SVM models is below.

![image](https://user-images.githubusercontent.com/118774600/213288829-2884b4a0-cb4d-4650-b1fc-ea44b4f5fe0c.png)

The Variable Importance's when the quality was discretized is shown below indicating that chlorides and alcohol have the most significance.

![image](https://user-images.githubusercontent.com/118774600/213291695-deb211d8-4a2e-4e29-a284-2ad0dc7a11d8.png)

### K-means Clustering

At first in an attempt to gain most information possible both elbow and shoulder methods were implimented to find the optimum number of clusters.
5 clusters was selected as a starting point.

![image](https://user-images.githubusercontent.com/118774600/213292298-2344a0d6-9928-45a7-8f40-c16f9f369b0d.png)

This was found to be problematic as the clusters were very random with what quality of wines were in each one.  After attempting to use specific attributes, while eliminating others, it was not yielding good results.  

![image](https://user-images.githubusercontent.com/118774600/213292421-c1250d4c-333a-4d81-a487-d855b3018548.png)

The quality numbers were very skewed towards ratings 5 and 6.  Outside those ranges the data was quite limited only having 10 examples of wine with a quality of 3.  To observe high quality versus low quality the quality data was discretized so any wine with quality rating of 5 or lower was marked as “low quality” and any above was marked as “high quality”.  The data set was broken apart into the different levels of quality and 10 random samples were taken from each of the levels.  This new dataset was clustered into two clusters attributes that were highly correlated to others as well as attributes that were proving to have no change on the clustering were removed.  This led to having two clusters one comprised of red wines with high quality and one with red wines that contained low quality.  Accuracy, precision, recall, and F measure were all found to be 80%.

With the same attribute columns and the discretized quality of low and high the entire data set was then used with the k-means algorithm using 2 clusters.  This yielded some much better results than previously.  The clusters were close to one another but no overlapping and when checking one appeared to high quality wine and one low.  Checking the Accuracy was 65%, the precision was 55.9% the recall was 72.4% and the F measure was 63.1%.  This tells us that some red wines are being placed into high quality that are not at a higher rate, but it is less likely for the clusters to place a high-quality wine into the low quality.    

![image](https://user-images.githubusercontent.com/118774600/213292564-cfb44214-b6b3-422d-8bb9-d8848cf6ef3f.png)

![image](https://user-images.githubusercontent.com/118774600/213292574-8f3807c0-8729-44cf-acd8-27b5b7b7880f.png)

The same procedure was done with the white wine dataset.  The white wine dataset has larger range of quality ratings but has less of quantity in those ranges.  When the full dataset was clustered using the high and low discretized data it was found to have many high or low errors.  To investigate this the quality was discretized into low being a rating of 5 or less mid being a rating of 6 and high being 7 or greater.  The K-means was run but still set to 2 clusters attempting to predict high and low quality and showing how the “mid” quality rating wines were being handled by the algorithm.  As was suspected the mid quality was being divided into the two clusters.  1049 in high quality and 1149 in low quality.  This places our accuracy very high assuming the white wine with a rating of 6 is being divided correctly.  This leads to some follow up questions about how the quality ratings were obtained.  

![image](https://user-images.githubusercontent.com/118774600/213292611-f0816b5f-2013-4c32-82e9-ffd389df131e.png)

### KNN

KNN was used to predict the quality of the wine. The wine datasets were split into Training and validation sets.   The accuracy was found to be low only ~61.2% the quality data was discretized into high and low using the same parameters as before (low <=5 and high >5) The KNN was rerun finding ~75.6% accuracy and a kappa of 50.8%.  This was followed up with predicting the validation set and a confusion matrix was used to find the accuracy of 76.8% and a kappa of 53.11%.  This is a very fast method to achieve good results.  
The KNN was run on the White data set and found similar results from the model.  Accuracy of ~79.9% and a Kappa of 54.5%.  The validation set prediction confusion matrix found ~80% accuracy and a kappa of ~54.9.

Both datasets were combined adding a column named type.  This identified the wine as either red or white.  A KNN was run on this in the same way and found to be ~78.3% accuracy on the model with a kappa of 52.9%.  While the confusion matrix run on the predictions of the validation set were found to be ~79% with a kappa of 54.58%.  This tells that combining all the data gives us a fast way of identifying new batches created.

### Random Forest + Naive Bayes

Initially, a Random Forest and Naïve Bayes models were generated with R Studio, however it became apparent that memory management issues were avoided with the python scikit-learn implementation. In our first R implementation we encountered memory utilization issues which may interactive development both time consuming and impractical due to RStudio crashing. After this development, it became clear that the python implementation is better suited for our analysis. 

Each of these models were generated utilizing a test and train split. Various splits were tested through iteration, however we limited this to not exceed an 80/20 (train/test) split. Exceeding this 80/20 split was tested, however it produced unrealistic accuracy when predicting the test dataset. Though we do not have additional data to test these models, we concluded that we had encountered a situation where overfitting should not only be suspected but expected. 

Like our Support Vector implementation within this analysis, we decided to segment the wine quality into three (3) groups, low, medium and high, with their respective ranges of 3-5,6-7 and 8-9 

Random Forest Results
    precision    recall  f1-score   support

           0       0.76      0.74      0.75       830
           1       0.82      0.86      0.84      1383
           2       0.95      0.31      0.47        61

    accuracy                           0.80      2274
   macro avg       0.85      0.64      0.69      2274
weighted avg       0.81      0.80      0.80      2274

Naive Bayes Results
              precision    recall  f1-score   support

           0       0.59      0.58      0.59       718
           1       0.72      0.72      0.72      1183
           2       0.13      0.12      0.12        49

    accuracy                           0.65      1950
   macro avg       0.48      0.48      0.48      1950
weighted avg       0.65      0.65      0.65      1950

### Association Rule MIning

To see if we could decide a mid to high quality wine based on associated attributes, we applied Association Rule Mining Apriori algorithm to the White and Red wine data sets.  First attempt at getting a good set of rules the ARM code was applied to each data sets individually. It was discovered that the first set of rules were not getting a strong enough confidence due to the lack of records in each set.

The Red and White data were combined into a single data set and the Apriori algorithm was applied to the set as a whole.  This time the rules had high confidence, as well as a high lift indicator.  This gave a indication that we had strong rules that we could use to see what attributes could be recommended for a Mid to High quality of wine.

Next step was to Discretize the quality attribute from numbers to quality indicators.  A quality rating of 0-4 was considered low quality, 5-7 was mid and anything above 8 was considered a high-quality wine.

When Apriori is applied to the discretized data we find that a Mid qualilty wine can be assotiated with a Sulfer Dioxide count greater than 220 and Density measured obobe .995.  Where as a High quality wine is assotiated with a Sulfer Dioxide range around 48.5, and a Alcholol level around 13.7%

 Mid Quality

![image](https://user-images.githubusercontent.com/118774600/213297012-6888e573-e79f-4b04-a0d1-3bda0720dd78.png)

High Quality

![image](https://user-images.githubusercontent.com/118774600/213297063-ca4c67e8-50c8-4586-9942-9cd6942f228e.png)
