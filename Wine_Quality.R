# Wine Quality 
library(readr)
library(tidyr)
library(ggplot2)
library(factoextra)
## Setting the seed 
set.seed(10) 

#load Red Data
winequality.red <- read.csv("winequality-red.csv",sep=';')
# Quick peak
head(winequality.red)

# making a variable to say type is red
winequality.red <- cbind(grapetype = "red", winequality.red)
winequality.red$grapetype <- as.factor(winequality.red$grapetype)
# Looking at the structure of the dataset
str(winequality.red)

## Load White Data
winequality.white <- read.csv("winequality-white.csv", sep=";")
# qucik look
head(winequality.white)

# making a variable to say type is white
winequality.white <- cbind(grapetype = "white", winequality.white)
winequality.white$grapetype <- as.factor(winequality.white$grapetype)
# Looking at the structure of the dataset
str(winequality.white)

# Combining the data sets into one 
winequality <- rbind(winequality.red, winequality.white)
str(winequality)


# CHecking for NA
any(is.na(winequality))


#Check for incomplete cases (rows)
nrow(winequality[!complete.cases(winequality) , ])

#Summary statistics
summary(winequality) 
######################## EDA #######################
#distribution of the alcohol% attribute
plot(table(winequality$alcohol))
hist(winequality$alcohol)

#distribution of the residual.sugar
plot(table(winequality$residual.sugar))
hist(winequality$residual.sugar)

#distribution of pH
plot(table(winequality$pH))
hist(winequality$pH)


#Correlation
WQ <- winequality[ , -1] #remove the 'grapetype' 
WQcor <- cor(WQ)
WQcor

#Plot correlation
library(corrplot)
corrplot(WQcor)


library(caret)
library(rpart)
library(kernlab)

#Create Training and Validation Datasets
train_index <- createDataPartition(winequality$quality, p = 0.70, list = FALSE) #tell it which column is the classifier

WQ_train <- winequality[train_index, ]
WQ_test <- winequality[-train_index, ]

# looking at the distribution of samples at each quality
table(WQ_train$quality)
table(WQ_test$quality)

#Chaning the quality to a factor
WQ_train$quality <- as.factor(WQ_train$quality)
WQ_test$quality <- as.factor(WQ_test$quality)

#Decision Tree Model training
dt_model_tune <- train(quality~. , data=WQ_train, 
                       metric="Accuracy", 
                       method = "rpart", 
                       tuneLength=5) #default is 3
print(dt_model_tune$finalModel)

#Check decision tree classifiers
print(dt_model_tune) #returns cp, accuracy, kappa (another method for accuracy)

#Decision Tree Model prediction using Test data
dt_predict2 <- predict(dt_model_tune, newdata = WQ_test, type="raw")
head(dt_predict2, 10)

dt_accuracy <- data.frame(dt_predict2, WQ_test[,13])
colnames(dt_accuracy) <- c("prediction","actual")

#Check the Decision Tree prediction accuracy
perc_dt <- length(which(dt_accuracy$actual==dt_accuracy$prediction))/dim(dt_accuracy)[1]
perc_dt

#Decision Tree Confusion matrix
confusionMatrix(dt_predict2, WQ_test$quality)

#Decision Tree Model post-pruning
dt_model_postprune <- prune(dt_model_tune$finalModel, cp=0.2)
print(dt_model_postprune)


#Check decision tree classifers and visualize
plot(dt_model_tune$finalModel)
text(dt_model_tune$finalModel)

library(rattle)
fancyRpartPlot(dt_model_tune$finalModel)

library(rpart.plot)
prp(dt_model_tune$finalModel)


#Support Vector Machine (SVM) model hyperparameter
fitControl <- trainControl(method = "repeatedcv", number = 4, repeats=2)

#SVM Linear Model Training
WQmodel_svm_linear <- train(quality ~ ., data = WQ_train,
                            method = "svmLinear", 
                            trControl = fitControl,
                            tuneGrid = expand.grid(C = 0.4))

WQmodel_svm_linear


#Linear SVM Performance Evaluation: Hold-out Method
predict_svm_linear <- predict(WQmodel_svm_linear, newdata = WQ_test)
#SVM Linear Confusion Matrix
confusionMatrix(predict_svm_linear, WQ_test$quality)

#Check the SVM prediction accuracy
svm_lin_accuracy <- data.frame(predict_svm_linear, WQ_test[,13])
colnames(svm_lin_accuracy) <- c("prediction","actual")

perc_svm1 <- length(which(svm_lin_accuracy$actual==svm_lin_accuracy$prediction))/dim(svm_lin_accuracy)[1]
perc_svm1 #svm linear

#SVM Linear Model Training with scaled data
WQmodel_svm_linear2 <- train(quality ~ ., data = WQ_train,
                             method = "svmLinear", 
                             preProcess = c("center", "scale"),
                             trControl = fitControl,
                             tuneGrid = expand.grid(C = 0.4))

WQmodel_svm_linear2


#Linear SVM Performance Evaluation: Hold-out Method with scaled data
predict_svm_linear2 <- predict(WQmodel_svm_linear2, newdata = WQ_test)
#SVM Linear Confusion Matrix with scaled data
confusionMatrix(predict_svm_linear2, WQ_test$quality)


#Check the SVM prediction accuracy with scaled data
svm_lin_accuracy2 <- data.frame(predict_svm_linear2, WQ_test[,13])
colnames(svm_lin_accuracy2) <- c("prediction","actual")

perc_svm2 <- length(which(svm_lin_accuracy2$actual==svm_lin_accuracy2$prediction))/dim(svm_lin_accuracy2)[1]
perc_svm2 #svm linear with scaled data

#Train Model: SVM with Non-linear Kernel: RBF
WQmodel_svm_rbf <- train(quality ~ ., data = WQ_train, method = "svmRadial", 
                         trControl = fitControl,
                         tuneGrid = expand.grid(sigma = 1, C = 1))

WQmodel_svm_rbf

#Predict SVM with Non-linear Kernel: RBF
predict_svm_rbf <- predict(WQmodel_svm_rbf, newdata = WQ_test)
#Confusion matrix SVM with Non-linear Kernel: RBF
confusionMatrix(predict_svm_rbf, WQ_test$quality)

#Check the SVM with Non-linear Kernel: RBF prediction accuracy
svm_rbf_accuracy <- data.frame(predict_svm_rbf, WQ_test[,13])
colnames(svm_rbf_accuracy) <- c("prediction","actual")

perc_svm3 <- length(which(svm_rbf_accuracy$actual==svm_rbf_accuracy$prediction))/dim(svm_rbf_accuracy)[1]
perc_svm3 #svm RBF

#Train Model: SVM with Non-linear Kernel: RBF with scaled data
WQmodel_svm_rbf2 <- train(quality ~ ., data = WQ_train, method = "svmRadial", 
                          preProcess = c("center", "scale"),
                          trControl = fitControl,
                          tuneGrid = expand.grid(sigma = 1, C = 1))

WQmodel_svm_rbf2

#Predict SVM with Non-linear Kernel: RBF with scaled data
predict_svm_rbf2 <- predict(WQmodel_svm_rbf2, newdata = WQ_test)
#Confusion matrix SVM with Non-linear Kernel: RBF with scaled data
confusionMatrix(predict_svm_rbf2, WQ_test$quality)

#Check the SVM with Non-linear Kernel: RBF prediction accuracy with scaled data
svm_rbf_accuracy2 <- data.frame(predict_svm_rbf2, WQ_test[,13])
colnames(svm_rbf_accuracy2) <- c("prediction","actual")

perc_svm4 <- length(which(svm_rbf_accuracy2$actual==svm_rbf_accuracy2$prediction))/dim(svm_rbf_accuracy2)[1]
perc_svm4

#SVM Model Comparison (1)
model_comparison <- resamples(list(SVMLinear = WQmodel_svm_linear, SVMLinearScale = WQmodel_svm_linear2, SVMRBF = WQmodel_svm_rbf, SVMRBFScale = WQmodel_svm_rbf2))
summary(model_comparison)

#Graphically Compare Performance
scales <- list(x = list(relation = "free"),
               y = list(relation = "free"))

bwplot(model_comparison, scales = scales)

#SVM Model Performance accuracy
SVM_comparison <- cbind(perc_svm1, perc_svm2, perc_svm3, perc_svm4)
colnames(SVM_comparison) <- c("SVM.Linear", "SVM.Linear.Scaled", "SVM.RBF", "SVM.RBF.Scaled")
SVM_comparison

#Discretize the quality prediction variable
WQ_train$quality <- as.numeric(WQ_train$quality)

WQ_train$quality <- cut(WQ_train$quality, breaks=c(0, 3, 6,Inf), labels=c("low","med","high"))

head(WQ_train, 5)

#Discretize the quality prediction variable (testing dataset)
WQ_test$quality <- as.numeric(WQ_test$quality)

WQ_test$quality <- cut(WQ_test$quality, breaks=c(0, 3, 6, Inf), labels=c("low","med","high"))

head(WQ_test, 5)

#Train SVM RBF discretized model
WQmodel_svm_rbf4 <- train(quality ~ ., data = WQ_train, method = "svmRadial", 
                          preProcess = c("center", "scale"),
                          trControl = fitControl,
                          tuneGrid = expand.grid(sigma = 1, C = 1))

WQmodel_svm_rbf4

#Predict quality as “High, Medium, Low” - quality REMOVED
predict_svm_rbf4 <- predict(WQmodel_svm_rbf4, newdata = WQ_test)
#Confusion matrix SVM RBF as “High, Medium, Low” - quality REMOVED
confusionMatrix(predict_svm_rbf4, WQ_test$quality)

#Check the SVM RBF prediction accuracy as “High, Medium, Low” - quality REMOVED
svm_rbf_accuracy4 <- data.frame(predict_svm_rbf4, WQ_test[,13]) 
colnames(svm_rbf_accuracy4) <- c("prediction","actual")

perc_svm6 <- length(which(svm_rbf_accuracy4$actual==svm_rbf_accuracy4$prediction))/dim(svm_rbf_accuracy4)[1]
perc_svm6 #svm rbf discretize

#SVM Model Comparison- with discretized data
model_comparison2 <- resamples(list(SVMLinear = WQmodel_svm_linear, SVMLinearScale = WQmodel_svm_linear2, SVMRBF = WQmodel_svm_rbf, SVMRBFScale = WQmodel_svm_rbf2, SVMDiscretize = WQmodel_svm_rbf4))
summary(model_comparison2)

#Graphically Compare Performance with added discretized prediction models
scales <- list(x = list(relation = "free"),
               y = list(relation = "free"))

bwplot(model_comparison2, scales = scales)

#Model Preformance check
SVM_comparison2 <- cbind(perc_dt, perc_svm1, perc_svm2, perc_svm3, perc_svm4, perc_svm6)
colnames(SVM_comparison2) <- c("DecisionTree", "SVM.Linear", "SVM.Linear.Scaled", "SVM.RBF", "SVM.RBF.Scaled", "SVM.Discretize")
SVM_comparison2

#Removing grape type to run variable importance
WQ_train <- WQ_train[ , -1] #remove the grapetype/character column for variable importance?? 
WQmodel_svm_rbf4 <- train(quality ~ ., data = WQ_train, method = "svmRadial", 
                          preProcess = c("center", "scale"),
                          trControl = fitControl,
                          tuneGrid = expand.grid(sigma = 1, C = 1))


#Variable importance in best performing model
varimp_svmrbf <- varImp(WQmodel_svm_rbf4)
varimp_svmrbf

#Visualize variable importance (1)
plot(varimp_svmrbf, main = "Variable Importance")

############### Clustering #############################
WE_Red_origin <- read.csv("winequality-red.csv",sep=';')
WE_White_origin <- read.csv("winequality-white.csv",sep=';')

## COnverting the columns
WE_Red_origin$quality <- as.factor(WE_Red_origin$quality)
WE_White_origin$quality <- as.factor(WE_White_origin$quality)


WE_Red_Full <- WE_Red_origin
Quality_Red_Full <- WE_Red_Full$quality
WE_Red_Full$quality <- NULL
pre_process_Train <- preProcess(WE_Red_Full, method = c("nzv","scale","center"))
reD_train_Full <- predict(pre_process_Train, newdata = WE_Red_Full)





k <- list()
for(i in 1:10){
  k[[i]] <- kmeans(reD_train_Full,centers = i,nstart=25, iter.max = 100,algorithm = "Hartigan-Wong")
}

betweenss_totss <- list()
for(i in 1:10){
  betweenss_totss[[i]] <- k[[i]]$betweenss/k[[i]]$totss
}

plot(1:10,betweenss_totss,type="b")



wss2 <- function(k){
  return(kmeans(reD_train_Full, k, nstart = 25)$tot.withinss)
}

k_values <- 1:8

wss_values <- purrr::map_dbl(k_values, wss2)

plot(x = k_values, y = wss_values, 
     type = "b", frame = F,
     main = "Red Wine",
     xlab = "Number of clusters K",
     ylab = "Total within-clusters sum of square")

# Training kmeans
RedKMF <- kmeans(reD_train_Full,centers = 5,nstart=25, iter.max = 100,algorithm = "Hartigan-Wong")

# Visualizing clusters
fviz_cluster(RedKMF, data = WE_Red_Full)


#looking at the clusters vs true quality
table(Quality_Red_Full, RedKMF$cluster)


##########################   RED KMEANS Sample ######################################

#Creating an equal number of quality examples
WE_Red_e <- WE_Red_origin
red3 <- which(WE_Red_e$quality==3)
red4 <- which(WE_Red_e$quality==4)
red5 <- which(WE_Red_e$quality==5)
red6 <- which(WE_Red_e$quality==6)
red7 <- which(WE_Red_e$quality==7)
red8 <- which(WE_Red_e$quality==8)

Q3 <- sample(red3,10,replace=FALSE)
Q4 <- sample(red4,10,replace=FALSE)
Q5 <- sample(red5,10,replace=FALSE)
Q6 <- sample(red6,10,replace=FALSE)
Q7 <- sample(red7,10,replace=FALSE)
Q8 <- sample(red8,10,replace=FALSE)

rowsamples <- c(Q3,Q4,Q5,Q6,Q7,Q8)

WE_Red_Equal <- WE_Red_e[rowsamples,]



#Descitizing quality into low and high
HQe6 <- which(WE_Red_Equal$quality==6)
HQe7 <- which(WE_Red_Equal$quality==7)
HQe8 <- which(WE_Red_Equal$quality==8)
HQe <- c(HQe6,HQe7, HQe8)
WE_Red_Equal$HLQ <- "Low_Quality"
WE_Red_Equal$HLQ[HQe] <- "High_Quality"
Quality_Red_Equal <- WE_Red_Equal$HLQ


#Removing HLQ and quality for our analysis and scaling our data

WE_Red_Equal$HLQ <- NULL
WE_Red_Equal$quality <- NULL
pre_process_Train <- preProcess(WE_Red_Equal, method = c("nzv","scale","center"))
reD_train <- predict(pre_process_Train, newdata = WE_Red_Equal)


#Selecting attributes and training Kmeans


UC <- c(
  2
  ,3
  ##,4
  ##,5
  ,6
  ##     ,7
  ##,8
  ##      ,9
  ,10
  ,11
)
WE_Red_UC <- reD_train[,UC]
RedKMUC <- kmeans(WE_Red_UC,centers = 2,nstart=25, iter.max = 100,algorithm = "Hartigan-Wong")


#Confusion Matrix

table(Quality_Red_Equal, RedKMUC$cluster)
#Visualizing our clusters

fviz_cluster(RedKMUC, data = WE_Red_UC)

#Most important Attributes for clustering

## Making our centers into a data frame.
RKMe <- data.frame(RedKMUC$centers)
## Cluster 1
which.max(RKMe[1,])
## Cluster 2
which.max(RKMe[2,])

#Accuracy, precision, recall, and Fmeasure. 

accuracy <- function(TP,TN,FP,FN)
{A <-(TP+TN)/(TP+FP+FN+TN)
return(A)
}

precision <- function(TP,TN,FP,FN)
{P <-(TP)/(TP+FP)
return(P)
}

recall <- function(TP,TN,FP,FN)
{R <-(TP)/(TP+FN)
return(R)
} 

accuracy(24,24,6,6)

precision(24,24,6,6)

recall(24,24,6,6)

Fmeasure <- function(p,r)
{(2*p*r)/(p+r)
}

Fmeasure(precision(24,24,6,6),recall(24,24,6,6))

##########################   RED KMEANS FULL ######################################

#Creating an equal number of quality examples
WE_Red_f <- WE_Red_origin

#Descitizing quality into low and high
HQf6 <- which(WE_Red_f$quality==6)
HQf7 <- which(WE_Red_f$quality==7)
HQf8 <- which(WE_Red_f$quality==8)
HQf <- c(HQf6,HQf7, HQf8)
WE_Red_f$HLQ <- "Low_Quality"
WE_Red_f$HLQ[HQf] <- "High_Quality"
Quality_Red_F <- WE_Red_f$HLQ
table(Quality_Red_F)


#Removing HLQ and quality for our analysis and scaling our data

WE_Red_f$HLQ <- NULL
WE_Red_f$quality <- NULL
pre_process_Train_2 <- preProcess(WE_Red_f, method = c("nzv","scale","center"))
reD_train_f <- predict(pre_process_Train_2, newdata = WE_Red_f)


#Selecting attributes and training Kmeans
UC2 <- c(
  2
  ,3
  ##,4
  ##,5
  ,6
  ##     ,7
  ##,8
  ##      ,9
  ,10
  ,11
)
WE_Red_UC2 <- reD_train_f[,UC2]
RedKMUC2 <- kmeans(WE_Red_UC2,centers = 2,nstart=25, iter.max = 100,algorithm = "Hartigan-Wong")

#Confusion Matrix
table(Quality_Red_F, RedKMUC2$cluster)

#Visualizing our clusters

fviz_cluster(RedKMUC2, data = WE_Red_UC2)


#Most important Attributes for clustering
## Making our centers into a data frame.
RKMe2 <- data.frame(RedKMUC2$centers)
## Cluster 1
which.max(RKMe2[1,])
## Cluster 2
which.max(RKMe2[2,])


#Accuracy, precision, recall, and Fmeasure. 
accuracy <- function(TP,TN,FP,FN)
{A <-(TP+TN)/(TP+FP+FN+TN)
return(A)
}

precision <- function(TP,TN,FP,FN)
{P <-(TP)/(TP+FP)
return(P)
}

recall <- function(TP,TN,FP,FN)
{R <-(TP)/(TP+FN)
return(R)
} 

accuracy(478,562,377,182)

precision(478,562,377,182)

recall(478,562,377,182)

Fmeasure <- function(p,r)
{(2*p*r)/(p+r)
}

Fmeasure(precision(478,562,377,182),recall(478,562,377,182))

##########################   WHITE KMEANS Sample ######################################
#Creating an equal number of quality examples
WE_White_e <- WE_White_origin
White3 <- which(WE_White_e$quality==3)
White4 <- which(WE_White_e$quality==4)
White5 <- which(WE_White_e$quality==5)
White6 <- which(WE_White_e$quality==6)
White7 <- which(WE_White_e$quality==7)
White8 <- which(WE_White_e$quality==8)
White9 <- which(WE_White_e$quality==9)

Qw3 <- sample(White3,5,replace=FALSE)
Qw4 <- sample(White4,5,replace=FALSE)
Qw5 <- sample(White5,5,replace=FALSE)
Qw6 <- sample(White6,5,replace=FALSE)
Qw7 <- sample(White7,5,replace=FALSE)
Qw8 <- sample(White8,5,replace=FALSE)
Qw9 <- sample(White8,5,replace=FALSE)
rowsamples2 <- c(Qw3,Qw4,Qw5,Qw6,Qw7,Qw8)

WE_White_Equal <- WE_White_e[rowsamples2,]

#Descitizing quality into low and high
HQew6 <- which(WE_White_Equal$quality==6)
HQew7 <- which(WE_White_Equal$quality==7)
HQew8 <- which(WE_White_Equal$quality==8)
HQew9 <- which(WE_White_Equal$quality==9)
HQew <- c(HQew6,HQew7, HQew8,HQew9)
WE_White_Equal$HLQ <- "Low_Quality"
WE_White_Equal$HLQ[HQew] <- "High_Quality"
Quality_White_Equal <- WE_White_Equal$HLQ

#Removing HLQ and quality for our analysis and scaling our data
WE_White_Equal$HLQ <- NULL
WE_White_Equal$quality <- NULL
pre_process_Train_3 <- preProcess(WE_White_Equal, method = c("nzv","scale","center"))
White_train <- predict(pre_process_Train_3, newdata = WE_White_Equal)

#Selecting attributes and training Kmeans
UC <- c(
  2
  ,3
  ##,4
  ##,5
  ,6
  ##     ,7
  ##,8
  ##      ,9
  ,10
  ,11
)
WE_White_UC <- White_train[,UC]
WhiteKMUC <- kmeans(WE_White_UC,centers = 2,nstart=25, iter.max = 100,algorithm = "Hartigan-Wong")

#Confusion Matrix
table(Quality_White_Equal, WhiteKMUC$cluster)

#Visualizing our clusters
fviz_cluster(WhiteKMUC, data = WE_White_UC)


#Most important Attributes for clustering

## Making our centers into a data frame.
WKMe <- data.frame(WhiteKMUC$centers)
## Cluster 1
which.max(WKMe[1,])
## Cluster 2
which.max(WKMe[2,])

#Accuracy, precision, recall, and Fmeasure. 
accuracy <- function(TP,TN,FP,FN)
{A <-(TP+TN)/(TP+FP+FN+TN)
return(A)
}

precision <- function(TP,TN,FP,FN)
{P <-(TP)/(TP+FP)
return(P)
}

recall <- function(TP,TN,FP,FN)
{R <-(TP)/(TP+FN)
return(R)
} 

accuracy(10,9,5,6)

precision(10,9,5,6)

recall(10,9,5,6)

Fmeasure <- function(p,r)
{(2*p*r)/(p+r)
}

Fmeasure(precision(10,9,5,6),recall(10,9,5,6))
##########################   WHITE KMEANS FULL ######################################

#Creating an equal number of quality examples
WE_White_f <- WE_White_origin

#Descitizing quality into low and high
HQfw6 <- which(WE_White_f$quality==6)
HQfw7 <- which(WE_White_f$quality==7)
HQfw8 <- which(WE_White_f$quality==8)
HQfw9 <- which(WE_White_f$quality==9)
HQfwH <- c(HQfw7, HQfw8,HQfw9)
WE_White_f$HLQ <- "Low_Quality"
WE_White_f$HLQ[HQfw6] <- "Mid_Quality"
WE_White_f$HLQ[HQfwH] <- "High_Quality"
Quality_White_Full <- WE_White_f$HLQ
table(Quality_White_Full)

#Removing HLQ and quality for our analysis and scaling our data
WE_White_f$HLQ <- NULL
WE_White_f$quality <- NULL
pre_process_Train_4 <- preProcess(WE_White_f, method = c("nzv","scale","center"))
White_train_f <- predict(pre_process_Train_4, newdata = WE_White_f)


#Selecting attributes and training Kmeans
UC2 <- c(
  2
  ,3
  ##,4
  ##,5
  ,6
  ##     ,7
  ##,8
  ##      ,9
  ,10
  ,11
)
WE_White_UC2 <- White_train_f[,UC2]
WhiteKMUC2 <- kmeans(WE_White_UC2,centers = 2,nstart=25, iter.max = 100,algorithm = "Hartigan-Wong")

#Confusion Matrix
table(Quality_White_Full, WhiteKMUC2$cluster)

#Visualizing our clusters
fviz_cluster(WhiteKMUC2, data = WE_White_UC2)


#Most important Attributes for clustering
## Making our centers into a data frame.
WKMe2 <- data.frame(WhiteKMUC2$centers)
## Cluster 1
which.max(WKMe2[1,])
## Cluster 2
which.max(WKMe2[2,])

#Accuracy, precision, recall, and Fmeasure. 
accuracy <- function(TP,TN,FP,FN)
{A <-(TP+TN)/(TP+FP+FN+TN)
return(A)
}

precision <- function(TP,TN,FP,FN)
{P <-(TP)/(TP+FP)
return(P)
}

recall <- function(TP,TN,FP,FN)
{R <-(TP)/(TP+FN)
return(R)
} 




accuracy(1772,2120,337,669)

precision(1772,2120,337,669)

recall(1772,2120,337,669)

Fmeasure <- function(p,r)
{(2*p*r)/(p+r)
}

Fmeasure(precision(1772,2120,337,669),recall(1772,2120,337,669))

######################   RED KNN  ###########################
WE_Red <- WE_Red_origin

pre_process_Train <- preProcess(WE_Red, method = c("nzv","scale","center"))
reD_train_KNN <- predict(pre_process_Train, newdata = WE_Red)

reDSplit <- createDataPartition(reD_train_KNN$quality, p = .8, list = FALSE)
reDTrainUC <- reD_train_KNN[reDSplit,]
redDValidateUC  <- reD_train_KNN[-reDSplit,]

Red_model_KNN <- train(quality ~ ., data = reDTrainUC, method = "knn",
                       tuneGrid = data.frame(k = seq(1, 25)),
                       trControl = trainControl(method = "repeatedcv", 
                                                number = 10, repeats = 3))
print(Red_model_KNN)

reD_predict_KNN <- predict(Red_model_KNN, newdata = redDValidateUC)
confusionMatrix(reD_predict_KNN, redDValidateUC$quality, positive = "pos")
######################   RED KNN High Low  ###########################
WE_Red_HL <- WE_Red_origin

HQknn6 <- which(WE_Red_HL$quality==6)
HQknn7 <- which(WE_Red_HL$quality==7)
HQknn8 <- which(WE_Red_HL$quality==8)

HQknn <- c(HQknn6,HQknn7, HQknn8)

WE_Red_HL$HLQ <- "Low_Quality"
WE_Red_HL$HLQ[HQknn] <- "High_Quality"
WE_Red_HL$HLQ <- as.factor(WE_Red_HL$HLQ)

WE_Red_HL$quality<- NULL

pre_process_Train_KNN <- preProcess(WE_Red_HL, method = c("nzv","scale","center"))
reD_train_KNN2 <- predict(pre_process_Train_KNN, newdata = WE_Red_HL)

reDSplit <- createDataPartition(reD_train_KNN2$HLQ, p = .8, list = FALSE)
reDTrainUC2 <- reD_train_KNN2[reDSplit,]
redDValidateUC2  <- reD_train_KNN2[-reDSplit,]


Red_model_KNN2 <- train(HLQ ~ ., data = reDTrainUC2, method = "knn",
                        tuneGrid = data.frame(k = seq(1, 25)),
                        trControl = trainControl(method = "repeatedcv", 
                                                 number = 10, repeats = 3))
print(Red_model_KNN2)


reD_predict_KNN2 <- predict(Red_model_KNN2, newdata = redDValidateUC2)
confusionMatrix(reD_predict_KNN2, redDValidateUC2$HLQ)

######################   White KNN High Low  ###########################
WE_White_HL <- WE_White_origin

HQknnw6 <- which(WE_White_HL$quality==6)
HQknnw7 <- which(WE_White_HL$quality==7)
HQknnw8 <- which(WE_White_HL$quality==8)
HQknnw9 <- which(WE_White_HL$quality==9)

HQknnw <- c(HQknnw6,HQknnw7, HQknnw8,HQknnw9)

WE_White_HL$HLQ <- "Low_Quality"
WE_White_HL$HLQ[HQknnw] <- "High_Quality"
WE_White_HL$HLQ <- as.factor(WE_White_HL$HLQ)

WE_White_HL$quality<- NULL

pre_process_Train_KNN2 <- preProcess(WE_White_HL, method = c("nzv","scale","center"))
White_train_KNN2 <- predict(pre_process_Train_KNN2, newdata = WE_White_HL)

WhiteSplit <- createDataPartition(White_train_KNN2$HLQ, p = .8, list = FALSE)
WhiteTrainUC2 <- White_train_KNN2[WhiteSplit,]
WhiteValidateUC2  <- White_train_KNN2[-WhiteSplit,]

White_model_KNN2 <- train(HLQ ~ ., data = WhiteTrainUC2, method = "knn",
                          tuneGrid = data.frame(k = seq(1, 25)),
                          trControl = trainControl(method = "repeatedcv", 
                                                   number = 10, repeats = 3))
print(White_model_KNN2)

White_predict_KNN2 <- predict(White_model_KNN2, newdata = WhiteValidateUC2)
confusionMatrix(White_predict_KNN2, WhiteValidateUC2$HLQ)

######################   Both KNN High Low  ###########################
WE_Red_origin2 <- WE_Red_origin
WE_White_origin2 <- WE_White_origin
WE_Red_origin2$type <- "red"
WE_White_origin2$type <- "white"
Both_HL <- rbind(WE_White_origin2,WE_Red_origin2)


HQknnb6 <- which(Both_HL$quality==6)
HQknnb7 <- which(Both_HL$quality==7)
HQknnb8 <- which(Both_HL$quality==8)
HQknnb9 <- which(Both_HL$quality==9)

HQknnb <- c(HQknnb6,HQknnb7, HQknnb8,HQknnb9)

Both_HL$HLQ <- "Low_Quality"
Both_HL$HLQ[HQknnb] <- "High_Quality"
Both_HL$HLQ <- as.factor(Both_HL$HLQ)

Both_HL$quality<- NULL

pre_process_Train_KNN3 <- preProcess(Both_HL, method = c("nzv","scale","center"))
Both_train_KNN3 <- predict(pre_process_Train_KNN3, newdata = Both_HL)

bothSplit <- createDataPartition(Both_train_KNN3$HLQ, p = .8, list = FALSE)
BothTrain <- Both_train_KNN3[bothSplit,]
BothValidate  <- Both_train_KNN3[-bothSplit,]


Both_model_KNN <- train(HLQ ~ ., data = BothTrain, method = "knn",
                        tuneGrid = data.frame(k = seq(1, 25)),
                        trControl = trainControl(method = "repeatedcv", 
                                                 number = 10, repeats = 3))
print(Both_model_KNN)

Both_predict_KNN <- predict(Both_model_KNN, newdata = BothValidate)
confusionMatrix(Both_predict_KNN, BothValidate$HLQ)

#Loading Libraries for our discovery
library(arules)
library(arulesViz)
library(plyr)
library(dplyr)

#Load data sets
red <- read.csv("winequality-red.csv",sep=';')
white <- read.csv("winequality-white.csv",sep=';')

#Viewing Data
hist(red$quality)
hist(white$quality)


#Discretize Categories
##quality 5 - 7 mid ##quality < 5 low ##quality > 7 high

# Convert to numeric
red$quality <- as.numeric(red$quality)
white$quality <- as.numeric(white$quality)

#Discretize the quality attribute
red$quality_disc <- cut(red$quality, breaks = c(0,4,6,Inf),
                        labels=c("low","mid","high"))


table(red$quality, red$quality_disc)


white$quality_disc <- cut(white$quality, breaks = c(0,4,6,Inf),
                          labels=c("low","mid","high"))

table(white$quality, white$quality_disc)


str(red)
#Remove quality attribute so it does not create unecesarry rules
red <- red %>% select(-quality)
white <- white %>% select(-quality)
#Create one data set for all wines to see if rules for each compare with combined DS
red_white <- rbind(red,white)

#Create some rules for discovery
#Red Wine Rules
rules_highQ <- apriori(data=red, parameter = list(supp=0.001, conf=0.08,minlen=2),
                       appearance = list(default="rhs",lhs="quality_disc=high"),
                       control=list(verbose=T))

rules_highQ <- sort(rules_highQ, decreasing = T, by = "lift")
rules_highQ

#mid 
rules_midQ <- apriori(data=red, parameter = list(supp=0.001, conf=0.08,minlen=2),
                      appearance = list(default="rhs",lhs="quality_disc=mid"),
                      control=list(verbose=T))
rules_midQ <- sort(rules_midQ, decreasing = T, by = "lift")
inspect(rules_midQ)

#low
rules_lowQ <- apriori(data=red, parameter = list(supp=0.001, conf=0.08,minlen=2),
                      appearance = list(default="rhs",lhs="quality_disc=low"),
                      control=list(verbose=T))

rules_lowQ <- sort(rules_lowQ, decreasing = T, by = "lift")
inspect(rules_lowQ)

#White Wine Rules
rules_wht <- apriori(white, parameter=list(supp=0.001, conf=0.9, maxlen=4))

rules_wht <- sort(rules_wht, decreasing = TRUE, by ="confidence")
inspect(rules_wht[1:50])

#Combined Red White Rules
rules_redwht <- apriori(red_white, parameter=list(supp=0.001, conf=0.9, maxlen=4))

rules_redwht <- sort(rules_redwht, decreasing = T, by="confidence")
inspect(rules_redwht[1:50])


rules_rdwht_mid <- apriori(data=red_white, parameter = list(supp=0.001, conf=0.9,maxlen=4),
                           appearance = list(default="lhs",rhs="quality_disc=mid"),
                           control=list(verbose=T))
rules_rdwht_mid <- sort(rules_rdwht_mid, decreasing = T, by="lift")
inspect(rules_rdwht_mid[1:10])



rules_rdwht_high <- apriori(data=red_white, parameter = list(supp=0.001, conf=0.9,maxlen=4),
                            appearance = list(default="lhs",rhs="quality_disc=high"),
                            control=list(verbose=T))


rules_rdwht_high <- sort(rules_rdwht_high, decreasing = T, by="lift")
inspect(rules_rdwht_high[1:10])

