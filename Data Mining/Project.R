########################################################################################################################################
###############################       DATA MINING PROJECT       ########################################################################
########################################################################################################################################

getwd()                       #Get the current directory

#Set the working directory
setwd("/Users/joseguedes/Documents/Mestrado em gestão/2ºsemestre/Data mining/trabalho/data")   

df <- read.csv("data.csv")    #Reading data in csv file

###############################       LIBRARY       ####################################################################################

#Loading necessary libraries for data processing and modeling 
library(fastDummies)          #To convert categorical variables into binary (dummy)
library(ggplot2)              #For data visualization 
library(splitTools)           #To create training and test sets
library(FNN)                  #K-NN Model
library(e1071)                #Naïve Bayes
library(rpart)                #Decision tree    
library(rpart.plot)           #Plotting Decision tree
library(randomForest)         #Random Forest
library(caret)                #Cross Validation
library(nnet)                 #Logistic regression
library(glmnet)               #Logistic regression

#####################     DATA CHARACTERISTICS & DATA PREPARATION    ########################################################################

#Data Preprocessing and understanding the nature of the dataset
df <- df[,-1]                 #Remove 1st col with ID because it's not necessary 
str(df)                       #To understand the type of each variable 

#Handling missing values in categorical variables
df[df == ""] <- NA            #Replace empty info with NA

#Amending the data type of specific columns
df[,c(1,2,4,5,7,9,10)] <- lapply(df[,c(1,2,4,5,7,9,10)],as.factor)        #Type factor
df[,c(3,6,8)] <- lapply(df[,c(3,6,8)],as.numeric)                         #Type numeric
str(df)                                                                   #Final type check

apply(df,2,function(x) sum(is.na(x)))                                     #Check NAs in each col

aux <- apply(df,1,function(x)sum(is.na(x)))                             
length(aux[aux<2])                                                        #Num of rows with <2 NA's

df_1 <- df[which(aux<2),]                                                 #Remove rows with >2 NA's
summary(df_1)

#Replace NA's in categorical variables with the mode
for (i in c('Ever_Married', 'Graduated', 'Profession', "Var_1")){
  df_1[is.na(df_1[,i]),i] <- names(which.max(table(df_1[,i])))
}

apply(df_1,2,function(x) sum(is.na(x)))                                   #Check remaining NA's

#We analyse the distribution of the data regarding the work experience and the family size
#to replace NA's for variables "Work_Experience" and "Family_Size" by the mean or the median accordingly

#Visualize Distributions
hist(df_1$Work_Experience, main = "Histogram of Work Experience", xlab = "Variable Value", ylab = "Frequency")
hist(df_1$Family_Size, main = "Histogram of Family Size", xlab = "Variable Value", ylab = "Frequency")

#Histogram A have postive skeweness (mean > median)
#we opted to use the median to fill the remaining NA's
df_1[is.na(df_1$Work_Experience),'Work_Experience'] <- median(df_1$Work_Experience, na.rm = TRUE)
#Histogram B have negative skeweness (median > mean)
#we opted to use the mean to fill the remaining NA's
df_1[is.na(df_1$Family_Size),'Family_Size'] <- mean(df_1$Family_Size, na.rm = TRUE) 
summary(df_1) #check if the NA's are filled

###############################       SETS DIVISION      ###############################################################################

#Set seed for reproducibility -> every time, our sample will be the same as before
set.seed(7) 

#Partition the data into training and test sets (70% vs 30%)
part_results <- partition(df_1$Segmentation, c(train=0.7, test=0.3))
str(part_results)

#Training set
trainset <- df_1[part_results$train, ]
str(trainset)

#Test set
testset <-df_1[part_results$test,]
str(testset)

numericas <- c(3,6,8)         #List of numerical variables

#Normalize data using z-score
z_score_function <- function(x) {(x - mean(x, na.rm=TRUE))/(sd(x,na.rm=TRUE))}

#Normalize numerical variables in training and test sets
trainset[, numericas] <- apply(trainset[, numericas],2, z_score_function)
testset[, numericas] <- apply(testset[, numericas],2, z_score_function)

########################################################################################################################################
###############################       MODELS      ######################################################################################

###############################       NAïVE BAYES      #################################################################################

#Fine tune
laplace_values <- c(0, 0.01, 0.1, 0.5, 1, 2, 5, 10)
usekernel_values <- c(TRUE, FALSE)  

best_accuracy_naive <- 0
best_laplace <- NULL
best_usekernel <- NULL

for (laplace in laplace_values) {
  for (usekernel in usekernel_values) {
    model <- naiveBayes(Segmentation ~ ., data = trainset, laplace = laplace, usekernel = usekernel)
    predictions <- predict(model, testset)
    accuracy <- sum(predictions == testset$Segmentation) / nrow(testset)
    cat("Laplace:", laplace, "Usekernel:", usekernel, "Accuracy:", accuracy, "\n")
    
    if (accuracy > best_accuracy_naive) {
      best_accuracy_naive <- accuracy
      best_laplace <- laplace
      best_usekernel <- usekernel
    }
  }
}

cat("Best Laplace:", best_laplace)
cat("Best Usekernel:", best_usekernel)
cat("Best Accuracy:", best_accuracy_naive)

##Final Model

#Train the Naïve Bayes model
naivebayes_model <- naiveBayes(Segmentation ~. ,trainset, laplace = 0.5, usekernel = TRUE)
str(naivebayes_model)

#Make predictions on the test set
predictions_naivebayes <- predict(naivebayes_model, testset)

#Confusion matrix to evaluate predictions
conf_matrix_naive <- table(predictions_naivebayes, testset$Segmentation)
conf_matrix_naive
#Calculate the Accuracy
accuracy_naivebayes <- sum(predictions_naivebayes==testset$Segmentation)/nrow(testset)
accuracy_naivebayes

#precision and Recall
precision_naive <- recall_naive <- numeric(length(rownames(conf_matrix_naive)))

for (i in 1:length(rownames(conf_matrix_naive))) {
  TP <- conf_matrix_naive[i, i]
  FP <- sum(conf_matrix_naive[i, ]) - TP
  FN <- sum(conf_matrix_naive[, i]) - TP
  
  precision_naive[i] <- TP / (TP + FP)
  recall_naive[i] <- TP / (TP + FN)
}

cat("Precision for each class:\n")
print(precision_naive)

cat("Recall for each class:\n")
print(recall_naive)
###############################       DECISION TREE      ###############################################################################

#Fine tune
cp_values <- seq(0.01, 0.1, by = 0.01)  
maxdepth_values <- c(3, 5, 7, 10, 15)
minsplit_values <- c(10, 20, 30, 40, 50)

best_accuracy_decisiontree <- 0
best_cp <- NULL
best_maxdepth <- NULL
best_minsplit <- NULL

for (cp in cp_values) {
  for (maxdepth in maxdepth_values) {
    for (minsplit in minsplit_values) {
      control <- rpart.control(cp = cp, maxdepth = maxdepth, minsplit = minsplit)
      model <- rpart(Segmentation ~ ., data = trainset, method = "class", control = control)
      predictions <- predict(model, testset, type = "class")
      accuracy <- sum(predictions == testset$Segmentation) / nrow(testset)
      cat("CP:", cp, "MaxDepth:", maxdepth, "MinSplit:", minsplit, "Accuracy:", best_accuracy_decisiontree, "\n")
      
      if (accuracy > best_accuracy_decisiontree) {
        best_accuracy_decisiontree <- accuracy
        best_cp <- cp
        best_maxdepth <- maxdepth
        best_minsplit <- minsplit
      }
    }
  }
}

cat("Best CP:", best_cp, "Best MaxDepth:", best_maxdepth, "Best MinSplit:", best_minsplit, 
    "Best Accuracy:", best_accuracy_decisiontree, "\n")

#Final Model

#Train the Decision Tree model
model_decision_tree <- rpart(Segmentation~., data=trainset, 
                             method = "class", control = rpart.control(cp = 0.01, maxdepth = 3, minsplit = 10))
predictions_decisiontree <- predict(model_decision_tree, testset, type = "class")
str(model_decision_tree)

#Confusion matrix to evaluate predictions
conf_matrix_tree<- table(predictions_decisiontree, testset$Segmentation)
conf_matrix_tree

#Calculate the Accuracy
accuracy_decisiontree <- sum(predictions_decisiontree == testset$Segmentation) / nrow(testset)
accuracy_decisiontree

#Plot the decision tree for visualization
rpart.plot(model_decision_tree)

#precision and Recall
precision_tree <- recall_tree <- numeric(length(rownames(conf_matrix_tree)))

for (i in 1:length(rownames(conf_matrix_tree))) {
  TP <- conf_matrix_tree[i, i]
  FP <- sum(conf_matrix_tree[i, ]) - TP
  FN <- sum(conf_matrix_tree[, i]) - TP
  
  precision_tree[i] <- TP / (TP + FP)
  recall_tree[i] <- TP / (TP + FN)
}

cat("Precision for each class:\n")
print(precision_tree)

cat("Recall for each class:\n")
print(recall_tree)

###############################       K-NN      ########################################################################################

#Convert categorical variables to binary (dummy) variables
selected_columns <- names(df_1)[c(1,2,4,5,7,9)]
trainset_dummy <- dummy_cols(trainset, select_columns = selected_columns, 
                             remove_first_dummy = TRUE, remove_selected_columns = TRUE)
testset_dummy <- dummy_cols(testset, select_columns = selected_columns, 
                            remove_first_dummy = TRUE, remove_selected_columns = TRUE)
str(trainset_dummy)

#Determine the best value of k for K-NN
k_values <- seq(1, 50)                 #best k
best_accuracy <- 0                     #best accuracy
best_k <- NULL        
accuracy_values <- numeric(length(k_values))

#Loop to find best k value
for (i in seq_along(k_values)) {
  k <- k_values[i]
  #Train the K-NN model
  model <- knn(trainset_dummy[,-4], testset_dummy[,-4], factor(trainset_dummy[,"Segmentation"]), k = k)
  #Calculate accuracy
  accuracy <- sum(model == factor(testset_dummy$Segmentation))/nrow(testset_dummy)
  accuracy_values[i] <- accuracy
  #Print progress
  cat("k:", k, "Accuracy:", accuracy, "\n")
  #Update best model if current model is better
  if (accuracy > best_accuracy) {
    best_accuracy <- accuracy
    best_k <- k
  }
}

cat("Best k:", best_k, "Best Accuracy:", best_accuracy, "\n")

#Plotting K-NN accuracy for different values of k
accuracy_df <- data.frame(k = k_values, accuracy = accuracy_values)

ggplot(accuracy_df, aes(x = k, y = accuracy)) +
  geom_line() +
  geom_point() +
  geom_point(aes(x = best_k, y = best_accuracy), color = 'red', size = 3) +
  ggtitle("KNN Accuracy for Different Values of k") +
  xlab("Number of Neighbors (k)") +
  ylab("Accuracy") +
  theme_minimal()

#Final Model

#Best K-NN model is the one with k=24, where maximization of the accuracy is obtained 
modelo_KNN <- knn(trainset_dummy[,-4], testset_dummy[,-4], factor(trainset_dummy[,"Segmentation"]), k = 24)
str(modelo_KNN)

#Confusion matrix to evaluate predictions
conf_matrix_knn <- table(modelo_KNN, factor(testset_dummy$Segmentation))
conf_matrix_knn

#Calculate the Accuracy
accuracy_knn <- sum(modelo_KNN== factor(testset_dummy$Segmentation))/nrow(testset_dummy)
accuracy_knn

#precision and Recall
precision_knn <- recall_knn <- numeric(length(rownames(conf_matrix_knn)))

for (i in 1:length(rownames(conf_matrix_knn))) {
  TP <- conf_matrix_knn[i, i]
  FP <- sum(conf_matrix_knn[i, ]) - TP
  FN <- sum(conf_matrix_knn[, i]) - TP
  
  precision_knn[i] <- TP / (TP + FP)
  recall_knn[i] <- TP / (TP + FN)
}

cat("Precision for each class:\n")
print(precision_knn)

cat("Recall for each class:\n")
print(recall_knn)

###############################       RANDOM FOREST      ###############################################################################

#Define a maximum value for mtry and ntree
max_mtry <- min(5, ncol(trainset))  
mtry_values <- seq(1, max_mtry, by = 1)
ntree_values <- seq(50, 200, by = 10)

#To store values for best results
best_oob_error <- Inf
best_mtry <- 0
best_ntree <- 0
#Loop to find the best mtry and ntree values
for (mtry in mtry_values) {
  for (ntree in ntree_values) {
    temp.model <- randomForest(Segmentation ~ ., data = trainset, mtry = mtry, ntree = ntree)
    oob_error <- temp.model$err.rate[nrow(temp.model$err.rate), "OOB"]
    if (oob_error < best_oob_error) {
      best_oob_error <- oob_error
      best_mtry <- mtry
      best_ntree <- ntree
    }
  }
}

cat("Best OOB Error:", best_oob_error, "\n") #0.466 hence accuracy is 0.53
cat("Best mtry:", best_mtry, "\n")  #2
cat("Best ntree:", best_ntree, "\n") #90

#Best Random Forest Model
#Train the best random forest model using the optimal number of trees (90) and mtry (2)
model_randomforest <- randomForest(Segmentation ~ ., data = trainset, mtry = 2, ntree = 90)
model_randomforest

#Make prediction on test set
predictions_randomforest <- predict(model_randomforest, testset)

#Confusion matrix to evaluate predictions
conf_matrix_randomforest <- table(predictions_randomforest, testset$Segmentation)
conf_matrix_randomforest

#Calculate the Accuracy
accuracy_randomforest <- sum(predictions_randomforest==testset$Segmentation)/nrow(testset)
accuracy_randomforest #0.52

#precision and Recall
precision_randomforest <- recal_randomforest <- numeric(length(rownames(conf_matrix_randomforest)))

for (i in 1:length(rownames(conf_matrix_randomforest))) {
  TP <- conf_matrix_randomforest[i, i]
  FP <- sum(conf_matrix_randomforest[i, ]) - TP
  FN <- sum(conf_matrix_randomforest[, i]) - TP
  
  precision_randomforest[i] <- TP / (TP + FP)
  recal_randomforest[i] <- TP / (TP + FN)
}

cat("Precision for each class:\n")
print(precision_randomforest)

cat("Recall for each class:\n")
print(recal_randomforest)
###############################       SUPPORT VECTOR MACHINE      ######################################################################

C_values <- c(0.01, 0.1, 1, 10, 100, 200)
gamma_values <- c(0.001, 0.01, 0.1, 1)

best_accuracy_SVM <- 0
best_C <- 0
best_gamma <- 0

for (C in C_values) {
  for (gamma in gamma_values) {
    svm_model <- svm(Segmentation ~ ., data = trainset, kernel = "radial", cost = C, gamma = gamma)
    predictions <- predict(svm_model, testset)
    accuracy <- sum(predictions==testset$Segmentation)/nrow(testset)
    cat("Best Gamma:", best_gamma,"Best C:", best_C, "Best Accuracy:", best_accuracy_SVM, "\n")
    
    if (accuracy > best_accuracy_SVM) {
      best_accuracy_SVM <- accuracy
      best_C <- C
      best_gamma <- gamma
    }
  }
}

cat("Best Gamma:", best_gamma,"Best C:", best_C, "Best Accuracy:", best_accuracy_SVM, "\n") #0.01 #100 #0.526

#Train the SVM Model
svm_model <- svm(formula = Segmentation ~ ., data = trainset, kernel = "radial", cost = 100, gamma = 0.01)

#Make predictions on the test set
svm_predictions <- predict(svm_model, testset)

#Confusion matrix to evaluate predictions
conf_matrix_svm <- table(svm_predictions, testset$Segmentation)
conf_matrix_svm

#Calculate the Accuracy
accuracy_svm <- sum(svm_predictions==testset$Segmentation)/nrow(testset)
accuracy_svm

#precision and Recall
precision_svm <- recal_svm <- numeric(length(rownames(conf_matrix_svm)))

for (i in 1:length(rownames(conf_matrix_svm))) {
  TP <- conf_matrix_svm[i, i]
  FP <- sum(conf_matrix_svm[i, ]) - TP
  FN <- sum(conf_matrix_svm[, i]) - TP
  
  precision_svm[i] <- TP / (TP + FP)
  recal_svm[i] <- TP / (TP + FN)
}

cat("Precision for each class:\n")
print(precision_svm)

cat("Recall for each class:\n")
print(recal_svm)





