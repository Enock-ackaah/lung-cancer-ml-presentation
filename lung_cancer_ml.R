# Predicting Lung Cancer Using Machine Learning
# Author: Frederick Asamoah, Enock Ackaah, Jing Sun,
Yongqing Cai
# Course Project – Graduate Statistics
# Description: ML models for lung cancer classification



## Predicting and classify lung cancer 
## by Frederick Asamoah, Enock Ackaah, Jing Sun, Yongqing Cai

#########################################################  
################### 1. Packages used
#########################################################  

library(pROC)
library(tidyverse)
library(caret)
library(tree)
library(rpart)
library(rpart.plot)
library(ipred)
library(randomForest)
library(gbm)
library(e1071)
library(dplyr)
library(neuralnet)

#########################################################  
################### 2. Dataset and Prelim
#########################################################  

LC = read.csv("D:\\UA STAT\\Fall 2025\\STAT584 Machine Learning\\project\\lung cancer.csv")
cat("Dataset Dimensions:\n")
print(dim(LC))        # Rows and columns
cat("Number of rows:", nrow(LC), "\n")
cat("Number of columns:", ncol(LC), "\n")
cat("Column names:\n")
print(names(LC))
cat("Summary statistics:\n")
print(summary(LC))
# chi-square association
LC_pre <- LC %>% mutate(LUNG_CANCER=ifelse(LUNG_CANCER=="YES", 1, 0))
model <- glm(LUNG_CANCER ~ ., data = LC_pre, family = binomial)
summary(model)
# Check missing value
summary(LC)
sum(is.na(LC))

#########################################################  
################### 3. Logistic Regression Model
#########################################################   

# Identify target variable (assumes last column)
target_name <- names(LC)[ncol(LC)]

# Ensure target is binary factor
unique_vals <- unique(LC[[target_name]])
if(length(unique_vals) != 2) stop("Target must have exactly 2 unique values for binary classification.")
LC[[target_name]] <- factor(LC[[target_name]], levels = unique_vals)

# Train-test split (70%-30%)
set.seed(123)
n <- nrow(LC)
train_index <- sample(1:n, 0.7 * n)
train <- LC[train_index, ]
test  <- LC[-train_index, ]

# Check train and test datasets
cat("Training set dimensions:\n")
print(dim(train))
cat("Test set dimensions:\n")
print(dim(test))
cat("Target distribution in training set:\n")
print(table(train[[target_name]]))
cat("Target distribution in test set:\n")
print(table(test[[target_name]]))

# Fit logistic regression
model <- glm(
  formula = as.formula(paste(target_name, "~ .")),
  data = train,
  family = binomial
)

# Predictions
# Probabilities
prob <- predict(model, newdata = test, type = "response")

# Convert probabilities to predicted class
pred_class <- ifelse(prob > 0.5, levels(LC[[target_name]])[2], levels(LC[[target_name]])[1])
pred_class <- factor(pred_class, levels = levels(LC[[target_name]]))

# Actual values
actual <- factor(test[[target_name]], levels = levels(LC[[target_name]]))

# Metrics (Corrected RMSE)
# Convert actual factor to numeric 0/1
actual_num <- ifelse(actual == levels(actual)[1], 0, 1)

# RMSE
RMSE <- sqrt(mean((actual_num - prob)^2))

# AUC
AUC <- auc(actual, prob)

# Confusion matrix
cm <- confusionMatrix(pred_class, actual)

# Print results
cat("\nModel Performance Metrics:\n")
cat("RMSE:", round(RMSE, 4), "\n")
cat("AUC:", round(AUC, 4), "\n")
cat("Confusion Matrix:\n")
print(cm)

# ROC curve
roc_obj <- roc(actual, prob)
plot(roc_obj, main = "ROC Curve", col = "blue")

#########################################################  
################### 4. Decision Tree Model
#########################################################   
LC_DT <- LC %>%
  rename(resp=LUNG_CANCER) %>%
  relocate(resp) %>%
  mutate(resp=as.factor(resp),
         GENDER=as.factor(GENDER),
         SMOKING=as.factor(SMOKING),
         YELLOW_FINGERS=as.factor(YELLOW_FINGERS),
         ANXIETY=as.factor(ANXIETY),
         PEER_PRESSURE=as.factor(PEER_PRESSURE),
         CHRONIC_DISEASE=as.factor(CHRONIC.DISEASE),
         FATIGUE=as.factor(FATIGUE),
         ALLERGY=as.factor(ALLERGY),
         WHEEZING=as.factor(WHEEZING),
         ALCOHOL_CONSUMING=as.factor(ALCOHOL.CONSUMING),
         COUGHING=as.factor(COUGHING),
         SHORTNESS_OF_BREATH=as.factor(SHORTNESS.OF.BREATH),
         SWALLOWING_DIFFICULTY=as.factor(SWALLOWING.DIFFICULTY),
         CHEST_PAIN=as.factor(CHEST.PAIN))
Orig = LC_DT

# 5 fold CV
source("https://nmimoto.github.io/R/ML-00.txt")   # load CreateCV()
CreateCV(Orig, numFolds=5, seed=7211)

# CV.train fit
tree31 <- tree(resp ~., CV.train[[1]])
summary(tree31)
plot(tree31)
text(tree31, pretty=1, cex=.7)

tree32 <- tree(resp ~., CV.train[[2]])
summary(tree32)
plot(tree32)
text(tree32, pretty=0, cex=.7)

tree33 <- tree(resp ~., CV.train[[3]])
summary(tree33)
plot(tree33)
text(tree33, pretty=0, cex=.7)

tree34 <- tree(resp ~., CV.train[[4]])
summary(tree34)
plot(tree34)
text(tree34, pretty=0, cex=.7)

tree35 <- tree(resp ~., CV.train[[5]])
summary(tree35)
plot(tree35)
text(tree35, pretty=0, cex=.7)

# Decision Tree Growing
tree00<- rpart(resp~., data=Train.set)
summary(tree00)
rpart.plot(tree00)
plot(tree00)
text(tree00)

# Grow and Prune
tree1 = tree(resp~., Train.set)
summary(tree1)
tree1

plot(tree1)
text(tree1, pretty=0, cex=1)

threshold = .5
my.seed = 123

Train.prob = predict(tree1, type="vector")[,"YES"]
Train.pred = ifelse(Train.prob > threshold, "YES", "NO")
Test.prob  = predict(tree1, Test.set, type="vector")[,"YES"]
Test.pred  = ifelse(Test.prob > threshold, "YES", "NO")

# Pruning the tree
set.seed(my.seed)
cv.for.pruning = cv.tree(tree1, FUN=prune.misclass, K=5)
names(cv.for.pruning)

plot(cv.for.pruning$size, cv.for.pruning$dev, type="b")
plot(cv.for.pruning$k,    cv.for.pruning$dev, type="b")

cv.for.pruning

pruned1 = prune.tree(tree1, best=5)
plot(pruned1)
text(pruned1, pretty=0, cex=1)

# visualize the fit
Chosen.model <- pruned1
threshold = .5

# Check the training set accuracy
Train.prob = predict(Chosen.model, type="vector")[,"YES"]
Train.pred = ifelse(Train.prob > threshold, "YES", "NO")  # Turn the fitted values to Up/Down using the threshold
Test.prob  = predict(Chosen.model, Test.set, type="vector")[,"YES"]
Test.pred  = ifelse(Test.prob > threshold, "YES", "NO")
CM.train <- caret::confusionMatrix(factor(Train.pred), factor(as.matrix(Train.resp)), positive="YES")
CM.test <- caret::confusionMatrix(factor(Test.pred), factor(as.matrix(Test.resp)), positive="YES")

CM.train 
CM.train$table

CM.train[["byClass"]][["Sensitivity"]]
CM.train[["byClass"]][["Specificity"]]

CM.test
CM.test$table

colSums(CM.test$table) / sum(colSums(CM.test$table))    # % of Actual Yes/No
rowSums(CM.test$table) / sum(rowSums(CM.test$table))    # % of predicted Yes/No

# Output ROC curve and AUC for all threshold
# Training Set
plot.roc(factor(as.matrix(Train.resp)),  Train.prob, levels=c("NO", "YES"))
# point corresponding to CM.train
abline(h=CM.train[["byClass"]][["Sensitivity"]], v=CM.train[["byClass"]][["Specificity"]], col="red")
auc.train = auc(factor(as.matrix(Train.resp)), Train.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Train AUC=",round(auc.train, 3)))

# Test Set
plot.roc(factor(as.matrix(Test.resp)),  Test.prob, levels=c("NO", "YES"))
# point corresponding to CM.test
abline(h=CM.test[["byClass"]][["Sensitivity"]], v=CM.test[["byClass"]][["Specificity"]], col="red")
auc.test = auc(factor(as.matrix(Test.resp)), Test.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Test AUC=",round(auc.test, 3)))

GrowAndPrune = c(auc.train, auc.test)
GrowAndPrune

# Training ROC and Test ROC side by side
layout(matrix(1:2, 1, 2))
plot.roc(factor(as.matrix(Train.resp)),  Train.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Train AUC=",round(auc.train, 3)))
plot.roc(factor(as.matrix(Test.resp)),  Test.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Test AUC=",round(auc.test, 3)))
layout(1)

GrowAndPrune

# Bagging 
set.seed(my.seed)
treeBag01 = bagging(resp~., data=Train.set, nbagg=500,
                    coob = TRUE,
                    control = rpart.control(minsplit = 2, cp = 0))
treeBag01

# visualize the fit
Chosen.model <- treeBag01

# Check the training set accuracy
Train.prob = predict(Chosen.model, type="prob")[,"YES"]
Train.pred = ifelse(Train.prob > threshold, "YES", "NO")  # Turn the fitted values to Up/Down using the threshold
Test.prob = predict(Chosen.model, newdata=Test.set, type="prob")[,"YES"]
Test.pred = ifelse(Test.prob > threshold, "YES", "NO")  # Turn the fitted values to Up/Down using the threshold
CM.train <- caret::confusionMatrix(factor(Train.pred), factor(as.matrix(Train.resp)), positive="YES")
CM.test  <- caret::confusionMatrix(factor(Test.pred),  factor(as.matrix(Test.resp)), positive="YES")

CM.train 
CM.train$table

CM.train[["byClass"]][["Sensitivity"]]
CM.train[["byClass"]][["Specificity"]]

CM.test
CM.test$table

colSums(CM.test$table) / sum(colSums(CM.test$table))    # % of Actual Yes/No
rowSums(CM.test$table) / sum(rowSums(CM.test$table))    # % of predicted Yes/No

# Output ROC curve and AUC for all threshold
#- Training Set
plot.roc(factor(as.matrix(Train.resp)),  Train.prob, levels=c("NO", "YES"))
# point corresponding to CM.train
abline(h=CM.train[["byClass"]][["Sensitivity"]], v=CM.train[["byClass"]][["Specificity"]], col="red")
auc.train = auc(factor(as.matrix(Train.resp)), Train.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Train AUC=",round(auc.train, 3)))

#- Test Set
plot.roc(factor(as.matrix(Test.resp)),  Test.prob, levels=c("NO", "YES"))
# point corresponding to CM.test
abline(h=CM.test[["byClass"]][["Sensitivity"]], v=CM.test[["byClass"]][["Specificity"]], col="red")
auc.test = auc(factor(as.matrix(Test.resp)), Test.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Test AUC=",round(auc.test, 3)))

Bagging = c(auc.train, auc.test)
Bagging

# Training ROC and Test ROC side by side
layout(matrix(1:2, 1, 2))
plot.roc(factor(as.matrix(Train.resp)),  Train.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Train AUC=",round(auc.train, 3)))
plot.roc(factor(as.matrix(Test.resp)),  Test.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Test AUC=",round(auc.test, 3)))
layout(1)

Bagging

# Random Forest
set.seed(my.seed)
treeBag01 = randomForest(resp~., data=Train.set, mtry=13, ntree=500, importance=TRUE)
treeBag01

importance (treeBag01)
varImpPlot (treeBag01)

# visualize the fit
Chosen.model <- treeBag01

# Check the training set accuracy
Train.prob = predict(Chosen.model, type="prob")[,"YES"]
Train.pred = ifelse(Train.prob > threshold, "YES", "NO")  # Turn the fitted values to Up/Down using the threshold
Test.prob = predict(Chosen.model, newdata=Test.set, type="prob")[,"YES"]
Test.pred = ifelse(Test.prob > threshold, "YES", "NO")  # Turn the fitted values to Up/Down using the threshold
CM.train <- caret::confusionMatrix(factor(Train.pred), factor(as.matrix(Train.resp)), positive="YES")
CM.test  <- caret::confusionMatrix(factor(Test.pred),  factor(as.matrix(Test.resp)), positive="YES")

CM.train 
CM.train$table 

CM.train[["byClass"]][["Sensitivity"]]
CM.train[["byClass"]][["Specificity"]]

CM.test
CM.test$table 

colSums(CM.test$table) / sum(colSums(CM.test$table))    # % of Actual Yes/No
rowSums(CM.test$table) / sum(rowSums(CM.test$table))    # % of predicted Yes/No

# Output ROC curve and AUC for all threshold
# Training Set
plot.roc(factor(as.matrix(Train.resp)),  Train.prob, levels=c("NO", "YES"))
abline(h=CM.train[["byClass"]][["Sensitivity"]], v=CM.train[["byClass"]][["Specificity"]], col="red")
auc.train = auc(factor(as.matrix(Train.resp)), Train.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Train AUC=",round(auc.train, 3)))

# Test Set
plot.roc(factor(as.matrix(Test.resp)),  Test.prob, levels=c("NO", "YES"))
abline(h=CM.test[["byClass"]][["Sensitivity"]], v=CM.test[["byClass"]][["Specificity"]], col="red")
auc.test = auc(factor(as.matrix(Test.resp)), Test.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Test AUC=",round(auc.test, 3)))

RandomForest = c(auc.train, auc.test)
RandomForest

# Training ROC and Test ROC side by side
layout(matrix(1:2, 1, 2))
plot.roc(factor(as.matrix(Train.resp)),  Train.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Train AUC=",round(auc.train, 3)))
plot.roc(factor(as.matrix(Test.resp)),  Test.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Test AUC=",round(auc.test, 3)))
layout(1)

RandomForest

# Boosting
set.seed(my.seed)
treeBT01 = gbm::gbm(as.numeric(resp=="YES") ~., data=Train.set,
                    distribution="bernoulli",   # for binary classification
                    n.trees=5000,
                    interaction.depth=4)
summary(treeBT01)

plot(treeBT01, i="YELLOW_FINGERS")
plot(treeBT01, i="FATIGUE")

# visualize the fit
Chosen.model <- treeBT01

# Check the training set accuracy
Train.prob = predict(Chosen.model, type="response", n.trees=1000)
Train.pred = ifelse(Train.prob > threshold, "YES", "NO")  # Turn the fitted values to Up/Down using the threshold
Test.prob = predict(Chosen.model, newdata=Test.set, type="response", n.trees=1000)
Test.pred = ifelse(Test.prob > threshold, "YES", "NO")  # Turn the fitted values to Up/Down using the threshold
CM.train <- caret::confusionMatrix(factor(Train.pred), factor(as.matrix(Train.resp)), positive="YES")
CM.test  <- caret::confusionMatrix(factor(Test.pred),  factor(as.matrix(Test.resp)), positive="YES")

CM.train
CM.train$table

CM.train[["byClass"]][["Sensitivity"]]
CM.train[["byClass"]][["Specificity"]]

CM.test
CM.test$table

colSums(CM.test$table) / sum(colSums(CM.test$table))    # % of Actual Yes/No
rowSums(CM.test$table) / sum(rowSums(CM.test$table))    # % of predicted Yes/No

# Output ROC curve and AUC for all threshold
# Training Set
plot.roc(factor(as.matrix(Train.resp)),  Train.prob, levels=c("NO", "YES"))
abline(h=CM.train[["byClass"]][["Sensitivity"]], v=CM.train[["byClass"]][["Specificity"]], col="red")
auc.train = auc(factor(as.matrix(Train.resp)), Train.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Train AUC=",round(auc.train, 3)))

# Test Set
plot.roc(factor(as.matrix(Test.resp)),  Test.prob, levels=c("NO", "YES"))
abline(h=CM.test[["byClass"]][["Sensitivity"]], v=CM.test[["byClass"]][["Specificity"]], col="red")
auc.test = auc(factor(as.matrix(Test.resp)), Test.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Test AUC=",round(auc.test, 3)))

Boosting = c(auc.train, auc.test)
Boosting

# Training ROC and Test ROC side by side
layout(matrix(1:2, 1, 2))
plot.roc(factor(as.matrix(Train.resp)),  Train.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Train AUC=",round(auc.train, 3)))
plot.roc(factor(as.matrix(Test.resp)),  Test.prob, levels=c("NO", "YES"))
text(.2, .2, paste("Test AUC=",round(auc.test, 3)))
layout(1)

Boosting

# Compare all DT models
GrowAndPrune
Bagging
RandomForest
Boosting

# Threshold Picker
cost.list = c(0,0,1,2)/3
threshold.list = seq(0.01,.99,.01)
cost=0
Chosen.model <- pruned1;    Test.prob  = predict(Chosen.model, Test.set, type="vector")[,"YES"]
# Chosen.model <- treeBag01;  Test.prob = predict(Chosen.model, newdata=Test.set, type="prob")[,"YES"]
# Chosen.model <- treeBT01;   Test.prob = predict(Chosen.model, newdata=Test.set, type="response", n.trees=1000)

for (i in 1:length(threshold.list)){
  threshold = threshold.list[i]
  Test.pred  = ifelse(Test.prob  > threshold, "YES", "NO")
  CM.test  <- confusionMatrix(factor(Test.pred),
                              factor(as.matrix(Test.resp)),
                              positive="YES")
  TP = CM.test$table[2,2]   # True  Pos
  TN = CM.test$table[1,1]   # True  Neg
  FP = CM.test$table[2,1]   # False Pos
  FN = CM.test$table[1,2]   # False Neg
  cost[i] = sum(c(TP, TN, FP, FN) * cost.list)
}
plot(threshold.list, cost, xlab="threshold")

cost.list
which.min(cost)
min(cost)
threshold.list[which.min(cost)]

#########################################################  
################### 5. Support Vector Machine Models
#########################################################  
my.seed=123
lung = LC
names(lung) <- make.names(names(lung))
# Convert response to factor
lung$LUNG_CANCER <- factor(lung$LUNG_CANCER)

# Keep only numeric predictor columns
num_pred_cols <- sapply(lung, is.numeric)

# Create dataset with numeric predictors only
lung_num <- lung[, num_pred_cols]

# Add back the response (factor)
lung_num$LUNG_CANCER <- lung$LUNG_CANCER

str(lung_num)

# Train / Test Split (70% / 30%)
set.seed(my.seed)
n <- nrow(lung_num)
index <- sample(1:n, size = round(0.7 * n))

Train.set <- lung_num[index, ]
Test.set  <- lung_num[-index, ]

table(Train.set$LUNG_CANCER)
table(Test.set$LUNG_CANCER)

# Scale numeric predictors (using training-set mean/SD)
# Identify numeric columns in Train/Test (predictors only)
num_cols_train <- sapply(Train.set, is.numeric)    # includes only numeric
num_cols_train["LUNG_CANCER"] <- FALSE             # exclude response

# Get centers and scales from training set
center_vals <- sapply(Train.set[, num_cols_train, drop = FALSE], mean)
scale_vals  <- sapply(Train.set[, num_cols_train, drop = FALSE], sd)

# Scale training predictors
Train.x.scaled <- scale(
  Train.set[, num_cols_train, drop = FALSE],
  center = center_vals,
  scale  = scale_vals
)

# Scale test predictors using *training* mean and sd
Test.x.scaled <- scale(
  Test.set[, num_cols_train, drop = FALSE],
  center = center_vals,
  scale  = scale_vals
)

# Rebuild final Train/Test data frames for SVM:
Train.svm <- data.frame(Train.x.scaled, LUNG_CANCER = Train.set$LUNG_CANCER)
Test.svm  <- data.frame(Test.x.scaled,  LUNG_CANCER = Test.set$LUNG_CANCER)

str(Train.svm)
str(Test.svm)

# Positive class (for probabilities, RMSE, R2)
positive_class <- levels(Train.svm$LUNG_CANCER)[2]

# Helper to compute RMSE and R^2 from probabilities
calc_RMSE_R2 <- function(actual_factor, prob_vec) {
  actual_num <- ifelse(actual_factor == levels(actual_factor)[2], 1, 0)
  RMSE <- sqrt(mean((actual_num - prob_vec)^2))
  SSE  <- sum((actual_num - prob_vec)^2)
  SST  <- sum((actual_num - mean(actual_num))^2)
  R2   <- 1 - SSE / SST
  list(RMSE = RMSE, R2 = R2)
}

# SVM with Linear Kernel (Tuned01)
set.seed(my.seed)

Tuned01 <- e1071::tune(
  svm,
  LUNG_CANCER ~ ., 
  data   = Train.svm,
  kernel = "linear",
  ranges = list(
    cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100, 1000)
  ),
  probability = TRUE,              # <-- needed for probs
  scale = FALSE,                   # already scaled
  tunecontrol = tune.control(cross = 5)  # 5-fold CV
)

summary(Tuned01)
best.linear <- Tuned01$best.model
best.linear

# ---- Linear SVM Performance ----
# Train probabilities & class preds
linear_pred_train <- predict(best.linear, Train.svm, probability = TRUE)
linear_prob_train <- attr(linear_pred_train, "probabilities")[, positive_class]

linear_pred_test  <- predict(best.linear, Test.svm, probability = TRUE)
linear_prob_test  <- attr(linear_pred_test, "probabilities")[, positive_class]

# AUC
roc_linear_train <- roc(Train.svm$LUNG_CANCER, linear_prob_train)
auc_linear_train <- auc(roc_linear_train)

roc_linear_test  <- roc(Test.svm$LUNG_CANCER, linear_prob_test)
auc_linear_test  <- auc(roc_linear_test)

# RMSE & R^2
lin_train_metrics <- calc_RMSE_R2(Train.svm$LUNG_CANCER, linear_prob_train)
lin_test_metrics  <- calc_RMSE_R2(Test.svm$LUNG_CANCER,  linear_prob_test)

linear_RMSE_train <- lin_train_metrics$RMSE
linear_R2_train   <- lin_train_metrics$R2
linear_RMSE_test  <- lin_test_metrics$RMSE
linear_R2_test    <- lin_test_metrics$R2

# Confusion Matrix - Test
table(Predicted = linear_pred_test, Actual = Test.svm$LUNG_CANCER)

# Optional: Plot ROC curve for test
plot(roc_linear_test, main = "Linear SVM - Test ROC")

# SVM with Radial (RBF) Kernel (Tuned02)
set.seed(my.seed)
Tuned02 <- e1071::tune(
  svm,
  LUNG_CANCER ~ .,
  data   = Train.svm,
  kernel = "radial",
  ranges = list(
    gamma = 2^(-1:4),
    cost  = c(0.001, 0.01, 0.1, 1, 5, 10, 100, 1000)
  ),
  probability = TRUE,
  scale = FALSE,  # already scaled
  tunecontrol = tune.control(cross = 5)
)

summary(Tuned02)
best.rbf <- Tuned02$best.model
best.rbf

# Radial SVM Performance
rbf_pred_train <- predict(best.rbf, Train.svm, probability = TRUE)
rbf_prob_train <- attr(rbf_pred_train, "probabilities")[, positive_class]

rbf_pred_test  <- predict(best.rbf, Test.svm, probability = TRUE)
rbf_prob_test  <- attr(rbf_pred_test, "probabilities")[, positive_class]

# AUC
roc_rbf_train <- roc(Train.svm$LUNG_CANCER, rbf_prob_train)
auc_rbf_train <- auc(roc_rbf_train)

roc_rbf_test  <- roc(Test.svm$LUNG_CANCER, rbf_prob_test)
auc_rbf_test  <- auc(roc_rbf_test)

# RMSE & R^2
rbf_train_metrics <- calc_RMSE_R2(Train.svm$LUNG_CANCER, rbf_prob_train)
rbf_test_metrics  <- calc_RMSE_R2(Test.svm$LUNG_CANCER,  rbf_prob_test)

rbf_RMSE_train <- rbf_train_metrics$RMSE
rbf_R2_train   <- rbf_train_metrics$R2
rbf_RMSE_test  <- rbf_test_metrics$RMSE
rbf_R2_test    <- rbf_test_metrics$R2

# Confusion Matrix - Test
table(Predicted = rbf_pred_test, Actual = Test.svm$LUNG_CANCER)

# Optional: Plot ROC curve for test
plot(roc_rbf_test, main = "Radial SVM - Test ROC")

# SVM with Polynomial Kernel (Tuned03)
set.seed(my.seed)
Tuned03 <- e1071::tune(
  svm,
  LUNG_CANCER ~ .,
  data   = Train.svm,
  kernel = "polynomial",
  ranges = list(
    degree = 1:3,
    gamma  = 2^(2:5),
    coef0  = c(0, 0.5, 1, 2),
    cost   = c(0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100)
  ),
  probability = TRUE,
  scale = FALSE,  # already scaled
  tunecontrol = tune.control(cross = 5)
)

summary(Tuned03)
best.poly <- Tuned03$best.model
best.poly

# Polynomial SVM Performance
poly_pred_train <- predict(best.poly, Train.svm, probability = TRUE)
poly_prob_train <- attr(poly_pred_train, "probabilities")[, positive_class]

poly_pred_test  <- predict(best.poly, Test.svm, probability = TRUE)
poly_prob_test  <- attr(poly_pred_test, "probabilities")[, positive_class]

# AUC
roc_poly_train <- roc(Train.svm$LUNG_CANCER, poly_prob_train)
auc_poly_train <- auc(roc_poly_train)

roc_poly_test  <- roc(Test.svm$LUNG_CANCER, poly_prob_test)
auc_poly_test  <- auc(roc_poly_test)

# RMSE & R^2
poly_train_metrics <- calc_RMSE_R2(Train.svm$LUNG_CANCER, poly_prob_train)
poly_test_metrics  <- calc_RMSE_R2(Test.svm$LUNG_CANCER,  poly_prob_test)

poly_RMSE_train <- poly_train_metrics$RMSE
poly_R2_train   <- poly_train_metrics$R2
poly_RMSE_test  <- poly_test_metrics$RMSE
poly_R2_test    <- poly_test_metrics$R2

# Confusion Matrix - Test
table(Predicted = poly_pred_test, Actual = Test.svm$LUNG_CANCER)

# Optional: Plot ROC curve for test
plot(roc_poly_test, main = "Polynomial SVM - Test ROC")

# Summary Table of Metrics
Perf.summary <- data.frame(
  Model       = c("Linear SVM",      "Radial SVM",      "Polynomial SVM"),
  Train.RMSE  = c(linear_RMSE_train, rbf_RMSE_train,    poly_RMSE_train),
  Test.RMSE   = c(linear_RMSE_test,  rbf_RMSE_test,     poly_RMSE_test),
  Train.R2    = c(linear_R2_train,   rbf_R2_train,      poly_R2_train),
  Test.R2     = c(linear_R2_test,    rbf_R2_test,       poly_R2_test),
  Train.AUC   = c(auc_linear_train,  auc_rbf_train,     auc_poly_train),
  Test.AUC    = c(auc_linear_test,   auc_rbf_test,      auc_poly_test)
)

Perf.summary

#########################################################  
################### 6.Neural Network models.
#########################################################  
LC_NN = LC
LC_NN <- as_tibble(LC_NN)
lapply(LC_NN,is.numeric)

LC_NN2 <- LC_NN %>%
  mutate(LUNG_CANCER=ifelse(LUNG_CANCER=="YES", "Yes", "No"))%>%
  mutate(GENDER=ifelse(GENDER=="M", 0, 1)) %>%       # MALE=0,FEMALE=1
  mutate(SMOKING=ifelse(SMOKING==1, 0, 1)) %>% 
  mutate(YELLOW_FINGERS=ifelse(YELLOW_FINGERS==1, 0, 1)) %>%  
  mutate(ANXIETY=ifelse(ANXIETY==1, 0, 1)) %>%   
  mutate(PEER_PRESSURE=ifelse(PEER_PRESSURE==1, 0, 1)) %>%   
  mutate(CHRONIC.DISEASE=ifelse(CHRONIC.DISEASE==1, 0, 1)) %>%   
  mutate(FATIGUE=ifelse(FATIGUE==1, 0, 1)) %>%   
  mutate(ALLERGY=ifelse(ALLERGY==1, 0, 1)) %>%  
  mutate(WHEEZING=ifelse(WHEEZING==1, 0, 1)) %>%   
  mutate(ALCOHOL.CONSUMING=ifelse(ALCOHOL.CONSUMING==1, 0, 1)) %>%  
  mutate(COUGHING=ifelse(COUGHING==1, 0, 1)) %>%  
  mutate(SHORTNESS.OF.BREATH=ifelse(SHORTNESS.OF.BREATH==1, 0, 1)) %>% 
  mutate(SWALLOWING.DIFFICULTY=ifelse(SWALLOWING.DIFFICULTY==1, 0, 1)) %>% 
  mutate(CHEST.PAIN=ifelse(CHEST.PAIN==1, 0, 1)) %>% 
  rename(resp=LUNG_CANCER) %>%
  mutate(resp=as.factor(resp)) %>%
  relocate(resp)
lapply(LC_NN2,is.numeric)
lapply(LC_NN2,is.factor)

# Scalse all variables to be used in NN
LC_NN3 <- LC_NN2 %>% mutate_at(-1, list(~base::scale(.)[,]))
Orig <- LC_NN3

# 5 fold CV
CreateCV(Orig, numFolds=5, seed=20251201)
my.seed=20251201

# Neural Network 5-fold CV
sigmoid <- function(x){1 / (1 + exp(-x))}
AUCs <- MSE.valid <- matrix(0, 5, 2)
colnames(AUCs) = c("Train AUC", "Valid AUC")
for (k in 1:5) {
  set.seed(my.seed)
  Fit00 = neuralnet::neuralnet(resp ~ SMOKING + YELLOW_FINGERS + PEER_PRESSURE + CHRONIC.DISEASE +
                                 FATIGUE + ALLERGY + COUGHING + SWALLOWING.DIFFICULTY ,
                               CV.train[[k]],
                               hidden=3, 
                               learningrate=5e-3,
                               act.fct=sigmoid,
                               linear.output=FALSE)
  
  Train.prob = predict(Fit00, newdata=CV.train[[k]], type="response")[,1]
  Valid.prob = predict(Fit00, newdata=CV.valid[[k]], type="response")[,1]
  AUCs[k,] <- round(c(auc(factor(as.matrix(CV.train.resp[[k]])), Train.prob, levels=c("No", "Yes")),
                      auc(factor(as.matrix(CV.valid.resp[[k]])), Valid.prob, levels=c("No", "Yes"))), 4)
}
AUCs
summary(Fit00)
Av.AUCs = apply(AUCs, 2, mean)
names(Av.AUCs) = c("Av.Train AUC", "Av.Valid AUC")
Av.AUCs

# NN final fit with Train set
sigmoid <- function(x) 1 / (1 + exp(-x))
set.seed(my.seed)
Fit01 = neuralnet::neuralnet(resp ~ SMOKING + YELLOW_FINGERS + PEER_PRESSURE + CHRONIC.DISEASE +
                               FATIGUE + ALLERGY + COUGHING + SWALLOWING.DIFFICULTY,
                             Train.set,
                             hidden=3,
                             learningrate=5e-3,
                             act.fct=sigmoid,
                             linear.output=FALSE)
summary(Fit01)
Train.prob = predict(Fit01, newdata=Train.set)[,1]
Test.prob  = predict(Fit01, newdata=Test.set)[,1]

plot(Fit01)

# Accuracy for given threshold value
threshold = 0.1

Train.pred = ifelse(Train.prob > threshold, "Yes", "No")
Test.pred  = ifelse(Test.prob  > threshold, "Yes", "No")
CM.train <- confusionMatrix(factor(Train.pred), factor(as.matrix(Train.resp)), positive="Yes")
CM.test  <- confusionMatrix(factor(Test.pred),  factor(as.matrix(Test.resp)),  positive="Yes")

CM.train 
CM.train$table 

CM.train[["byClass"]][["Sensitivity"]]
CM.train[["byClass"]][["Specificity"]]

CM.test 
CM.test$table

colSums(CM.test$table) / sum(colSums(CM.test$table))    # % of Actual Yes/No
rowSums(CM.test$table) / sum(rowSums(CM.test$table))    # % of predicted Yes/No

# ROC curve and AUC for all threshold
plot.roc(factor(as.matrix(Train.resp)),  Train.prob, levels=c("No", "Yes"))
abline(h=CM.train[["byClass"]][["Sensitivity"]], v=CM.train[["byClass"]][["Specificity"]], col="red")
auc.train = auc(factor(as.matrix(Train.resp)), Train.prob, levels=c("No", "Yes"))
text(.2, .2, paste("Train AUC=",round(auc.train, 3)))

# Test Set
plot.roc(factor(as.matrix(Test.resp)),  Test.prob, levels=c("No", "Yes"))
abline(h=CM.test[["byClass"]][["Sensitivity"]], v=CM.test[["byClass"]][["Specificity"]], col="red")
auc.test = auc(factor(as.matrix(Test.resp)), Test.prob, levels=c("No", "Yes"))
text(.2, .2, paste("Test AUC=",round(auc.test, 3)))

c(auc.train, auc.test)

layout(matrix(1:2, 1, 2))
plot.roc(factor(as.matrix(Train.resp)),  Train.prob, levels=c("No", "Yes"))
text(.2, .2, paste("Train AUC=",round(auc.train, 3)))
plot.roc(factor(as.matrix(Test.resp)),  Test.prob, levels=c("No", "Yes"))
text(.2, .2, paste("Test AUC=",round(auc.test, 3)))
layout(1)

AUCs
Av.AUCs
c(auc.train, auc.test)

# Threshold Picker
cost.list = c(0,0,2,2)/4           # order of (TP, TN, FP, FN)
threshold.list = seq(.01,.99,.01)    # grid for threshold

Chosen.Model <- Fit01
Train.prob = predict(Chosen.Model, newdata=Train.set)[,1]
Test.prob  = predict(Chosen.Model, newdata=Test.set)[,1]

cost=0
for (i in 1:length(threshold.list)){
  threshold = threshold.list[i]
  Test.pred  = ifelse(Test.prob  > threshold, "Yes", "No")
  CM.test  <- confusionMatrix(factor(Test.pred),
                              factor(as.matrix(Test.resp)),
                              positive="Yes")
  TP = CM.test$table[2,2]   # True  Pos
  TN = CM.test$table[1,1]   # True  Neg
  FP = CM.test$table[2,1]   # False Pos
  FN = CM.test$table[1,2]   # False Neg
  cost[i] = sum(c(TP, TN, FP, FN) * cost.list)
}
plot(threshold.list, cost, xlab="threshold")

cost.list
which.min(cost)
min(cost)
threshold.list[which.min(cost)]


