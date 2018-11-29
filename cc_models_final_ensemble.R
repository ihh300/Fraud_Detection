# # cleaning the environment
# rm(list=ls())
# # checking and increasing memory limit
# memory.limit()
# #increasing storage capacity
# memory.limit(size=56000)

library(tidyverse)
# Viz
library(ggplot2)
library(ggthemes)
library(gridExtra)
library(grid)
library(corrplot)
# Sampling
library(unbalanced)
library(DMwR) # for smote implementation
library(ROSE) # for rose implementation
# Models
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(fastAdaboost)
library(xgboost)
library(caret)
library(caretEnsemble)

############################

# Models evaluation
library(precrec)
library(pROC)
library(ROCR)
library(PRROC) # for Precision-Recall curve calculations
library(MLmetrics)

# Output
library(knitr)
library(kableExtra)

# Rest
# library(e1071)
# library(C50)



###################################################################################################
# importing the dataset
df <- read.csv("creditcard.csv")
head(df)

##################################################################################################
#Data preparation
##################################################################################################
# rename
# df$Class[df$Class==1]<- "Fraud"
# df$Class[df$Class==0]<- "Legit"
# head(df)

# Factorise
df  <- mutate(df, Class = factor(Class, levels = c(0,1)))
str(df$Class)
print(table(df$Class))

# Drop variables Time 
df <- select(df, -Time)
summary(df)


##################################################################################################
# Train / Test / Split
##################################################################################################

# Data Partition - Stratified Sampling between Train | Validation and Test hold out dataset

set.seed(1234)
split.test.index <- createDataPartition(df$Class, p = 0.7, list = FALSE, times = 1)
head(split.test.index)
train <- df[split.test.index,]
test <- df[-split.test.index,]

print(table(train$Class))
head(train)

##################################################################################################
# PRE-MODELING DATA PREPARATION
##################################################################################################


# whole scaler to train
scaler_train <- preProcess(train, method = "scale")
train_scaled <- predict(scaler_train,train)
print(table(train_scaled$Class))
head(train_scaled)

# and SEPARATELY whole scaler to test

scaler_test <- preProcess(test, method = "scale")
test_scaled <- predict(scaler_test, test)
print(table(test_scaled$Class))

# Or Normalization Amount into a new column

# train_validation$AmountNo <- scale(train_validation$Amount, center = TRUE, scale = TRUE)
# head(train_validation)

##################################################################################################
# Modeling
##################################################################################################

###########################################
## Convention
## Here fraud is a positive class and legitimate is negative class
## Fraud = 1 | Legitimate = 0
############################################


############################################
# if Re-sampling
############################################

# Synthetic data generation with ROSE
# set.seed(1234)
# train_scaled_rose <- ROSE(Class ~ ., data = train_scaled, seed = 1)$data
# table(train_scaled_rose$Class)

# # Synthetic data generation with SMOTE
# set.seed(1234)
# train_scaled_smote <- SMOTE(Class ~ ., data  = train_scaled)
# table(train_scaled_smote$Class)

# Both sampling
# set.seed(1234)
# train_scaled_both <- ovun.sample(Class ~ ., data = train_scaled, method = "both", p=0.5,N=10000, seed = 1)$data
# table(train_scaled_both$Class)
# head(train_scaled_both)

###########################################
### injection of trainset #################
###########################################

# no resampling
trainset <-train_scaled 

############################################
# Logistic Regression with GLM
############################################

# Model
glm.both <- glm(data = trainset, family = "binomial", formula = Class ~ .)
# glm summary
summary(glm.both)
anova(glm.both)
# Prediction
glm.pred.both <- predict(glm.both,newdata = test_scaled, type = "response")
glm.pred.obj.both <- mmdata(glm.pred.both,test$Class)
glm.perf.both<- evalmod(mdat = glm.pred.obj.both) 
glm.perf.both

# Confusion matrix
glm.thr.both <- ifelse(glm.pred.both>0.5,1,0)
cm_glm <- caret::confusionMatrix(data = as.factor(glm.thr.both), reference = as.factor(test$Class), positive ="1")
print(cm_glm)
cm_pr_glm <- caret::confusionMatrix(data = as.factor(glm.thr.both), reference = as.factor(test$Class), positive ="1",mode = "prec_recall")
print(cm_pr_glm)

#Plot
plot.roc(test$Class, glm.pred.both, print.auc=TRUE)


############################################
# rpart
############################################

# Model
rpart.both <- rpart(Class~., data = trainset)
rpart.pred.both <- predict(rpart.both, newdata = test_scaled,type="class")
rpart.pred.obj.both <- mmdata(as.numeric(rpart.pred.both),test$Class)
rpart.perf.both <- evalmod(mdat = rpart.pred.obj.both) 
rpart.perf.both

# decision tree
rattle::fancyRpartPlot(rpart.both, sub="Classification of fraudulent transactions", palettes=c("Greys", "Oranges"))

# Confusion matrix
cm_rpart <- caret::confusionMatrix(data = as.factor(rpart.pred.both), reference = as.factor(test$Class), positive ="1")
print(cm_rpart)
cm_pr_rpart <- caret::confusionMatrix(data = as.factor(rpart.pred.both), reference = as.factor(test$Class), positive ="1",mode = "prec_recall")
print(cm_pr_rpart)


#Plot
tree.pred.prob <- predict(rpart.both, test[, colnames(test) != "Class"], type= "prob")
plot.roc(test$Class, tree.pred.prob[,2], print.auc=TRUE)


########################################################################################################################
############# ENSEMBLE #############################################
########################################################################################################################



########################################################################################################################
### Bagging and Random Forests
########################################################################################################################

set.seed(1234)
rf.both <- randomForest(formula = Class~., data = trainset,ntree=72,replace=T,importance=TRUE)

# error rate of random forest
plot(rf.both, main = "Error rate of random forest")

#### Importance of variables
# table
df_var_importance<- data.frame(importance(rf.both))
df_var_importance %>% select(MeanDecreaseAccuracy:MeanDecreaseGini) %>% kable() %>% kable_styling(bootstrap_options = c("striped", "hover", "condensed"))

## Standard plot
varImpPlot(rf.both, main = "Variable Importance Metrics")
#varImpPlot(rf.both,sort=TRUE, n.var=min(20, nrow(rf.both$importance)), main = "Importance of Variables")
#df_var_importance <- data.frame(varImpPlot(rf.both))



################# Other plot

varimp <- data.frame(rf.both$importance)

plot.gini <- ggplot(varimp, aes(x=reorder(rownames(varimp),MeanDecreaseGini), y=MeanDecreaseGini)) +
  geom_bar(stat="identity", fill="purple", colour="lightgrey") +
  coord_flip() + theme_calc(base_size = 10) +
  labs(title="Prediction using RandomForest", subtitle="Variable importance (MeanDecreaseGini)", x="Variable", y="Variable importance (MeanDecreaseGini)")

plot.acc <- ggplot(varimp, aes(x=reorder(rownames(varimp),MeanDecreaseAccuracy), y=MeanDecreaseAccuracy)) +
  geom_bar(stat="identity", fill="lightgreen", colour="lightgrey") +
  coord_flip() + theme_calc(base_size = 10) +
  labs(title="Prediction using RandomForest", subtitle="Variable importance (MeanDecreaseAccuracy)", x="Variable", y="Variable importance (MeanDecreaseAccuracy)")

grid.arrange(plot.gini, plot.acc, ncol=2)

# Prediction
rf.pred.both <- predict(rf.both, newdata = test_scaled,type="class")
rf.pred.obj.both <- mmdata(as.numeric(rf.pred.both),test$Class)
rf.perf.both <- evalmod(mdat = rf.pred.obj.both) 
rf.perf.both

# Confusion matrix
cm_rf <- caret::confusionMatrix(data = as.factor(rf.pred.both), reference = as.factor(test$Class), positive ="1")
print(cm_rf)
cm_pr_rf <- caret::confusionMatrix(data = as.factor(rf.pred.both), reference = as.factor(test$Class), positive ="1",mode = "prec_recall")
print(cm_pr_rf)


#####################################################################################################################
### Boosting - AdaBoost 
#####################################################################################################################


ada.both <- adaboost(Class~., trainset, nIter=5)

# prediction
ada.pred.both <- predict(ada.both, newdata = test_scaled)
ada.pred.obj.both <- mmdata(ada.pred.both$prob[,2],test$Class)
ada.perf.both <- evalmod(ada.pred.obj.both)
ada.perf.both

# Confusion matrix
cm_ada <- confusionMatrix(test$Class, ada.pred.both$class,positive ="1")
print(cm_ada)
cm_pr_ada <- confusionMatrix(test$Class, ada.pred.both$class,positive ="1",mode = "prec_recall")
print(cm_pr_ada)

# Plot
plot.roc(test$Class, ada.pred.both$prob[,2], print.auc=TRUE)

########################################################################################################################
# Boosting -  XGBOOST
########################################################################################################################

library(xgboost)

## Pre-XGBoost

xgtrain <- xgb.DMatrix(data = as.matrix(trainset[-30]), label = as.numeric(as.character(trainset$Class)))
xgtest_scale <- xgb.DMatrix(data = as.matrix(test_scaled[-30]), label = as.numeric(as.character(test_scaled$Class)))

##################

## Initial XGBoost Model
xgb_init <- xgboost(data = xgtrain, nrounds = 100, gamma = 0.1, max_depth = 10, objective = "binary:logistic", nthread = 7)

## Features importance

#### Importance of variables
import.xg <- xgb.importance(colnames(xgtrain), model = xgb_init)

# table
df_var_importance.xg <- data.frame(import.xg)
df_var_importance.xg  %>% kable() %>% kable_styling(bootstrap_options = c("striped", "hover", "condensed"))

# plot
xgb.ggplot.importance(import.xg) + 
  theme_bw(base_size = 15) +
  labs(title="Prediction using XGBoost", x="Variable", y="Variable importance")


xg.pred.both.init <- predict(xgb_init,newdata = xgtest_scale)
xg.pred.obj.both.init <- mmdata(xg.pred.both.init,test$Class)
xg.perf.both.init <- evalmod(xg.pred.obj.both.init)
xg.perf.both.init

# Confusion matrix on initial Xgboost
xg.thr.init <- ifelse(xg.pred.both.init>0.5,1,0)
cm_xg.init <- caret::confusionMatrix(data = as.factor(xg.thr.init), reference = as.factor(test$Class), positive ="1")
print(cm_xg.init)
cm_pr_xg.init <- caret::confusionMatrix(data = as.factor(xg.thr.init), reference = as.factor(test$Class), positive ="1",mode = "prec_recall")
print(cm_pr_xg.init)


##################################################################################################
# Comparing the Models
##################################################################################################


### Confusion matrix #####################################

########Fourplot visualisation ########################

fourfoldplot(cm_pr_glm$table, color = c("#CC6666", "#99CC99"), main = "Logistic regression - Confusion Matrix")
fourfoldplot(cm_pr_rpart$table, color = c("#CC6666", "#99CC99"), main = "Regression tree - Oversampled - Confusion Matrix")
fourfoldplot(cm_pr_rf$table, color = c("#CC6666", "#99CC99"), main = "Random Forest- Confusion Matrix")
fourfoldplot(cm_pr_ada$table, color = c("#CC6666", "#99CC99"), main = "Adaboost - Confusion Matrix")
fourfoldplot(cm_pr_xg.init$table, color = c("#CC6666", "#99CC99"), main = "XGboost - Confusion Matrix")


########Rectangle visualisation ########################

# drawing confusion matrix function
draw_confusion_matrix <- function(cm) {
  cm_title <- substring(deparse(substitute(cm)), regexpr("_", deparse(substitute(cm))) + 4)
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title(paste("Confusion Matrix for", cm_title,"dataset"), cex.main=2)
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Class1', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Class2', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Class1', cex=1.2, srt=90)
  text(140, 335, 'Class2', cex=1.2, srt=90)
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}  



draw_confusion_matrix(cm_pr_glm)
draw_confusion_matrix(cm_pr_rpart)
draw_confusion_matrix(cm_pr_rf)
draw_confusion_matrix(cm_pr_ada)
draw_confusion_matrix(cm_pr_xg.init)


########################################################################################################################
############    Comparing the different models - with sensitivity, specificity, precision, recall, F1   #################
#########################################################################################################################


models <- list(glm = glm.both,
               rpart = rpart.both,
  rf =rf.both,
  ada = ada.both,
  xg.init = xgb_init)

name_model <- names(models)
name_model <- as.vector(name_model)


# create a blank data frame
comparison <- data.frame(model = names(models),
                         Specificity = rep(NA, length(models)),
                         Precision = rep(NA, length(models)),
                         Recall = rep(NA, length(models)),
                         F1 = rep(NA, length(models)))

for (name in names(models)) {
  model <- get(paste0("cm_pr_", name))
  num_row <- which(name_model == name) 
  comparison[num_row,"Specificity"] <- round(100*model$byClass["Specificity"],2)
  comparison[num_row,"Precision"] <- round(100*model$byClass["Precision"],2)
  comparison[num_row,"Recall"] <- round(100*model$byClass["Recall"],2)
  comparison[num_row,"F1"] <- round(model$byClass["F1"],3)
}

########### Output the models comparison in a nice TABLE ############################

comparison %>% select(model,Specificity:F1) %>% kable() %>% kable_styling()

########### PLOT the comparison #####################################################

comparison %>% gather(x, y, Specificity:F1) %>%
  ggplot(aes(x = x, y = y, color = model)) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 3)


#########################################################################################################################
############    Comparing the different models - Matthews correlation coefficient                       #################
#########################################################################################################################

# tn=as.double(conf_matrix_logistic[1,1]);
# fp=as.double(conf_matrix_logistic[1,2]);
# fn=as.double(conf_matrix_logistic[2,1]);
# tp=as.double(conf_matrix_logistic[2,2]);
# MCC_logistic = (tp*tn - fp*fn) / sqrt( (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))

#######################################################################################################################################


Matt_Coef <- function (conf_matrix)
{
  TP <- conf_matrix$table[1,1]
  TN <- conf_matrix$table[2,2]
  FP <- conf_matrix$table[1,2]
  FN <- conf_matrix$table[2,1]
  
  mcc_num <- (TP*TN - FP*FN)
  mcc_den <- as.double((TP+FP))*as.double((TP+FN))*as.double((TN+FP))*as.double((TN+FN))
  
  mcc_final <- mcc_num/sqrt(mcc_den)
  return(mcc_final)
}



Matt.glm <- Matt_Coef(cm_pr_glm)
Matt.rpart <- Matt_Coef(cm_pr_rpart)
Matt.rf <- Matt_Coef(cm_pr_rf)
Matt.ada <- Matt_Coef(cm_pr_ada)
Matt.xgboost <- Matt_Coef(cm_pr_xg.init)

# building a dataframe to compare models and scores

model_list <- c('logistic regression','decision tree','random forest','adaboost', 'xgboost')


Matt.corr.coeff <- c(Matt.glm, Matt.rpart, Matt.rf, Matt.ada, Matt.xgboost)

Matthews_Correlation_Coefficient <- data.frame(model_list, Matt.corr.coeff)

Matthews_Correlation_Coefficient %>% kable() %>% kable_styling()

# plot models against Matthews Correlation Coefficient
options(repr.plot.width=4, repr.plot.height=3)
plot.matt.coeff.compa <- ggplot(Matthews_Correlation_Coefficient, aes(x= Matt.corr.coeff, y= model_list, color = model_list)) + 
  geom_point(alpha = 0.5, size = 5) + 
  labs(x = "Matthews correlation coefficient", y = "Model name", title = "Matthews correlation coefficient")

plot.matt.coeff.compa

#######################################


#########################################################################################################################
################## Curves - ROC, AUC and  Area Under the Precision-Recall Curve (AUPRC) #################################
#########################################################################################################################

################################
#### AUC and ROCR Curve
################################


# glm
ROCR.pred.glm <- ROCR::prediction(as.numeric(glm.pred.both), test$Class)
ROCR.perf.glm <- ROCR::performance(ROCR.pred.glm, 'tpr','fpr')
#rpart
ROCR.pred.rpart <- ROCR::prediction(as.numeric(rpart.pred.both), test$Class)
ROCR.perf.rpart <- ROCR::performance(ROCR.pred.rpart, 'tpr','fpr')

#rf
ROCR.pred.rf <- ROCR::prediction(as.numeric(rf.pred.both), test$Class)
ROCR.perf.rf <- ROCR::performance(ROCR.pred.rf, 'tpr','fpr')

#adaboost
ROCR.pred.ada <- ROCR::prediction(as.numeric(ada.pred.both$prob[,2]), test$Class)
ROCR.perf.ada <- ROCR::performance(ROCR.pred.ada, 'tpr','fpr')

#xgboost
ROCR.pred.xg <- ROCR::prediction(as.numeric(xg.pred.both.init), test$Class)
ROCR.perf.xg <- ROCR::performance(ROCR.pred.xg, 'tpr','fpr')


################################
#### AUC
################################

AUC.glm <- as.numeric(ROCR::performance(ROCR.pred.glm, "auc")@y.values)
AUC.rpart <- as.numeric(ROCR::performance(ROCR.pred.rpart, "auc")@y.values)
AUC.rf <- as.numeric(ROCR::performance(ROCR.pred.rf, "auc")@y.values)
AUC.ada <- as.numeric(ROCR::performance(ROCR.pred.ada, "auc")@y.values)
AUC.xg <-as.numeric(ROCR::performance(ROCR.pred.xg, "auc")@y.values)

# building a dataframe to compare models and AUC

model_list <- c('logistic regression','decision tree','random forest','adaboost', 'xgboost')

AUC_Score <- c(AUC.glm,  AUC.rpart, AUC.rf, AUC.ada, AUC.xg)

# Comparison table generation
AUC.Comparison <- data.frame(model_list, AUC_Score)
AUC.Comparison %>% transmute('Model name' = model_list, 'AUC value' = round(AUC.Comparison$AUC_Score,2)) %>% kable() %>% kable_styling()

# plot models against AUC
options(repr.plot.width=4, repr.plot.height=3)
plot.mod.compa.AUC <- ggplot(AUC.Comparison , aes(AUC_Score, model_list, color = model_list)) + 
  geom_point(alpha = 0.5, size = 5) + 
  labs(x = "AUC", y = "Model name", title = "Comparison of models based on AUC")

plot.mod.compa.AUC

##########################################################
####### ROC curves take 2 - Sensitivity=f(specificity)
##########################################################

modelroc.glm <- roc(test$Class,as.numeric(glm.pred.both))
modelroc.rpart <- roc(test$Class,as.numeric(rpart.pred.both))
modelroc.rf <- roc(test$Class,as.numeric(rf.pred.both))
modelroc.ada <- roc(test$Class,as.numeric(ada.pred.both$prob[,2]))
modelroc.xg <- roc(test$Class,as.numeric(xg.pred.both.init))

#########################################################
##### All the curves in one
##########################################################


roc.list <- list(glm=modelroc.glm, rpart=modelroc.rpart, rf=modelroc.rf, ada=modelroc.ada, xgboost=modelroc.xg)

globalroc <- ggroc(roc.list)

#all curves
globalroc + theme_igray() + ggtitle("Comparison of the ROC curves")
#Curves separate
globalroc + facet_wrap( .~name, ncol = 2) + theme_igray()


##########################################################
#### Area Under the Precision-Recall Curve (AUPRC)
##########################################################


######## Area Under the Precision-Recall Curve (AUPRC)

PRRC.perf.glm <- ROCR::performance(ROCR.pred.glm, "prec", "rec")
PRRC.perf.rpart <- ROCR::performance(ROCR.pred.rpart,  "prec", "rec")
PRRC.perf.rf <- ROCR::performance(ROCR.pred.rf, "prec", "rec")
PRRC.perf.ada <- ROCR::performance(ROCR.pred.ada, "prec", "rec")
PRRC.perf.xg <- ROCR::performance(ROCR.pred.xg, "prec", "rec")

plot(PRRC.perf.glm, main="Precision-Recall Curve - Logistic regression", colorize = TRUE, text.adj = c(-0.2,1.7))
plot(PRRC.perf.rpart, main="Precision-Recall Curve - Regression tree", colorize = TRUE, text.adj = c(-0.2,1.7))
plot(PRRC.perf.rf, main="Precision-Recall Curve - Random forest", colorize = TRUE, text.adj = c(-0.2,1.7))
plot(PRRC.perf.ada, main="Precision-Recall Curve - AdaBoost", colorize = TRUE, text.adj = c(-0.2,1.7))
plot(PRRC.perf.xg, main="Precision-Recall Curve - Xgboost", colorize = TRUE, text.adj = c(-0.2,1.7))


# sessionInfo()


