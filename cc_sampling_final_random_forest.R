
# # cleaning the environment
# rm(list=ls())
# # checking and increasing memory limit
# memory.limit()
# #increasing storage capacity
# memory.limit(size=56000)

library(tidyverse)
library(reshape2)
# Viz
library(ggplot2)
library(ggthemes)
library(gridExtra)
library(grid)
library(corrplot)
# Dimension reduction
library(Rtsne)
# Sampling
library(unbalanced)
library(DMwR) # for smote implementation
library(ROSE) # for rose implementation
# Models
library(rattle)
library(precrec)
# library(caTools)
library(caret)
# library(ipred)
# library(e1071)
# library(C50)
library(randomForest)
library(MLmetrics)
# Models evaluation
library(pROC)
library(ROCR)
library(PRROC) # for Precision-Recall curve calculations

# Output
library(knitr)
library(kableExtra)

###################################################################################################
# importing the dataset
df <- read.csv("creditcard.csv")
head(df)


##################################################################################################
#Data preparation
##################################################################################################
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

# Whole scaling

# whole scaler to train
scaler_train <- preProcess(train, method = "scale")
train_scaled <- predict(scaler_train,train)
print(table(train_scaled$Class))

# and SEPARATELY whole scaler to test
scaler_test <- preProcess(test, method = "scale")
test_scaled <- predict(scaler_test, test)
print(table(test_scaled$Class))

# Or Normalization Amount into a new column

###############################################################
# Or Normalization Amount into a new column
# 
# normalize <- function(x){
#   return((x - mean(x, na.rm = TRUE))/sd(x, na.rm = TRUE))
# }
# 
# train_scaled <- train
# train_scaled$Amount <- normalize(train_scaled$Amount)
# 
# test_scaled <- test
# test_scaled$Amount <- normalize(test_scaled$Amount)


# First method
# train_validation$AmountNo <- scale(train_validation$Amount, center = TRUE, scale = TRUE)
# head(train_validation)

# p1 <-ggplot(train_validation,aes(x=log(Amount)))+
#   geom_histogram(aes(y=..density..),binwidth=.25,
#                  colour="black", fill="white")+
#   geom_density(alpha =0.2,adjust=0.25,fill="#0072B2")+
#   ggtitle("Amount distribution - Density curve")
# 
# p2 <- ggplot(train_validation,aes(x=log(AmountNo)))+
#   geom_histogram(aes(y=..density..),binwidth=.25,
#                  colour="black", fill="white")+
#   geom_density(alpha =0.2,adjust=0.25,fill="#0072B2")+
#   ggtitle("Amount Normalised distribution - Density curve")
# 
# grid.arrange(p1, p2, ncol = 2)

##################################################################################################
# Modeling
##################################################################################################

###########################################
## Convention
## Here fraud is a positive class and legitimate is negative class
## Fraud = 1 | Legitimate = 0
############################################


############################################
# Modeling the original unbalanced data - Using Regression Trees
############################################

set.seed(1234)
# Model
rf.original <- randomForest(formula = Class~., data = train_scaled,ntree=72,replace=T,importance=TRUE)
####################################################################################
# Prediction
rf.pred.original <- predict(rf.original, newdata = test_scaled,type="class")
rf.pred.obj.original <- mmdata(as.numeric(rf.pred.original),test$Class)
rf.perf.original <- evalmod(mdat = rf.pred.obj.original) 
rf.perf.original

####################################################################################################
# Variables importance results
importance(rf.original)

# error rate of random forest
plot(rf.pred.original, main = "Error rate of random forest")

################## Plot
## Standard plot of Importance of variables
varImpPlot(rf.original,sort=TRUE, n.var=min(20, nrow(rf.original$importance)), main = "Importance of Variables")

################# Other plot

varimp <- data.frame(rf.original$importance)

plot.gini <- ggplot(varimp, aes(x=reorder(rownames(varimp),MeanDecreaseGini), y=MeanDecreaseGini)) +
  geom_bar(stat="identity", fill="purple", colour="lightgrey") +
  coord_flip() + theme_calc(base_size = 10) +
  labs(title="Prediction using RandomForest", subtitle="Variable importance (MeanDecreaseGini)", x="Variable", y="Variable importance (MeanDecreaseGini)")

plot.acc <- ggplot(varimp, aes(x=reorder(rownames(varimp),MeanDecreaseAccuracy), y=MeanDecreaseAccuracy)) +
  geom_bar(stat="identity", fill="lightgreen", colour="lightgrey") +
  coord_flip() + theme_calc(base_size = 10) +
  labs(title="Prediction using RandomForest", subtitle="Variable importance (MeanDecreaseAccuracy)", x="Variable", y="Variable importance (MeanDecreaseAccuracy)")

grid.arrange(plot.gini, plot.acc, ncol=2)


#####################################################################################


# Confusion matrix
cm_original <- caret::confusionMatrix(data = as.factor(rf.pred.original), reference = as.factor(test$Class), positive ="1")
print(cm_original)
cm_pr_original <- caret::confusionMatrix(data = as.factor(rf.pred.original), reference = as.factor(test$Class), positive ="1",mode = "prec_recall")
print(cm_pr_original)

#### Performance of the model trained on unbalanced data

fourfoldplot(cm_pr_original$table, color = c("#CC6666", "#99CC99"), main = "Regression tree - Original - Confusion Matrix")

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


draw_confusion_matrix(cm_pr_original)

######
# original AUC
ROCR.pred.original <- prediction(as.numeric(rf.pred.original), test$Class)
ROCR.perf.original <- performance(ROCR.pred.original, 'tpr','fpr')
# AUC
AUC.original <- as.numeric(performance(ROCR.pred.original, "auc")@y.values)
AUC.original

#ROC
modelroc.original <- roc(test$Class,as.numeric(rf.pred.original))
modelroc.original
ggroc(modelroc.original) + theme_igray() + ggtitle("ROC curve - Decision tree - Imbalance data")

######## Area Under the Precision-Recall Curve (AUPRC)
# Precision-Recall
PRRC.perf.original <- performance(ROCR.pred.original, "prec", "rec")
PRRC.perf.original

autoplot(rf.perf.original, "PRC") + theme_igray() + ggtitle("Precision-Recall Curve - Original")

# F1 score

F1.original <- cm_pr_original$byClass["F1"]
F1.original
# Matthews correlation coefficient 

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

Matt.original <- Matt_Coef(cm_pr_original)

Matt.original




###########################################################################################################################################
############################ Comparing various sampling method producing to balanced training dataset
###########################################################################################################################################


############################################
# Under-sampling + Modeling
############################################
# Undersampling
set.seed(1234)
train_scaled_under <- ovun.sample(Class ~ ., data = train_scaled, method = "under", N = 690, seed = 1)$data
table(train_scaled_under$Class)

# Model
rf.under <- randomForest(formula = Class~., data = train_scaled_under,ntree=72,replace=T,importance=TRUE)
rf.pred.under <- predict(rf.under, newdata = test_scaled,type="class")
rf.pred.obj.under <- mmdata(as.numeric(rf.pred.under),test$Class)
rf.perf.under <- evalmod(mdat = rf.pred.obj.under) 
rf.perf.under

# Confusion matrix
cm_under <- caret::confusionMatrix(data = as.factor(rf.pred.under), reference = as.factor(test$Class), positive ="1")
print(cm_under)
cm_pr_under <- caret::confusionMatrix(data = as.factor(rf.pred.under), reference = as.factor(test$Class), positive ="1",mode = "prec_recall")
print(cm_pr_under)


############################################
# Oversampling + Modeling
############################################

# Oversampling
set.seed(1234)
train_scaled_over <- ovun.sample(Class ~ ., data = train_scaled, method = "over",N = 398042)$data
table(train_scaled_over$Class)

# Model
rf.over <- randomForest(formula = Class~., data = train_scaled_over,ntree=72,replace=T,importance=TRUE)
rf.pred.over <- predict(rf.over, newdata = test_scaled,type="class")
rf.pred.obj.over <- mmdata(as.numeric(rf.pred.over),test$Class)
rf.perf.over <- evalmod(mdat = rf.pred.obj.over) 
rf.perf.over

# Confusion matrix
cm_over <- caret::confusionMatrix(data = as.factor(rf.pred.over), reference = as.factor(test$Class), positive ="1")
print(cm_over)
cm_pr_over <- caret::confusionMatrix(data = as.factor(rf.pred.over), reference = as.factor(test$Class), positive ="1",mode = "prec_recall")
print(cm_pr_over)



############################################
# Both + Modeling
############################################

# Both sampling
set.seed(1234)
train_scaled_both <- ovun.sample(Class ~ ., data = train_scaled, method = "both", p=0.5, seed = 1)$data
table(train_scaled_both$Class)

# Model
rf.both <- randomForest(formula = Class~., data = train_scaled_both,ntree=72,replace=T,importance=TRUE)
rf.pred.both <- predict(rf.both, newdata = test_scaled,type="class")
rf.pred.obj.both <- mmdata(as.numeric(rf.pred.both),test$Class)
rf.perf.both <- evalmod(mdat = rf.pred.obj.both) 
rf.perf.both

# Confusion matrix
cm_both <- caret::confusionMatrix(data = as.factor(rf.pred.both), reference = as.factor(test$Class), positive ="1")
print(cm_both)
cm_pr_both <- caret::confusionMatrix(data = as.factor(rf.pred.both), reference = as.factor(test$Class), positive ="1",mode = "prec_recall")
print(cm_pr_both)


############################################
# ROSE + Modeling
############################################

# Synthetic data generation with ROSE
set.seed(1234)
train_scaled_rose <- ROSE(Class ~ ., data = train_scaled, seed = 1)$data
table(train_scaled_rose$Class)

# Model
rf.rose <- randomForest(formula = Class~., data = train_scaled_rose,ntree=72,replace=T,importance=TRUE)
rf.pred.rose <- predict(rf.rose, newdata = test_scaled,type="class")
rf.pred.obj.rose <- mmdata(as.numeric(rf.pred.rose),test$Class)
rf.perf.rose <- evalmod(mdat = rf.pred.obj.rose) 
rf.perf.rose

# Confusion matrix
cm_rose <- caret::confusionMatrix(data = as.factor(rf.pred.rose), reference = as.factor(test$Class), positive ="1")
print(cm_rose)
cm_pr_rose <- caret::confusionMatrix(data = as.factor(rf.pred.rose), reference = as.factor(test$Class), positive ="1",mode = "prec_recall")
print(cm_pr_rose)

############################################
# SMOTE + Modeling
############################################

# Synthetic data generation with SMOTE
set.seed(1234)
train_scaled_smote <- SMOTE(Class ~ ., data  = train_scaled) 
table(train_scaled_smote$Class)

# Model
rf.smote <- randomForest(formula = Class~., data = train_scaled_smote,ntree=72,replace=T,importance=TRUE)
rf.pred.smote <- predict(rf.smote, newdata = test_scaled,type="class")
rf.pred.obj.smote <- mmdata(as.numeric(rf.pred.smote),test$Class)
rf.perf.smote <- evalmod(mdat = rf.pred.obj.smote) 
rf.perf.smote

# Confusion matrix
cm_smote <- caret::confusionMatrix(data = as.factor(rf.pred.smote), reference = as.factor(test$Class), positive ="1")
print(cm_smote)
cm_pr_smote <- caret::confusionMatrix(data = as.factor(rf.pred.smote), reference = as.factor(test$Class), positive ="1",mode = "prec_recall")
print(cm_pr_smote)



##################################################################################################
# Comparing the Models
##################################################################################################

########################################################
# Comparing the sampling proportion in one table
########################################################

#Original proportion on whole dataset
original_prop <- table(df$Class)/nrow(df) # unbalanced 

#Original proportion on unbalanced dataset
unbalanced_prop <- table(train_scaled$Class)/nrow(train_scaled) # unbalanced 

#Model Performance on sampled data
post_under <- table(train_scaled_under$Class)/nrow(train_scaled_under) # Post-down sampling
post_over <- table(train_scaled_over$Class)/nrow(train_scaled_over) # Post-up sampling
post_both <- table(train_scaled_both$Class)/nrow(train_scaled_both)    # post both
post_rose <- table(train_scaled_rose$Class)/nrow(train_scaled_rose)     # Post-ROSE
post_smote <- table(train_scaled_smote$Class)/nrow(train_scaled_smote)     # Post-SMOTE

sample_list <- c('whole dataset', 'unbalanced dataset','under-sampled','over-sampled','Both Over+Under','after ROSE','after SMOTE')

comp_value <- bind_rows(original_prop,unbalanced_prop,post_under,post_over,post_both,post_rose,post_smote)
Comp_sample <- as.data.frame(cbind(sample_list,comp_value)) 
comp_sample.prop <- Comp_sample %>% transmute('dataset' = sample_list, 'legitimate' = round(Comp_sample$`0`*100,2), 'fraud' = round(Comp_sample$`1`*100,2)) 
comp_sample.prop %>% kable() %>% kable_styling()


#########################################################################################################################
######################################     Confusion matrix - visualisations    #########################################
#########################################################################################################################


########Fourplot visualisation ########################

fourfoldplot(cm_pr_original$table, color = c("#CC6666", "#99CC99"), main = "Regression tree - Original - Confusion Matrix")
fourfoldplot(cm_pr_under$table, color = c("#CC6666", "#99CC99"), main = "Regression tree - Undersampled - Confusion Matrix")
fourfoldplot(cm_pr_over$table, color = c("#CC6666", "#99CC99"), main = "Regression tree - Oversampled - Confusion Matrix")
fourfoldplot(cm_pr_both$table, color = c("#CC6666", "#99CC99"), main = "Regression tree -Both-sampled - Confusion Matrix")
fourfoldplot(cm_pr_smote$table, color = c("#CC6666", "#99CC99"), main = "Regression tree - SMOTE - Confusion Matrix")
fourfoldplot(cm_pr_rose$table, color = c("#CC6666", "#99CC99"), main = "Regression tree - ROSE - Confusion Matrix")


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


draw_confusion_matrix(cm_pr_original)
draw_confusion_matrix(cm_pr_under)
draw_confusion_matrix(cm_pr_over)
draw_confusion_matrix(cm_pr_both)
draw_confusion_matrix(cm_pr_smote)
draw_confusion_matrix(cm_pr_rose)


#########################################################################################################################
############    Comparing the different models - with sensitivity, specificity, precision, recall, F1   #################
#########################################################################################################################


models <- list(original = rf.original,
               under = rf.under,
               over = rf.over,
               both=rf.both,
               smote = rf.smote,
               rose = rf.rose)

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

comparison %>% select(model,Specificity:F1) %>% kable() %>%  kable_styling(bootstrap_options = c("striped", "hover", "condensed"))

########### PLOT the comparison #####################################################

comparison %>% gather(x, y, Specificity:F1) %>%
  ggplot(aes(x = x, y = y, color = model)) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 3)


#########################################################################################################################
#################      Comparing the different models  - with F1 score only     #########################################
#########################################################################################################################

# Giving the various F1 score

print(paste("F1 of regression tree model with unbalanced dataset is:", round(cm_pr_original$byClass["F1"],3)))
print(paste("F1 of regression tree model with under sampled dataset is:", round(cm_pr_under$byClass["F1"],3)))
print(paste("F1 of regression tree model with over sampled dataset is:", round(cm_pr_over$byClass["F1"],3)))
print(paste("F1 of regression tree model with both Over+Under sampled dataset is:", round(cm_pr_both$byClass["F1"],3)))
print(paste("F1 of regression tree model with SMOTE dataset is:", round(cm_pr_smote$byClass["F1"],3)))
print(paste("F1 of regression tree model with ROSE dataset is:", round(cm_pr_rose$byClass["F1"],3)))


comparison %>% select(model,F1) %>% kable() %>% kable_styling()

########### PLOT the comparison #####################################################

comparison %>% gather(x, y, F1) %>%
  ggplot(aes(x = x, y = y, color = model)) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 3)



### Alternatively #################################################################

F1.original <- cm_pr_original$byClass["F1"]
F1.under <- cm_pr_under$byClass["F1"]
F1.over <- cm_pr_over$byClass["F1"]
F1.both <- cm_pr_both$byClass["F1"]
F1.SMOTE <- cm_pr_smote$byClass["F1"]
F1.ROSE <-cm_pr_rose$byClass["F1"]


# building a dataframe to compare models and scores
# with original
model_list <- c('baseline unbalanced dataset', 'under sampled dataset','over sampled dataset','both', 'post-ROSE dataset','post-SMOTE dataset')
F1_Score <- c(F1.original, F1.under, F1.over, F1.both, F1.ROSE, F1.SMOTE)

# table generation
Mod.F1.Comparison <- data.frame(model_list, F1_Score)
Mod.F1.Comparison %>% mutate_if(is.numeric, round, digits=3) %>% kable() %>% kable_styling()

# plot models against F1
options(repr.plot.width=4, repr.plot.height=3)
plot.mod.compa <- ggplot(Mod.F1.Comparison, aes(F1_Score, model_list)) + 
  geom_point() + 
  labs(x = "F1 Score", y = "Model name", title = "Comparison of models based on F1 Score Performance")

plot.mod.compa


#########################################################################################################################
################## Curves - ROC, AUC and  Area Under the Precision-Recall Curve (AUPRC) #################################
#########################################################################################################################

################################
#### AUC and ROCR Curve
################################

# original
ROCR.pred.original <- prediction(as.numeric(rf.pred.original), test$Class)
ROCR.perf.original <- performance(ROCR.pred.original, 'tpr','fpr')
# under
ROCR.pred.under <- prediction(as.numeric(rf.pred.under), test$Class)
ROCR.perf.under <- performance(ROCR.pred.under, 'tpr','fpr')
#over
ROCR.pred.over <- prediction(as.numeric(rf.pred.over), test$Class)
ROCR.perf.over <- performance(ROCR.pred.over, 'tpr','fpr')

#both
ROCR.pred.both <- prediction(as.numeric(rf.pred.both), test$Class)
ROCR.perf.both <- performance(ROCR.pred.both, 'tpr','fpr')

#ROSE
ROCR.pred.rose <- prediction(as.numeric(rf.pred.rose), test$Class)
ROCR.perf.rose <- performance(ROCR.pred.rose, 'tpr','fpr')

#SMOTE
ROCR.pred.smote <- prediction(as.numeric(rf.pred.smote), test$Class)
ROCR.perf.smote <- performance(ROCR.pred.smote, 'tpr','fpr')


################################
#### AUC
################################

AUC.original <- as.numeric(performance(ROCR.pred.original, "auc")@y.values)
AUC.under <- as.numeric(performance(ROCR.pred.under, "auc")@y.values)
AUC.over <- as.numeric(performance(ROCR.pred.over, "auc")@y.values)
AUC.both <- as.numeric(performance(ROCR.pred.both, "auc")@y.values)
AUC.ROSE <-as.numeric(performance(ROCR.pred.rose, "auc")@y.values)
AUC.SMOTE <- as.numeric(performance(ROCR.pred.smote, "auc")@y.values)

# with original
#building a dataframe to compare models and AUC
model_list <- c('baseline unbalanced dataset', 'under sampled dataset','over sampled dataset','both', 'post-ROSE dataset','post-SMOTE dataset')
AUC_Score <- c(AUC.original,AUC.under, AUC.over, AUC.both, AUC.ROSE, AUC.SMOTE)

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

modelroc.original <- roc(test$Class,as.numeric(rf.pred.original))
modelroc.under <- roc(test$Class,as.numeric(rf.pred.under))
modelroc.over <- roc(test$Class,as.numeric(rf.pred.over))
modelroc.both <- roc(test$Class,as.numeric(rf.pred.both))
modelroc.rose <- roc(test$Class,as.numeric(rf.pred.rose))
modelroc.smote <- roc(test$Class,as.numeric(rf.pred.smote))

#########################################################
##### All the curves in one
##########################################################


roc.list <- list(original=modelroc.original, over=modelroc.over, under=modelroc.under, both=modelroc.both, rose=modelroc.rose, smote=modelroc.smote)

#roc.list <- list(over=modelroc.over, under=modelroc.under, both=modelroc.both, rose=modelroc.rose, smote=modelroc.smote)
globalroc <- ggroc(roc.list)

#all curves
globalroc + theme_igray() + ggtitle("Comparison of the ROC curves")
#Curves separate
globalroc + facet_wrap( .~name, ncol = 2) + theme_igray()


##########################################################
#### Area Under the Precision-Recall Curve (AUPRC)
##########################################################


######## Area Under the Precision-Recall Curve (AUPRC)

# PRRC.perf.original <- performance(ROCR.pred.original, "prec", "rec")
# PRRC.perf.under <- performance(ROCR.pred.under,  "prec", "rec")
# PRRC.perf.over <- performance(ROCR.pred.over, "prec", "rec")
# PRRC.perf.both <- performance(ROCR.pred.both, "prec", "rec")
# PRRC.perf.ROSE <- performance(ROCR.pred.rose, "prec", "rec")
# PRRC.perf.SMOTE <- performance(ROCR.pred.smote, "prec", "rec")
# 
# plot(PRRC.perf.original, main="RF-Precision-Recall Curve - Original", colorize = TRUE, text.adj = c(-0.2,1.7))
# plot(PRRC.perf.under, main="RF-Precision-Recall Curve - Undersampled", colorize = TRUE, text.adj = c(-0.2,1.7))
# plot(PRRC.perf.over, main="RF-Precision-Recall Curve - Oversampled", colorize = TRUE, text.adj = c(-0.2,1.7))
# plot(PRRC.perf.both, main="RF-Precision-Recall Curve - Both", colorize = TRUE, text.adj = c(-0.2,1.7))
# plot(PRRC.perf.ROSE, main="RF-Precision-Recall Curve - ROSE", colorize = TRUE, text.adj = c(-0.2,1.7))
# plot(PRRC.perf.SMOTE, main="RF-Precision-Recall Curve - SMOTE", colorize = TRUE, text.adj = c(-0.2,1.7))


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

Matt.original <- Matt_Coef(cm_pr_original)
Matt.under <- Matt_Coef(cm_pr_under)
Matt.over <- Matt_Coef(cm_pr_over)
Matt.both <- Matt_Coef(cm_pr_both)
Matt.rose <- Matt_Coef(cm_pr_rose)
Matt.smote <- Matt_Coef(cm_pr_smote)

# building a dataframe to compare models and scores

model_list <- c('baseline unbalanced dataset', 'under sampled dataset','over sampled dataset','both', 'post-ROSE dataset','post-SMOTE dataset')
Matt.corr.coeff <- c(Matt.original, Matt.under, Matt.over, Matt.both, Matt.rose, Matt.smote)

Matthews_Correlation_Coefficient <- data.frame(model_list, Matt.corr.coeff)

Matthews_Correlation_Coefficient %>% kable() %>% kable_styling()

# plot models against Matthews Correlation Coefficient
options(repr.plot.width=4, repr.plot.height=3)
plot.matt.coeff.compa <- ggplot(Matthews_Correlation_Coefficient, aes(x= Matt.corr.coeff, y= model_list, color = model_list)) + 
  geom_point(alpha = 0.5, size = 5) + 
  labs(x = "Matthews correlation coefficient", y = "Model name", title = "Matthews correlation coefficient")

plot.matt.coeff.compa

#######################################

# sessionInfo()









