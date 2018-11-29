####################################################################################################################################
####################################################################################################################################

library(tidyverse)
library(ggplot2)
library(ggthemes)
library(gridExtra)
library(grid)
library(Rtsne)
library(caret)
library(ROSE)

# importing the dataset
df <- read.csv("creditcard.csv")

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
# test <- df[-split.test.index,]

print(table(train$Class))
head(train)

##################################################################################################
# PRE-MODELING DATA PREPARATION
##################################################################################################


# whole scaler to train
scaler_train <- preProcess(train, method = "scale")
train_scaled <- predict(scaler_train,train)
print(table(train_scaled$Class))

# # and SEPARATELY whole scaler to test
# 
# scaler_test <- preProcess(test, method = "scale")
# test_scaled <- predict(scaler_test, test)
# print(table(test_scaled$Class))

############################################
# t-SNE on imbalanced data
############################################

#### Sampling the Imbalance Data
df.imbalance.sample <- sample_n(train_scaled, 25000)
print(table(df.imbalance.sample$Class))  

#### Executing the algorithm on balanced data 
tsne_subset <- select(df.imbalance.sample, -Class)
tsne <- Rtsne(tsne_subset, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500,check_duplicates = F)

#### Plotting the t-SNE
classes.imb <- as.factor(df.imbalance.sample$Class)
df_tsne <- as.data.frame(tsne$Y)
head(df_tsne)

plot_tsne <- ggplot(df_tsne, aes(x = V1, y = V2)) + 
  geom_point(aes(color = classes.imb)) + 
  ggtitle("t-SNE visualisation of transactions - Imbalanced data") + 
  scale_color_manual(values = c("#0A0AFF","#AF0000"))+ 
  theme_bw()

plot_tsne


############################################
# t-SNE on undersampled data
############################################

#### Undersampling
set.seed(1234)
train_scaled_under <- ovun.sample(Class ~ ., data = train_scaled, method = "under", N = 690, seed = 1)$data
table(train_scaled_under$Class)

#### Executing the algorithm on imbalanced data 
tsne_subset_under <- select(train_scaled_under, -Class)
tsne_under <- Rtsne(tsne_subset_under, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500,check_duplicates = F)

#### Plotting the t-SNE
classes <- as.factor(train_scaled_under$Class)
df_tsne_under <- as.data.frame(tsne_under$Y)
head(df_tsne_under)

plot_tsne_under <- ggplot(df_tsne_under, aes(x = V1, y = V2)) + 
  geom_point(aes(color = classes)) + 
  ggtitle("t-SNE visualisation of transactions - undersampled data") + 
  scale_color_manual(values = c("#0A0AFF","#AF0000"))+ 
  theme_bw()

plot_tsne_under

# ## Alternatively with ggplot2 
# tsne_under_plot <- data.frame(x = tsne_under$Y[,1], y = tsne_under$Y[,2], col = train_scaled_under$Class) 
# ggplot(tsne_under_plot) + geom_point(aes(x=x, y=y, color=col))


