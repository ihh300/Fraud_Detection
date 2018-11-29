# # cleaning the environment
# rm(list=ls())
# # checking and increasing memory limit
# memory.limit()
# #increasing storage capacity
# memory.limit(size=56000)



library(tidyverse)
library(reshape2)
library(plotly)
library(ggplot2)
library(ggthemes)
library(RColorBrewer)
library(skimr)
library(DataExplorer)
library(mice)
library(gridExtra)
library(grid)
library(lubridate)
library(ggridges)
library(corrplot)

###########################################################################################################
#################################### Exploratory Data Analysis ############################################
###########################################################################################################

# importing the dataset
df <- read.csv("creditcard.csv")
# alternative load if too slow
# library(data.table)
# df <- fread("creditcard.csv")

# Dataset summary

#check # of rows
ncol(df);
nrow(df);

# using summary
head(df)
summary(df)
str(df)

# using skim
skim(df)
# sk_df <- skim(df)
# head(sk_df)

#checking the dimension of the input datasets
plot_str(df)

#checking missing data in the datasets
sum(is.na(df))

#######################################################################################
######################    UNIVARIATE EXPLORATION      #################################
#######################################################################################

###############################################################
######################    DATA EXPLORATION: CATEGORICAL     ####
################################################################

#############Exploring the target/output Class


# Check distribution of target/output Class
table(df$Class)
summary(df$Class)
df$Class <- as.factor(df$Class)
prop_class <- prop.table(table(df$Class))*100

# Barchart for target Class
df %>% ggplot(aes(x = Class)) +
  geom_bar(position ="dodge", color = "lightgrey", fill = "#E69F00", width = 0.7) +
  scale_y_continuous()+
  scale_x_discrete()+
  ggtitle("Distribution of the labelled output - Class")+
  theme_bw()


################################################################
######################    DATA EXPLORATION: TIME     ###########
################################################################

# Exploring the Time variable
summary(df$Time)

## Conversion from time in seconds to days, hours, min,

### Transactions per day
df$byday <- ifelse(df$Time > 3600 * 24, "day2", "day1")

df %>% ggplot(aes(x = byday)) +
  geom_bar(color = "grey", fill = "#003366") +
  ggtitle("Transactions per day") +
  theme_bw()

### Transactions per hours

df$byhour <- df$Time/3600

plotbyhour1 <- ggplot(df,aes(x=byhour,fill=Class,color=Class))+
  geom_bar(color = "#0072B2", fill = "#0072B2") +
  geom_vline(aes(xintercept=mean(byhour, na.rm=T)),color="red", linetype="dashed", size=1)+
  ggtitle("Transactions per hours") +
  theme_bw()

#alternatively 
plotbyhour2 <- ggplot(df,aes(x=byhour,fill=Class,color=Class))+
  geom_histogram(aes(y=..density..),binwidth=1, colour="black", fill="white")+
  geom_density(alpha =0.2,adjust=0.25,fill="#0072B2")+
  geom_vline(aes(xintercept=mean(byhour, na.rm=T)),color="red", linetype="dashed", size=1)+
  ggtitle("Transactions per hours - Density curve") +
  theme_bw()

library(gridExtra)
library(grid)

grid.arrange(plotbyhour1, plotbyhour2, ncol = 1)


### Removing added variables
df <- select(df, -byday, -byhour)

head(df)

################################################################
######################    DATA EXPLORATION: NUMERICAL     ######
################################################################


# Exploring the Amount variable
ggplot(df,aes(x=log(Amount)))+
  geom_histogram(aes(y=..density..),binwidth=.25,
                 colour="black", fill="white")+
  geom_density(alpha =0.2,adjust=0.25,fill="#0072B2")+
  ggtitle("Amount distribution - Density curve")


# Exploring the Vi variables - BOX-plots

boxVi <- boxplot(df[, -c(1, 30, 31)])
boxVi


# Exploring the Vi variables - OVERVIEW with boxplots
df.vi <- select(df, -Time, -Amount, -Class)
meltData_box <- melt(df.vi[1:28])
ggplot(meltData_box, aes(factor(variable), value))+ 
  geom_boxplot(fill="darkred") + facet_wrap(~variable, scale="free")

# Exploring the Vi variables - OVERVIEW with histograms
ggplot(data = meltData_box, mapping = aes(x = value)) + geom_histogram(bins = 10, fill="darkred") + facet_wrap(~variable, scales = 'free_x')



#######################################################################################
#############   BIVARIATE RELATIONSHIPS   #############################################
#######################################################################################


df$Class <- as.factor(df$Class)
######### Time vs Class
ggplot(df, aes(x =Time,fill = Class))+ 
  geom_histogram(bins = 20) +
  facet_wrap( ~ Class, scales = "free", ncol = 2)+
  scale_fill_manual(values=c("darkgreen", "red"))+
  ggtitle("Transactions over time - splited by Class")+
  theme_bw()


######## Amount vs class
df_summarised <- df %>% group_by(Class) %>% summarise(amount_mean = mean(Amount), amount_median = median(Amount)) 
df_summarised <- melt(df_summarised,id.vars ="Class", measure.vars = c("amount_mean","amount_median"))

#Density curve Amount vs Class
ggplot(df, aes(Amount, fill = Class) ) + 
  geom_density(alpha = 0.4,  col = "black") +
  geom_vline(data = df_summarised, aes(linetype = variable, xintercept=value, color = Class), size=1.2, show.legend = TRUE) +
  scale_fill_manual(values=c("darkgreen", "red"),labels = c("Legit", "Fraud"))+
  scale_linetype_discrete(labels = c(amount_mean = "mean", amount_median = "median") ) +
  scale_color_discrete(breaks = NULL) +
  xlim(0,500) +
  labs(linetype = "Stats",
  title = "Density distribution of Legitimate and Fraudulent transaction - Amount", caption = "*x axis limited at 500 for better visualization") +
  theme_bw()

#Boxplot Amount vs Class
ggplot(df,aes(x=Class, y=Amount,group=Class,fill=Class))+
  geom_boxplot()+
  scale_fill_manual(values=c("darkgreen", "red"))+
  scale_y_continuous()+
  scale_x_discrete()+
  ylim(0, max(df$Amount[df$Class==1]))+
  ggtitle("Boxplot comparison - Amount - for legitimate and fraudulent class")+
  theme_bw()


######## Vi variables in regards to output Class
df_reshaped <- select(df, -Amount, -Time)
df_reshaped <- melt(df_reshaped,id.vars ="Class")

df_vi_summary <- df_reshaped %>% group_by(Class,variable) %>% summarise(mean = mean(value), median = median(value))
df_vi_summary <- melt(df_vi_summary,id.vars = c("Class","variable"), measure.vars = c("mean","median"))
colnames(df_vi_summary) <- c("Class","variable","Stats","Value")
head(df_vi_summary)


ggplot(df_reshaped, aes(x = value, fill = Class) ) + 
  geom_density(alpha = 0.4,  col = "black") +
  geom_vline(data = df_vi_summary, aes(colour = Class,linetype = Stats, xintercept=Value), size=1.2, show.legend = TRUE) +
  scale_fill_manual(values=c("darkgreen", "red"),labels = c("Legit", "Fraud"))+
  facet_wrap("variable", ncol = 4, nrow = 7, scales = "free_y") +
  xlim(-5,5) +
  scale_fill_discrete(labels = c("Legit", "Fraud")) +
  scale_color_discrete(breaks = NULL) +
  labs(title = "Density distribution for each Vi variable")+
  theme (axis.title.y = element_blank())+
  theme_bw()

# Ridge - Visualising all the Vi variables vs Class
# percent_rank(): a number between 0 and 1 computed by rescaling min_rank to [0, 1]
# value_percent_rank are the percentile values (percent ranks) in each Class

df_perc_rank <- df_reshaped %>% group_by(variable) %>% mutate(value_percent_rank = percent_rank(value))

df_perc_rank %>% ggplot(aes(y = as.factor(variable), fill = as.factor(Class), x = percent_rank(value))) + 
  geom_density_ridges(scale = 4) + 
  scale_fill_cyclical(values = c("green", "red"), guide = "legend")+
  theme_ridges(center_axis_labels = TRUE)+
  labs(x = "Percentile values (percent ranks) in each Class",
    y = "Variables Vi",
    title = "Fraud versus legitimate for each variable Vi") +
  scale_x_continuous(expand = c(0.01, 0)) +
  scale_y_discrete(expand = c(0.01, 0))


# # Vi mean versus Class
# 
# rownames(df) <- 1:nrow(df)
# legitimate <- df[df$Class == 0,]
# fraud <- df[df$Class == 1,]
# 
# # 492 frauds 
# mean_legitimate <- apply(legitimate[sample(rownames(legitimate), size = 492), -c(1, 30, 31)], 2, mean)
# mean_fraud <- apply(fraud[, -c(1, 30, 31)], 2, mean)
# plot(mean_fraud, col = "red", xlab = "Features", ylab = "Mean")
# lines(mean_fraud, col = "red", lwd = 2)
# points(mean_legitimate, col = "green")
# lines(mean_legitimate, col = "green", lwd = 2)
# legend("topright", legend = c("Legitimate", "Fraud"), lty = c(1,1), col = c("green", "red"), lwd = c(2,2))


##########################################################################################################################
##########################################################################################################################
###################    DATA PREPARATION: OUTLIERS VALUES     #############################################################
##########################################################################################################################
##########################################################################################################################

########################################################
# OUTLIER : ANALYSIS                                  #
########################################################

############## outlier analysis - histograms generation for vi #####################

ws = "/Users/isabelle/Desktop/ihh300/RProjects/creditcardfraud/"
ws_img = "/Users/isabelle/Desktop/ihh300/RProjects/creditcardfraud/eda/outlier/"
setwd(ws)

for (ind in 1:28){
  colnm <- paste("V",ind, sep="")
  flnm <- paste(ws,"hist_",colnm,".jpeg", sep="")
  jpeg(file = flnm)
  labnm <- paste("histogram of feature",colnm, sep=" ")
  hist(df[,ind+1], xlab= colnm, main =labnm, col = 'pink')
  dev.off()
}


##############  outlier analysis - boxplot generation for vi #####################

######### WIP fix with ORCA instead of export

library(plotly)

for (ind in 1:28){
  colnm <- paste("V",ind, sep="")
  flnm <- paste(ws,"box_",colnm,".jpeg", sep="")
  p <- plot_ly(type = 'box', data= df, x = ~Class, y = df[,ind+1],
               marker = list(color = 'rgb(7,40,89)'),
               line = list(color = 'rgb(7,40,89)'))
  export(p, file = flnm)
}

########################################################
# OUTLIER : REMOVAL                                    #
########################################################

df_rm_outliers <- df

remove_outliers <- function(x) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = TRUE)
  H <- 1.5 * IQR(x, na.rm = TRUE)
  y=x
  y[x < (qnt[1] - H)] = -999999
  y[x > (qnt[2] + H)] = -999999
  y
  
}

summary(df_rm_outliers)

# WIP appliquer au bonne - df_rm_outliers$VXX = df_rm_outliers(data_red_rm_outliers$VXX)
# ...
#...

#### df
df_outlier_removed = df_rm_outliers[rowSums(df_rm_outliers ==-999999)==0, , drop = FALSE]

### Checking variables post-OUTLIERS removal using the BOXPLOTS
meltData_box_post <- melt(df_outlier_removed[2:28])
ggplot(meltData_box_post, aes(factor(variable), value))+ 
  geom_boxplot(fill="darkred") + facet_wrap(~variable, scale="free")


##########################################################################################################################
##########################################################################################################################
###################    DATA PREPARATION: MISSING VALUES     ##############################################################
##########################################################################################################################
##########################################################################################################################

#################################################################################
# VISUALIZING MISSING VALUES                                                    #
#################################################################################

library(mice)
# calculation - combination of missing data
md.pattern(df)

##########################################
####### Or Missing data map
##########################################

library(Amelia)
missmap(df)

# alternatively 
# plot_missing(df)
# md.pattern(df)
# apply(apply(df,2,is.na),2,sum) 

##########################################################################################################################
##########################################################################################################################
###################    DATA PREPARATION: DATA TRANSFORMATION     ###############3#########################################
##########################################################################################################################
##########################################################################################################################

# factor
# Time


##########################################################################################################################
##########################################################################################################################
###################    FEATURE SELECTION     ###############3#############################################################
##########################################################################################################################
##########################################################################################################################


#############################################################
# CORRELATION-BASED FEATURE SELECTION - CORRELOGRAM         #
#############################################################


## FIRST METHOD - Exploring the correlations between data

df$Class <- as.numeric(df$Class)
corr_df <- select(df, -Time)
head(corr_df)

# compute correlation coefficients
corredf <- cor(corr_df,method="pearson")

# Visualisation using corrplot
library(corrplot)
# corrplot(corredf, type="upper", order="hclust", col=brewer.pal(n=8, name="PuOr"))
# col1 <- colorRampPalette(brewer.pal(9,"BrBG"))
# corrplot(corredf,method = "square", order = "FPC", tl.col = "black", tl.cex = 0.75, sig.level = 0.05, insig = "pch", pch.cex = 1, col = col1(100))

corrplot(corredf, method="shade", col=col1(20))


# ## SECOND METHOD - Checking colinearity between variables
# 
# df$Class <- as.numeric(df$Class)
# corr_df <- select(df, -Time)
# head(corr_df)
# 
# #### Extract the correlation coefficients
# library(Hmisc)
# cor_result=rcorr(as.matrix(corr_df))
# cor_result$r
# 
# #### FlattenCorrMatrix 
# flattenCorrMatrix <- function(cormat, pmat) {
#   ut <- upper.tri(cormat)
#   data.frame(
#     row = rownames(cormat)[row(cormat)[ut]],
#     column = rownames(cormat)[col(cormat)[ut]],
#     cor  =(cormat)[ut],
#     p = pmat[ut]
#   )
# }
# 
# cor_result_flat = flattenCorrMatrix(cor_result$r, cor_result$P)
# head(cor_result_flat)
# 
# ############ Visualisation of colinearity between variables
# 
# # visualisation with plot of the correlation matrix
# plot(cor_result_flat,cex=0.6)
#  
# # Visualisation using corrplot
# library(corrplot)
# corrplot(cor_result$r, method="color")
# corrplot(cor_result$r, method="circle")
# corrplot(cor_result$r, type = "upper", order = "hclust", col = brewer.pal(n = 8, name = "RdBu"), method="color",tl.col = "black",tl.srt = 45)
# 
# # Visualisation using ggcorrplot
# library(ggcorrplot)
# ggcorrplot(cor_result$r)
# ggcorrplot(cor_result$r, method = "circle")
# ggcorrplot(cor_result$r, hc.order = TRUE, outline.col = "white")
# 
# # Visualisation wih a network graph
# 



#############################################################
# CORRELATION-BASED FEATURE SELECTION - WITH CARET          #
#############################################################


### Using caret to confirm variables to be dropped
library(caret)
caret::findCorrelation(cor_result$r,cutoff = 0.5,names = T, verbose = T)

#############################################################
# Principal Component Analysis (PCA)                       #
#############################################################

# Not relevant as PCA was performed on variables Vi

#############################################################
# t-Distributed Stochastic Neighbor Embedding (t-SNE)       #
#############################################################

# See separate analysis

