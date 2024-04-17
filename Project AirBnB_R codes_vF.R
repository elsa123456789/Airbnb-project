###### ISOM 5610 - FINAL PROJECT - GROUP #18 [AirBnB]

#### Notes ####

# 1. The R codes and R markdown (together as "R Files") serve as a supplementary materials which provides cross-check and some additional support for our analysis. 

# 2. In specific, the R Files focus on the Section 5.1 in the Python code, which includes "Model 1: Logistic Regression on Feature Data" and "Model 2: Logistic Regression on Amenity Data".

# 3. For the full code, please kindly refer to the Python file. 


#### Prep ####

## Packages
install.packages('bit64')
library(bit64)

install.packages('data.table')
library(data.table)

install.packages('corrplot')
library(corrplot)

install.packages('ggplot2')
library(ggplot2)

##  Set working directory 
setwd("F:/HKUST/ISOM 5610/FINAL_PROJECT")
rm(list = ls())

## Read the table
Data <- fread("cleaned_Airbnb_Data.csv",header = T, sep=",") 
# used fread function from data.table package to accelerate the reading process
head(Data)

Data <-Data [,-1]
head(Data)

## Adjust the existence of specific values (1 if exist, 0 if not)
# Creating categorical variable for host_response_time
Data$response_time_few_hours=ifelse(Data$host_response_time=="within a few hours",1,0)
Data$response_time_an_hour=ifelse(Data$host_response_time=="within an hour",1,0)
Data$response_time_a_day=ifelse(Data$host_response_time=="within a day",1,0)
Data$response_time_a_few_days=ifelse(Data$host_response_time=="a few days or more",1,0)

attach(Data)
head(Data)
summary(Data)

# Setting training data set approx 80% of all data
n <- dim(Data)[1]
train_set = Data[0:4050,]
test_set = Data[4051:n,]


#### 5.Data Modeling ####

## 5.1 Model 1: Logistic Regression on Feature Data  ##

fit=glm('host_is_superhost~amenities_numbers+host_acceptance_rate+host_v_email+neighborhood_overview_exist+host_identity_verified+response_time_a_day+response_time_a_few_days+beds+instant_bookable+host_about_exist',data=train_set,family=binomial)
summary(fit)
# Such regression setting is based on the fitted result in Python code

## Estimated probabilities - initial attempt at cut-off level of 0.5  
prob <- predict(fit,newdata=test_set, type = 'response')

# Confusion Table
table(test_set$host_is_superhost, prob > 0.5)

# Confusion Matrix plot
install.packages('ROCR')
library(ROCR)

install.packages('caret')
library(caret)

install.packages('gridExtra')
library(gridExtra)

source("unbalanced_function.R")

p=predict(fit,type='response')
temp_train=cbind(train_set,p)

p=prob
temp_test=cbind(test_set,p)

cm_info <- ConfusionMatrixInfo( data = temp_test, predict = "p", 
                                actual = "host_is_superhost", cutoff = 0.5 )
cm_info$plot

## ROC and error measure
install.packages('devtools')
library(devtools)
devtools::install_github("selva86/InformationValue")
library(InformationValue) 

# AUC 
plotROC(test_set$host_is_superhost,prob)

# Error measurements
cat(" [1] The sensitivity is",formatC(sensitivity(test_set$host_is_superhost,prob),dig=10),"\n",
    "[2] The specificity is",formatC(specificity(test_set$host_is_superhost,prob),dig=10),"\n",
    "[3] The Miss Classification Rate is",formatC(misClassError(test_set$host_is_superhost,prob),dig=10))

# Confidence Interval for the odds ratio
OR_CI=exp(confint(fit))
cbind(exp(coef(fit)),OR_CI)

# Comment: the Confusion Matrix together with other measurements showed that we should decrease the cutoff in order to increase the sensitivity.


## Determining the optimal cut-off level
# Double density plot
ggplot(temp_train, aes( p, color = as.factor(host_is_superhost) ) ) + 
  geom_density( size = 1 ) +
  ggtitle( "Training Set's estimated probabilities" ) 

# Finding optimal cutoff by considering cost through cross-validation 
cost_fp <- 10
cost_fn <- 30
roc_info <- ROCInfo( data = cm_info$data, predict = "predict", 
                     actual = "actual", cost.fp = cost_fp, cost.fn = cost_fn )
grid.draw(roc_info$plot)

## Updated error measurements using the optimal cutoff
detach('package:caret')

# Updated Confusion Table
table(test_set$host_is_superhost, prob > roc_info$cutoff)

# Updated Confusion Matrix plot
cm_info_new <- ConfusionMatrixInfo( data = temp_test, predict = "p", 
                                    actual = "host_is_superhost", cutoff = roc_info$cutoff)
cm_info_new$plot

# Updated Error measurements
cat(" [1] The sensitivity is",formatC(sensitivity(test_set$host_is_superhost,prob,threshold = roc_info$cutoff),dig=10),"\n",
    "[2] The specificity is",formatC(specificity(test_set$host_is_superhost,prob,threshold = roc_info$cutoff),dig=10),"\n",
    "[3] The Miss Classification Rate is",formatC(misClassError(test_set$host_is_superhost,prob,threshold = roc_info$cutoff),dig=10))



## Final Model on Full Data Set
fit_full=glm('host_is_superhost~amenities_numbers+host_acceptance_rate+host_v_email+neighborhood_overview_exist+host_identity_verified+response_time_a_day+response_time_a_few_days+beds+instant_bookable+host_about_exist',data=Data,family=binomial)
summary(fit_full)


## 5.1 Model 2: Logistic Regression on Amenity data  ##

# Read the table
top_amenities <- fread("Airbnb_Amenities.csv",header = T, sep=",") 
top_amenities<-top_amenities[,-1]

# Setting training data set approx 80% of all data
n <- dim(top_amenities)[1]
train_set_a = top_amenities[0:4050,]
test_set_a = top_amenities[4051:n,]

# Full logistic regression on amenity 
fit_af=glm('host_is_superhost~.',data=train_set_a,family=binomial)
summary(fit_af)

# Variable selection using AIC
install.packages("HH")
library(HH)

null_m=glm('host_is_superhost~1',data=train_set_a,family=binomial)
stepAIC(null_m,scope=list(lower=null_m,upper=fit_af),direction = "forward")

# Based on the AIC-suggested variables, the chosen model is as below:
fit_a2=glm(formula = host_is_superhost ~ Shampoo + Iron + Hot_water_kettle + 
             First_aid_kit + Elevator + Coffee_maker + TV + Cable_TV + 
             Dedicated_workspace + Kitchen + Dryer + Dishes_and_silverware + 
             Hot_water + Extra_pillows_and_blankets + Fire_extinguisher + 
             Cooking_basics + Hair_dryer + Refrigerator + Air_conditioning + 
             Essentials + Hangers + Long_term_stays_allowed + Luggage_dropoff_allowed + 
             Carbon_monoxide_alarm + Lock_on_bedroom_door, family = binomial, 
           data = train_set_a)
summary(fit_a2)

# VIF check
install.packages("car")
library(car) 
vif (fit_a2)

# Comment: Based on the AIC-suggested model, we decided to remove air conditioning to highlight the key amenities to take into consideration.

# Fitted model
fit_a3=glm(formula = host_is_superhost ~ Shampoo + Iron + Hot_water_kettle + 
             First_aid_kit + Elevator + Coffee_maker + TV + Cable_TV + 
             Dedicated_workspace + Kitchen + Dryer + Dishes_and_silverware + 
             Hot_water + Extra_pillows_and_blankets + Fire_extinguisher + 
             Cooking_basics + Hair_dryer + Refrigerator + 
             Essentials + Hangers + Long_term_stays_allowed + Luggage_dropoff_allowed + 
             Carbon_monoxide_alarm + Lock_on_bedroom_door, family = binomial, 
           data = train_set_a)
summary(fit_a3)

## Estimated probabilities - firstly setting the cut-off at 0.5 
prob_a <- predict(fit_a3,newdata=test_set_a, type = 'response')

# Confusion Table
table(test_set_a$host_is_superhost, prob_a > 0.5)

# Confusion Matrix plot
p_a=predict(fit_a3,type='response')
temp_train_a=cbind(train_set_a,p_a)

p_a=prob_a
temp_test_a=cbind(test_set_a,p_a)

cm_info_a <- ConfusionMatrixInfo( data = temp_test_a, predict = "p_a", 
                                  actual = "host_is_superhost", cutoff = 0.5 )
cm_info_a$plot

# AUC 
plotROC(test_set_a$host_is_superhost,prob_a)

# Error measurements
cat(" [1] The sensitivity is",formatC(sensitivity(test_set_a$host_is_superhost,prob_a),dig=10),"\n",
    "[2] The specificity is",formatC(specificity(test_set_a$host_is_superhost,prob_a),dig=10),"\n",
    "[3] The Miss Classification Rate is",formatC(misClassError(test_set_a$host_is_superhost,prob_a),dig=10))

# Confidence Interval for the odds ratio
OR_CI_a=exp(confint(fit_a3))
cbind(exp(coef(fit_a3)),OR_CI_a)

# Comment: the Confusion Matrix together with other measurements showed that we should decrease the cutoff in order to increase the sensitivity.

## Determining the optimal cut-off level
# Double density plot
ggplot(temp_train_a, aes( p_a, color = as.factor(host_is_superhost) ) ) + 
  geom_density( size = 1 ) +
  ggtitle( "Training Set's estimated probabilities" ) 

# Finding optimal cutoff by considering cost through cross-validation 
cost_fp_a <- 10
cost_fn_a <- 30
roc_info_a <- ROCInfo( data = cm_info_a$data, predict = "predict", 
                     actual = "actual", cost.fp = cost_fp_a, cost.fn = cost_fn_a )
grid.draw(roc_info_a$plot)

## Updated error measure(Confusion Table & Matrix plot) using the optimal cutoff
detach('package:caret')

# Updated Confusion Table
table(test_set_a$host_is_superhost, prob_a > roc_info_a$cutoff)

# Updated Confusion Matrix plot
cm_info_a_new <- ConfusionMatrixInfo( data = temp_test_a, predict = "p_a", 
                                      actual = "host_is_superhost", cutoff = roc_info_a$cutoff)
cm_info_a_new$plot

# Updated Error measurements
cat(" [1] The sensitivity is",formatC(sensitivity(test_set_a$host_is_superhost,prob_a,threshold = roc_info_a$cutoff),dig=10),"\n",
    "[2] The specificity is",formatC(specificity(test_set_a$host_is_superhost,prob_a,threshold = roc_info_a$cutoff),dig=10),"\n",
    "[3] The Miss Classification Rate is",formatC(misClassError(test_set_a$host_is_superhost,prob_a,threshold = roc_info_a$cutoff),dig=10))

## Full Model on Amenities Model
fit_a3_full=glm(formula = host_is_superhost ~ Shampoo + Iron + Hot_water_kettle + 
             First_aid_kit + Elevator + Coffee_maker + TV + Cable_TV + 
             Dedicated_workspace + Kitchen + Dryer + Dishes_and_silverware + 
             Hot_water + Extra_pillows_and_blankets + Fire_extinguisher + 
             Cooking_basics + Hair_dryer + Refrigerator + 
             Essentials + Hangers + Long_term_stays_allowed + Luggage_dropoff_allowed + 
             Carbon_monoxide_alarm + Lock_on_bedroom_door, family = binomial, 
           data = top_amenities)

summary(fit_a3_full)