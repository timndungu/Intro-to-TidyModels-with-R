#1. Introduction
library(tidymodels) 
library(tidyverse)

# Helper packages
library(readr)       # for importing data
library(vip)         # for variable importance plots

#2.The Hotel Bookings Data
hotels <- 
  read_csv('hotels.csv') %>% mutate_if(is.character, as.factor)

##link to the hotels data
# hotels <- 
# read_csv('https://tidymodels.org/start/case-study/hotels.csv') %>%
#   mutate_if(is.character, as.factor) 

#We will build a model to predict which actual hotel stays included children and/or babies, and which did not. Our outcome variable children is a factor variable with two levels:
hotels %>% 
  count(children) %>% mutate(prop = n/sum(n))

#3. Data Splitting & Resampling
set.seed(123)
splits      <- initial_split(hotels, strata = children)

hotel_other <- training(splits)
hotel_test  <- testing(splits)

# training set proportions by children
hotel_other %>% 
  count(children) %>% mutate(prop = n/sum(n))

# test set proportions by children
hotel_test %>% 
  count(children) %>% mutate(prop = n/sum(n))

#let’s create a single resample called a validation set,treated as a single iteration of resampling
#We’ll use the validation_split() function to allocate 20% of the hotel_other stays to the validation set and 30,000 stays to the training set. 
#This function, like initial_split(), has the same strata argument, which uses stratified sampling to create the resample.
set.seed(234)
validation_set <- 
  validation_split(hotel_other,strata = children, prop = 0.80)

#4.0 Our first model: penalized logistic regression
#START LR ==============================================================
#Setting mixture to a value of one means that the glmnet model will potentially remove irrelevant predictors and choose a simpler model.
lr_model <- 
  logistic_reg(penalty = tune(),mixture = 1) %>% 
  set_engine("glmnet")

#4.1 Create the recipe
holidays <- c("AllSouls", "AshWednesday", "ChristmasEve", "Easter", 
"ChristmasDay", "GoodFriday", "NewYearsDay", "PalmSunday")

lr_recipe <- 
  #outcome variable children & data == hotel_other
  recipe(children ~ ., data = hotel_other) %>% 
  #create predictors for the year, month, and day of the week
  step_date(arrival_date) %>%
  #generate a set of indicator variables for specific holidays
  step_holiday(arrival_date, holidays = holidays) %>%
  #remove the original date variable
  step_rm(arrival_date) %>% 
  # convert characters or factors into one or more numeric binary model terms
  step_dummy(all_nominal(),-all_outcomes()) %>% 
  #remove indicator variables that only contain a single unique value
  step_nzv(all_predictors()) %>% 
  #center and scale numeric variables
  step_normalize(all_predictors())

#4.2 Create the workflow
lr_workflow <- 
  workflow() %>% 
  add_recipe(lr_recipe) %>% 
  add_model(lr_model)

#4.3 Create the grid for tuning
lr_grid <- 
  tibble(penalty = 10^seq(from = -4, to = -1, length.out = 30))

#4.4 Train and tune the model
lr_resamples <- 
  lr_workflow %>% 
  tune_grid(
    #use our validation set
    resamples = validation_set,
    #use our pre defined grid
    grid = lr_grid,
    #save the validation set predictions
    control = control_grid(save_pred = TRUE),
    #area under the ROC curve will be used to quantify how well the model performs across
    metrics = metric_set(roc_auc) 
  )

#plotting the metrics
#This plots shows us that model performance is generally better at the smaller penalty values
lr_resamples %>% 
  collect_metrics() %>% 
  ggplot(aes(x = penalty, y = mean)) +
  geom_point() + 
  geom_line() +
  scale_x_log10(labels = scales::label_number()) +
  ylab("Area under the ROC Curve")

#Our model performance seems to plateau at the smaller penalty values, so going by the roc_auc metric alone could lead us to multiple options for the “best” value for this hyperparameter:
top_lr_models <- 
  lr_resamples %>% 
  show_best(metric = "roc_auc",n = 15) %>% 
  arrange(penalty)

## *Results differ from those on the website - https://www.tidymodels.org/start/case-study/*
#best model 12 with a penalty value of 0.00137 has effectively the same performance as the numerically best model, but might eliminate more predictors.
lr_best <- lr_resamples %>% 
  collect_metrics() %>% 
  arrange(penalty) %>% 
  #per the websites reccomendation
  slice(12)

#4.5 plotting the corresponding 
lr_auc <- 
  lr_resamples %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(children, .pred_children) %>% 
  mutate(model = "Logistic Regression")
  
autoplot(lr_auc)
#END LR================================================================

#5.0 Our second model: tree-based ensemble
#START RF==============================================================
#A random forest is an ensemble model typically made up of thousands of decision trees, where each individual tree sees a slightly different version of the training data and learns a sequence of splitting rules to predict new data. Random forests require very little preprocessing

#5.1 Build the model and improve training time
cores <- parallel::detectCores()

#The mtry hyperparameter sets the number of predictor variables that each node in the decision tree “sees” and can learn about
#The min_n hyperparameter sets the minimum n to split at any node.
rf_model <- 
  rand_forest(mtry = tune(),min_n = tune(), trees = 1000) %>% 
  set_engine("ranger",num.threads = cores) %>% 
  set_mode("classification")


#5.2 Create the recipe and workflow
rf_recipe <- 
  recipe(children ~ ., data = hotel_other) %>% 
  step_date(arrival_date) %>% 
  step_holiday(arrival_date) %>% 
  step_rm(arrival_date)

rf_workflow <- 
  workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(rf_recipe)

#5.3 Train and tune the model
# show what will be tuned
rf_model %>%    
  parameters() 

#We will use a space-filling design to tune, with 25 candidate models:
set.seed(345)
rf_resamples <- 
  rf_workflow %>% 
  tune_grid(
    resamples = validation_set,
    grid = 25,
    control = control_grid(save_pred = TRUE),
    metrics = metric_set(roc_auc)
  )

# top 5 random forest models
rf_resamples %>% 
  show_best(metric = "roc_auc")

#Plotting the results of the tuning process highlights that both mtry (number of predictors at each node) and min_n (minimum number of data points required to keep splitting) should be fairly small to optimize performance.
autoplot(rf_resamples)

#select the best model according to the ROC AUC metric.
rf_best <- 
  rf_resamples %>% 
  select_best(metric = "roc_auc")

#filter the predictions for only our best random forest model
rf_auc <- 
  rf_resamples %>% 
  collect_predictions(parameters = rf_best) %>% 
  roc_curve(children,.pred_children) %>% 
  mutate(model = "Random Forest")

#compare the validation set ROC curves for our top penalized logistic regression model and random forest model:
bind_rows(lr_auc,rf_auc) %>% 
  ggplot(aes(x = 1 - specificity,y = sensitivity,col = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 3) +
  coord_equal() +
  scale_color_viridis_d(option = "plasma", end = .6)

#END RF================================================================

#6 The last fit
#The random forest model clearly performed better and would be our best bet for predicting hotel stays with and without children
#We’ll start by building our parsnip model object again from scratch
last_rf_model <- 
  rand_forest(mtry = 3, min_n = 3, trees = 1000) %>% 
  set_engine("ranger", num.threads = cores, importance = "impurity") %>% 
  set_mode("classification")

#last workflow
last_rf_workflow <- 
  rf_workflow %>% 
  update_model(last_rf_model)

# the last fit
set.seed(345)
last_rf_fit <- 
  last_rf_workflow %>% 
  last_fit(splits)

#Was the validation set a good estimate of future performance?
last_rf_fit %>% 
  collect_metrics()

#We can access those variable importance scores first pluck out the first element in the workflow column, then pull out the fit from the workflow object 
last_rf_fit %>% 
  pluck(".workflow",1) %>% 
  pull_workflow_fit() %>% 
  vip(num_features = 20)

#generate our last ROC curve to visualize 
#Based on these results, the validation set and test set performance statistics are very close, so we would have pretty high confidence that our random forest model with the selected hyperparameters would perform well when predicting new data.
last_rf_fit %>% 
  collect_predictions() %>% 
  roc_curve(children,.pred_children) %>% 
  autoplot()
