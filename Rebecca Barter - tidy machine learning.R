#http://www.rebeccabarter.com/blog/2020-03-25_machine_learning/
# load the relevant tidymodels libraries
library(tidymodels)
library(tidyverse)
library(workflows)
library(tune)
# load the Pima Indians dataset from the mlbench dataset
library(mlbench)

#dataset2 where all zero values of glucose, pressure, triceps, insulin and mass have been set to NA
data("PimaIndiansDiabetes2")

#rename dataset to have shorter name because lazy
diabetes_clean <- 
  PimaIndiansDiabetes2

#Split into train/test
set.seed(234589)
splits <- 
  initial_split(diabetes_clean)

diabetes_train <- training(splits)
diabetes_test <- testing(splits)

#we can create a cross-validated version of the training set in preparation for that moment using vfold_cv()
diabetes_cv <- vfold_cv(diabetes_train)

#Define a recipe
diabetes_recipe <- 
  recipe(diabetes ~ ., data = diabetes_train) %>% 
  #center and scale numeric variables
  step_normalize(all_numeric()) %>% 
  #remove indicator variables that only contain a single unique value
  step_nzv(all_predictors()) %>% 
  #impute missing values using knn
  step_knnimpute(all_predictors())

#If you want to extract the pre-processed dataset itself, you can first prep() the recipe for a specific dataset and juice()
diabetes_train_preprocessed <- 
  diabetes_recipe %>% 
  prep(diabetes_train) %>% 
  juice()

diabetes_train_preprocessed

#To enable parallel processing, we can pass engine-specific arguments like num.threads to ranger when we set the engine
#This bears repeating: if you use any other resampling method, let tune do the parallel processing for you 
cores <- parallel::detectCores()

#Specify the model
rf_model <- 
  #specify that the model is a random forest and set the `mtry` parameter needs to be tuned
  rand_forest(mtry = tune()) %>% 
  #select the engine/package that underlies the model
  set_engine("ranger", importance = "impurity") %>% 
  #choose either the continuous regression or binary classification mode
  set_mode("classification")

#Put it all together in a workflow
rf_workflow <- 
  workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(diabetes_recipe)

#Tune the parameters
## specify which values to try
rf_grid <- 
  expand.grid(mtry = c(3,4,5))

## extract results
rf_tune_results <- 
  rf_workflow %>% 
  tune_grid(
    #CV object
    resamples = diabetes_cv,
    ## grid of values to try
    grid = rf_grid,
    ## metrics we care about
    metrics = metric_set(roc_auc,accuracy),
    #save predictions
    control = control_grid(save_pred = TRUE)
  )
# print results
rf_tune_results %>%
  collect_metrics()
#Across both accuracy and AUC, mtry = 3 yields the best performance (just).


#Finalize the workflow
final_param <- 
  rf_tune_results %>%
  select_best(metric = "accuracy")

#Then we can add this parameter to the workflow using the finalize_workflow() function.
rf_workflow <- 
  rf_workflow %>% 
  finalize_workflow(final_param)

#Evaluate the model on the test set
#we will apply the last_fit() function to our workflow and our train/test split object.
rf_fit <- rf_workflow %>%
  # fit on the training set and evaluate on test set
  last_fit(splits)

