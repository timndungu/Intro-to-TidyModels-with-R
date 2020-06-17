#1. Introduction 
#Once we have a model trained, we need a way to measure how well that model predicts new data

library(tidymodels) # for the rsample package, along with the rest of tidymodels
# Helper packages
library(modeldata)  # for the cells data


#2. The cell image data
data(cells, package = "modeldata")
cells <- cells %>% select(-case) #remove train/test split by authors(case)

#The rates of the classes are somewhat imbalanced; there are more poorly segmented cells than well-segmented cells:
cells %>% 
  count(class) %>% mutate(prop = n/sum(n))

#3. Data splitting
#Here we use the strata argument, which conducts a stratified split. This ensures that, despite the imbalance we noticed in our class variable, our training and test data sets will keep roughly the same proportions of poorly and well-segmented cells as in the original data
set.seed(123)
cell_split <- initial_split(cells, strata = class)

cell_train <- training(cell_split)
cell_test  <- testing(cell_split)


#3. Modeling
#One of the benefits of a random forest model is that it is very low maintenance; it requires very little preprocessing of the data and the default parameters tend to give reasonable results. For that reason, we won’t create a recipe for the cells data.
#If you want to be able to examine the variable importance of your final model later, you will need to set importance argument when setting the engine. For ranger, the importance options are "impurity" or "permutation".

rf_model <- 
  #specify that the model is a random forest with 1000 trees
  rand_forest(trees = 1000) %>% 
  #select the engine/package that underlies the model
  set_engine("ranger", importance = "impurity") %>% 
  #choose either the continuous regression or binary classification mode
  set_mode("classification")

#Starting with this parsnip model object, the fit() function can be used with a model formula. Since random forest models use random numbers, we again set the seed prior to computing:
set.seed(234)
rf_workflow <- workflow() %>% add_model(rf_model) %>% add_formula(class ~ .)

rf_fit <- rf_workflow %>% fit(cell_train)

#4. Estimating performance
#some options we could use are:
#the area under the Receiver Operating Characteristic (ROC) curve, and
#overall classification accuracy.
#The yardstick package has functions for computing both of these measures called roc_auc() and accuracy().
#To evaluate performance based on the training set, we call the predict() method to get both types of predictions (i.e. probabilities and hard class predictions).
rf_training_pred <- 
  predict(rf_fit, cell_train) %>% 
  bind_cols(predict(rf_fit, cell_train, type = "prob")) %>% 
  # Add the true outcome data back in
  bind_cols(cell_train %>% select(class))

rf_training_pred %>%                # training set predictions
  roc_auc(truth = class, .pred_PS)

rf_training_pred %>%                # training set predictions
  accuracy(truth = class, .pred_class)

#We proceed to the test set. Unfortunately, we discover that, although our results aren’t bad, they are certainly worse than what we initially thought based on predicting the training set:
rf_testing_pred <- 
  predict(rf_fit, cell_test) %>% 
  bind_cols(predict(rf_fit, cell_test, type = "prob")) %>% 
  bind_cols(cell_test %>% select(class))

rf_testing_pred %>%                   # test set predictions
  roc_auc(truth = class, .pred_PS)

rf_testing_pred %>%                   # test set predictions
  accuracy(truth = class, .pred_class)

#5. Resampling to the rescue
#Resampling methods, such as cross-validation and the bootstrap, are empirical simulation systems. They create a series of data sets similar to the training/testing split,Resampling is always used with the training set.
#Let’s use 10-fold cross-validation (CV) in this example. This method randomly allocates the 1515 cells in the training set to 10 groups of roughly equal size, called “folds”. For the first iteration of resampling, the first fold of about 151 cells are held out for the purpose of measuring performance. This is similar to a test set but, to avoid confusion, we call these data the assessment set in the tidymodels framework.
#The other 90% of the data (about 1363 cells) are used to fit the model. Again, this sounds similar to a training set, so in tidymodels we call this data the analysis set. This model, trained on the analysis set, is applied to the assessment set to generate predictions, and performance statistics are computed based on those predictions.

#6. Fit a model with resampling
#There are several resampling methods implemented in rsample; cross-validation folds can be created using vfold_cv():
set.seed(345)
folds <- vfold_cv(cell_train, v = 10)
folds

#The list column for splits contains the information on which rows belong in the analysis and assessment sets. There are functions that can be used to extract the individual resampled data called analysis() and assessment().
#the tune package contains high-level functions that can do the required computations to resample a model for the purpose of measuring performance

set.seed(456)
rf_fit_rs <- 
  rf_workflow %>% fit_resamples(folds)

#the performance statistics can be manually unnested
rf_fit_rs %>% unnest(c(.metrics)) %>% select(id, .metric, .estimate) %>% pivot_wider(names_from = .metric, values_from = .estimate) %>% arrange(accuracy)

#The column .metrics contains the performance statistics created from the 10 assessment sets.
collect_metrics(rf_fit_rs)


