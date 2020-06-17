#1. Introduction
#Some model parameters cannot be learned directly from a data set during model training; these kinds of parameters are called hyperparameters.
#Some examples of hyperparameters include the number of predictors that are sampled at splits in a tree-based model (we call this mtry in tidymodels) or the learning rate in a boosted tree model (we call this learn_rate).
library(tidymodels)  # for the tune package, along with the rest of tidymodels

# Helper packages
library(modeldata)   # for the cells data
library(vip)         # for variable importance plots

#2. The cell image data, revisited 
data(cells, package = "modeldata")
cells

#3.Predicting image segmentation, but better
#Let’s explore:
#the complexity parameter (which we call cost_complexity in tidymodels) for the tree, and
#the maximum tree_depth.
#we split our data into training and testing sets
set.seed(123)
cell_split <- initial_split(cells %>% select(-case), strata = class)

cell_train <- training(cell_split)
cell_test  <- testing(cell_split)

#4.Tuning hyperparameters
#we create a model specification that identifies which hyperparameters we plan to tune using a decision_tree() model with the rpart engine.
tune_spec <- 
  decision_tree(cost_complexity = tune(), tree_depth = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

tune_spec

#Think of tune() here as a placeholder. After the tuning process, we will select a single numeric value for each of these hyperparameters. 
#we can train many models using resampled data and see which models turn out best.We can create a regular grid of values to try using some convenience functions for each hyperparameter:
tree_grid <- 
  grid_regular(cost_complexity(),tree_depth(),levels = 5)

#all 5 values of cost_complexity ranging up to 0.1. These values get repeated for each of the 5 values of tree_depth:
tree_grid %>% 
  count(tree_depth)

#create cross-validation folds for tuning,tuning in tidymodels requires a resampled object
set.seed(234)
cell_folds <- vfold_cv(cell_train)

#5.Model tuning with a grid
#Let’s use tune_grid() to fit models at all the different values we chose for each tuned hyperparameter
set.seed(345)
tree_workflow <- 
  workflow() %>% add_model(tune_spec) %>% add_formula(class ~ .)

tree_resamples <- 
  tree_workflow %>% 
  tune_grid(resamples = cell_folds, grid = tree_grid)

tree_resamples

#The function collect_metrics() gives us a tidy tibble with all the results. We had 25 candidate models and two metrics, accuracy and roc_auc, and we get a row for each 
tree_resamples %>% 
  collect_metrics()

#We might get more out of plotting these results:
tree_resamples %>%
  collect_metrics() %>%
  mutate(tree_depth = factor(tree_depth)) %>%
  ggplot(aes(cost_complexity, mean, color = tree_depth)) +
  geom_line(size = 1.5, alpha = 0.6) +
  geom_point(size = 2) +
  facet_wrap(~ .metric, scales = "free", nrow = 2) +
  scale_x_log10(labels = scales::label_number()) +
  scale_color_viridis_d(option = "plasma", begin = .9, end = 0)

#The show_best() function shows us the top 5 candidate models by default:
tree_resamples %>%
  show_best("roc_auc")

#We can also use the select_best() function to pull out the single set of hyperparameter values for our best decision tree model:
best_tree <- tree_resamples %>%
  select_best("roc_auc")

#6. Finalizing our model
#We can update (or “finalize”) our workflow object tree_wf with the values from select_best().
final_workflow <- 
  tree_workflow %>% finalize_workflow(best_tree)

#7.Exploring results
#Let’s fit this final model to the training data.
final_tree <- 
  final_workflow %>% fit(data = cell_train)

#This final_tree object has the finalized, fitted model object inside. You may want to extract the model object from the workflow. To do this, you can use the helper function pull_workflow_fit().
#We can use the vip package to estimate variable importance.
final_tree %>% 
  pull_workflow_fit() %>% vip()

#7. The last fit
#We can use the function last_fit() with our finalized model; this function fits the finalized model on the full training data set and evaluates the finalized model on the testing data.
final_fit <- 
  final_workflow %>% last_fit(cell_split)

final_fit %>%
  collect_metrics()

#plotting
final_fit %>% 
  collect_predictions() %>% roc_curve(class,.pred_PS) %>% autoplot()
