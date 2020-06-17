#1. Introduction
#we’ll explore recipes, which is designed to help you preprocess your data before training your model

library(tidymodels)      # for the recipes package, along with the rest of tidymodels

# Helper packages
library(nycflights13)    # for flight data
library(skimr)           # for variable summaries


#2. The New York City flight data
flight_data <- 
  flights %>% 
  mutate(
    # Convert the arrival delay to a factor
    arr_delay = ifelse(arr_delay >= 30, "late", "on_time"),
    arr_delay = factor(arr_delay),
    # We will use the date (not date-time) in the recipe below
    date = as.Date(time_hour)
  ) %>% 
  # Include the weather data
  #inner_join(weather, by = c("origin", "time_hour")) %>% 
  # Only retain the specific columns we will use
  select(dep_time, flight, origin, dest, air_time, distance, 
         carrier, date, arr_delay, time_hour) %>% 
  # Exclude missing data
  na.omit() %>% 
  # For creating models, it is better to have qualitative columns
  # encoded as factors (instead of character strings)
  mutate_if(is.character, as.factor)

#We can see that about 16% of the flights in this data set arrived more than 30 minutes late.
flight_data %>% 
  count(arr_delay) %>% mutate(prop = n/sum(n))

#let’s take a quick look at a few specific variables that will be important for both preprocessing and modeling
glimpse(flight_data)

#There are two variables that we don’t want to use as predictors in our model, but that we would like to retain as identification variables that can be used to troubleshoot poorly predicted data points. These are flight, a numeric value, and time_hour, a date-time value
flight_data %>% 
  skimr::skim(dest, carrier) 

#3. Data splitting 
# let’s split this single dataset into two: a training set and a testing set,to do this, we can use the rsample package to create an object that contains the information on how to split the data, and then two more rsample functions to create data frames for the training and testing sets:
# Fix the random numbers by setting the seed 
# This enables the analysis to be reproducible when random numbers are used 
set.seed(555)
# Put 3/4 of the data into the training set 
data_split <- initial_split(flight_data, prop = 3/4)

# Create data frames for the two sets:
train_data <- training(data_split)
test_data  <- testing(data_split)

#4. Create recipe and roles
#let’s create a recipe for a simple logistic regression model
flights_recipe <- 
  recipe(arr_delay ~ ., data = train_data) 

#The recipe() function as we used it here has two arguments:
#1.A formula. Any variable on the left-hand side of the tilde (~) is considered the model outcome (here, arr_delay)
#2.The data. A recipe is associated with the data set used to create the model.

#We can use the update_role() function to let recipes know that flight and time_hour are variables with a custom role that we called "ID",this tells the recipe to keep these two variables but not use them as either outcomes or predictors.
flights_recipe <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") 

#5. Create features
#
flights_recipe <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") %>% 
  #we create two new factor columns with the appropriate day of the week and the month
  step_date(date, features = c("dow", "month")) %>% 
  #we create a binary variable indicating whether the current date is a holiday or not
  step_holiday(date, holidays = timeDate::listHolidays("US")) %>%
  #remove the original date variable
  step_rm(date) %>% 
  #Create dummy variables for all of the factor or character columns unless they are outcomes.
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  #remove columns from the data when the training set data have a single value in this case dest_LEX  
  step_zv(all_predictors())

#If you want to extract the pre-processed dataset itself, you can first prep() the recipe for a specific dataset and juice() the prepped recipe to extract the pre-processed data
flights_train_preprocessed <- flights_recipe %>%
  # apply the recipe to the training data
  prep(train_data) %>%
  # extract the pre-processed training dataset
  juice()

flights_train_preprocessed

#6. Fit a model with a recipe 
# We will use a logistic regression model from Build a Model
lr_mod <- 
  logistic_reg() %>% set_engine("glm")

#use a model workflow, to pair a model and recipe together
flights_wflow <- 
  workflow() %>% add_model(lr_mod) %>% add_recipe(flights_recipe)

#now there is a single function that can be used to prepare the recipe and train the model
flights_fit <- 
  flights_wflow %>% fit(data = train_data)

#You may want to extract the model or recipe objects from the workflow
flights_fit %>% 
  pull_workflow_fit() %>% tidy()

#7. Use a trained workflow to predict
#use the trained workflow (flights_fit) to predict with the unseen test data
predict(flights_fit, test_data)

#To return predicted class probabilities for each flight,we can specify type = "prob" when we use predict()
flights_pred <- 
  predict(flights_fit, test_data, type = "prob") %>% 
  bind_cols(test_data %>% select(arr_delay, time_hour, flight)) 

#We can create the ROC curve with these values, using roc_curve() and then piping to the autoplot() method:
flights_pred %>% 
  roc_curve(truth = arr_delay, .pred_late) %>% autoplot()

#Similarly, roc_auc() estimates the area under the curve:
flights_pred %>% 
  roc_auc(truth = arr_delay, .pred_late)
