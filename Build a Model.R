#1.Introduction
library(tidymodels)  # for the parsnip package, along with the rest of tidymodels

# Helper packages
library(readr)       # for importing data

#2.The Sea Urchins Data
urchins <-
  # Data were assembled for a tutorial 
  # at https://www.flutterbys.com.au/stats/tut/tut7.5a.html
  read_csv("https://tidymodels.org/start/models/urchins.csv") %>% 
  # Change the names to be a little more verbose
  setNames(c("food_regime", "initial_volume", "width")) %>% 
  # Factors are very helpful for modeling, so we convert one column
  mutate(food_regime = factor(food_regime, levels = c("Initial", "Low", "High")))

#As a first step in modeling, it’s always a good idea to plot the data:
ggplot(data = urchins, aes(x = initial_volume, y = width, group = food_regime, col = food_regime)) +
  #plot a scatter plot
  geom_point() +
  #plot a fitted linear model
  geom_smooth(method = lm, se = FALSE) +
  #Aesthetics 
  scale_color_viridis_d(option = "plasma", end = .7)


#3. Build and fit a model
#A standard two-way analysis of variance (ANOVA) model makes sense for this dataset because we have both a continuous predictor and a categorical predictor. 
width ~ initial_volume * food_regime
#start by specifying the functional form of the model that we want using the parsnip package and to use ordinary least squares, set the engine to be lm and save it as a model object:
lm_mod <- 
  linear_reg() %>% set_engine("lm")

#Estimate or train the model using fit()
lm_fit <- 
  lm_mod %>% fit(width ~ initial_volume * food_regime, data = urchins)

#use the tidy.model_fit() method that provides the summary results in a more predictable and useful format
tidy.model_fit(lm_fit)

# 4. Use a model to predict
#Make new example data that we will make predictions for
new_points <- expand.grid(initial_volume = 20, 
                          food_regime = c("Initial", "Low", "High"))
# use the predict() function to find the mean values at 20ml
mean_pred <- predict(lm_fit, new_data = new_points)

#the tidymodels convention is to always produce a tibble of results with standardized column names
conf_int_pred <- predict(lm_fit, 
                         new_data = new_points, 
                         type = "conf_int")

# Now combine: 
plot_data <- 
  new_points %>% 
  bind_cols(mean_pred) %>% 
  bind_cols(conf_int_pred)

# and plot:
ggplot(plot_data, aes(x = food_regime)) + 
  geom_point(aes(y = .pred)) + 
  geom_errorbar(aes(ymin = .pred_lower, 
                    ymax = .pred_upper),
                width = .2) + 
  labs(y = "urchin size")

#5. Model with a different engine
#the model can be estimated using a Bayesian approach
#In such an analysis, a prior distribution needs to be declared for each model parameter that represents the possible values of the parameters
# set the prior distribution a Cauchy distribution (which is the same as a t-distribution with a single degree of freedom)
prior_dist <- rstanarm::student_t(df = 1)

set.seed(123)

# make the parsnip model
bayes_mod <-   
  linear_reg() %>% set_engine("stan", 
                              prior_intercept = prior_dist, 
                              prior = prior_dist) 
# train the model
bayes_fit <- 
  bayes_mod %>% fit(width ~ initial_volume * food_regime, data = urchins)

#format and tidy the model
tidy.model_fit(bayes_fit,intervals = TRUE)

#combine the bayes data
bayes_plot_data <- 
  new_points %>% 
  bind_cols(predict(bayes_fit, new_data = new_points)) %>% 
  bind_cols(predict(bayes_fit, new_data = new_points, type = "conf_int"))

#and plot
ggplot(bayes_plot_data, aes(x = food_regime)) + 
  geom_point(aes(y = .pred)) + 
  geom_errorbar(aes(ymin = .pred_lower, ymax = .pred_upper), width = .2) + 
  labs(y = "urchin size") + 
  ggtitle("Bayesian model with t(1) prior distribution")

#6. Why does it work that way?

#The problem with standard modeling functions is that they don’t separate what you want to do from the execution. For example, the process of executing a formula has to happen repeatedly across model calls even when the formula does not change; we can’t recycle those computations.
#
