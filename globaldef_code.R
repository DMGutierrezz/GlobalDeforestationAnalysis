##################################
#### EXTREM GRADIENT BOOSTING ####
##################################

# Code originally written by Raphael Ganzenmueller (LMU) and modified by Diana Gutierrez (CIAT) for the present analysis and extended to visualise the results.
# mail: ganzenmueller.r@posteo.de ; d.m.gutierrez@cgiar.org
# date: October 2023

#======================================================
# load libraries
#======================================================

library(plyr)
library(tidyverse)
library(xgboost)
library(caret)

#=====================================================
# set pathes
#=====================================================

#model1 (all var)
path_output = "//MyFolder/globaldef/xgb2023"

#======================================================
# define parameters and number of model runs
#======================================================

# parameters
cv_folds = 10 #complete sets of folds to compute
cv_repeats = 10 #for repeated k-fold cross-validation 
min_rsamp = 5 #minimum number of resamples used before models were removed
tlength = 100 #initialized models 

# number of model runs
nrun = 5 #vary according to the number of runs required

#======================================================
# load data
#======================================================

# load data base

db_globaldef <- readr::read_csv("//MyFolder/globaldef/globaldef_2004_2021.csv")
summary(db_globaldef)

##adding AUS to Asia continent

table(db_globaldef$continent)
db_globaldef$continent=ifelse(db_globaldef$iso3=="AUS","asia",db_globaldef$continent)
table(db_globaldef$continent)

unique(db_globaldef$year)


#======================================================
# data preparation
#======================================================


# define list with continents with more than 5 countries (if not enough data points xgb output don't make sense)
contn <- db_globaldef %>%
  select(iso3, continent) %>%
  unique() %>%
  group_by(continent) %>%
  count(continent) %>%
  filter(n>=5) %>%
  pull(continent)

# define variable for whole world (no filtering)
glb <- "glb"



#======================================================
# function to find a "good" xgb model for each cluster
#======================================================

xgb.model.function <- function(level, run, cv_folds, cv_repeats, min_rsamp, tlength){
  fseed=run^3 #seed defined using variable run
  
  # start time 
  time_start <- Sys.time()
  
  # print current level and run
  cat(crayon::red(paste("starting\n[group: ", level, "]\n[run: run", run, "]\n", sep="")))
  
  # subset data according to selected level
  if(level %in% contn) {
    db_globaldef_level = db_globaldef %>% 
      filter(continent == level)
    print(paste("continent == ", level, sep=""))
  } else if(level == "glb") {
    db_globaldef_level = db_globaldef
    print(paste("no filter, taking all contries", sep=""))
  } else stop("problem with group selection in xgb.model.function")
  
  #print heads of data to be used
  print(summary(db_globaldef_level))
  print(db_globaldef_level %>% arrange(iso3))
  
  # select relevant columns
  data_model <- db_globaldef_level %>%
    select(df_ha, food_infl,rural_pop,gdp_perc,exports_gdp,food_exports,food_imports,
           tmp_change, foreing_invest,dem_mdn,trv_med,gdp_growth,pop_growth)
  
  # create training and testing data set
  set.seed(fseed) # set seed for reproducibility
  indata <- createDataPartition(y = data_model$df_ha, p = 0.8)[[1]] # index for testing and training data
  training <- data_model[indata,] 
  testing <- data_model[-indata,]
  
  # trainControl object for repeated cross-validation with random search 
  adaptControl <- trainControl(method = "adaptive_cv",#the full set of resamples is not run for each model. As resampling continues, a futility analysis is conducted and models with a low probability of being optimal are removed
                               number = cv_folds, 
                               repeats = cv_repeats,
                               adaptive = list(min = min_rsamp,#the minimum number of resamples used before models are removed
                                               alpha = 0.05, #the confidence level of the one-sided intervals used to measure futility
                                               method = "gls",
                                               complete = FALSE),
                               search = "random",#describing how the tuning parameter grid is determined (random search procedure)
                               verbose=TRUE,
                               allowParallel = TRUE)
  
  # train model
  set.seed(fseed) # set seed for reproducibility
  xgb_model <- train(df_ha ~ ., data = training,
                     method = "xgbTree", 
                     trControl = adaptControl,
                     metric = "RMSE",
                     tuneLength = tlength,#By default, this argument is the number of levels for each tuning parameters that should be generated by train. If trainControl has the option search = "random", this is the maximum number of tuning parameter combinations that will be generated by the random search
                     na.action = na.pass)
  
  # model summary
  model_summary <- xgb_model$results %>%
    arrange(RMSE) %>%
    head(1) %>%
    transmute(rsquared = Rsquared, rmse = RMSE, eta, max_depth, gamma, colsample_bytree, 
              min_child_weight, subsample, nrounds)
  
  # prepare testing data
  x_test <- select(testing, -df_ha)
  y_test <- testing$df_ha
  
  # apply test data to model
  predicted <- predict(xgb_model, x_test, na.action=na.pass)
  
  # calcuate residuals
  test_residuals <- y_test - predicted
  
  # calculation of rsquare
  tss <- sum((y_test - mean(y_test))^2) #total sum of squares
  rss <- sum(test_residuals^2) #residual sum of squares
  test_rsquared <- 1 - (rss/tss) #rsquare
  
  # calculate root mean square error
  test_rmse <- sqrt(mean(test_residuals^2))
  # calculate mean absolut error
  test_mae <- mean(abs(test_residuals))
  
  # model evaluation
  model_eval <- list(x_test=x_test, y_test=y_test, predicted=predicted, test_residuals=test_residuals, 
                     test_rsquared=test_rsquared, test_rmse=test_rmse, test_mae=test_mae)
  
  # variable importance
  
  #model1
  model_var_importance <- 
    xgb.importance(feature_names = dimnames(data_model[,-1])[[2]], model = xgb_model$finalModel) %>%
    transmute(Feature, Gain) %>%
    pivot_wider(names_from = Feature, values_from = Gain, names_prefix = "imp_") %>% 
    transmute(imp_gdp_perc = ifelse(exists("imp_gdp_perc"), imp_gdp_perc, 0),
              imp_rural_pop = ifelse(exists("imp_rural_pop"), imp_rural_pop, 0),
              imp_gdp_growth = ifelse(exists("imp_gdp_growth"), imp_gdp_growth, 0),
              imp_exports_gdp = ifelse(exists("imp_exports_gdp"), imp_exports_gdp, 0),
              imp_food_exports = ifelse(exists("imp_food_exports"), imp_food_exports, 0),
              imp_food_imports = ifelse(exists("imp_food_imports"), imp_food_imports, 0),
              imp_food_infl = ifelse(exists("imp_food_infl"), imp_food_infl, 0),
              imp_foreing_invest = ifelse(exists("imp_foreing_invest"), imp_foreing_invest, 0),
              imp_tmp_change = ifelse(exists("imp_tmp_change"), imp_tmp_change, 0),
              imp_dem_mdn = ifelse(exists("imp_dem_mdn"), imp_dem_mdn, 0),
              imp_trv_med = ifelse(exists("imp_trv_med"), imp_trv_med, 0),
              imp_pop_growth = ifelse(exists("imp_pop_growth"), imp_pop_growth, 0)              
    )
  
  
  # save model settings
  model_settings <- data.frame(level=level, seed=fseed, cv_folds=cv_folds, cv_repeats= cv_repeats, 
                               min_rsamp=min_rsamp, tlength=tlength)
  
  # prepare output
  output <- list(xgb_model, model_settings, model_summary, model_eval, model_var_importance, training, testing)
  names(output) <- c(paste("xgb_model", level, sep = "_"),
                     paste("model_settings", level, sep = "_"),
                     paste("model_summary", level, sep = "_"),
                     paste("model_evaluation", level, sep = "_"),
                     paste("model_var_importance", level, sep = "_"),
                     paste("data_training", level, sep = "_"),
                     paste("data_testing", level, sep = "_"))
  
  # stop time
  time_end <- Sys.time()
  time_run <- time_end-time_start
  
  # print run time
  cat(crayon::green(paste("[model run time: ", round(time_run, 2), "]\n", sep="")))
  
  # write time file
  run_time <- data.frame(time_start, time_end, time_run, level, run)
  
  write.table(run_time, file.path(path_output, "run_time.csv"), 
              sep = ",", col.names = FALSE, append=TRUE, row.names = FALSE)
  
  #return xgb output
  assign(paste("output", level, fseed, sep = "_"), output)
}


#======================================================
# function to iterate over levels (global, continents)
#======================================================

xgb.run.function <- function(level, run, cv_folds, cv_repeats, min_rsamp, tlength){
  group = ifelse(identical(level, contn), "contn",
                 ifelse(identical(level, glb), "glb", 
                        stop("problem with group selection in xgb.run.function")))
  
  run_x <- lapply(level, xgb.model.function, run, cv_folds, cv_repeats, min_rsamp, tlength)
  names(run_x) <- level
  saveRDS(run_x, file=file.path(path_output, paste(group, "_run", run, ".RDS", sep="")))
}




#======================================================
# run model
#======================================================

#write run file and save starting time
write.table(Sys.time(), file.path(path_output, "run_time.csv"), sep = ",", row.names = FALSE)


# run with level glb
level = glb
lapply(c(1:nrun), xgb.run.function, level = level, cv_folds = cv_folds, cv_repeats = cv_repeats, min_rsamp = min_rsamp, tlength = tlength)

# run on level continent
level = contn
lapply(c(1:nrun), xgb.run.function, level = level, cv_folds = cv_folds, cv_repeats = cv_repeats, min_rsamp = min_rsamp, tlength = tlength)

# save run file and save ending time
write.table(Sys.time(), file.path(path_output, "run_time.csv"), sep = ",", col.names = FALSE, append=TRUE, row.names = FALSE)



#======================================================
# Read the results
#======================================================


#global level

glb1 <- readRDS("//MyFolder/globaldef/xgb2023/glb_run1.rds")
glb2 <- readRDS("//MyFolder/globaldef/xgb2023/glb_run2.rds")
glb3 <- readRDS("//MyFolder/globaldef/xgb2023/glb_run3.rds")
glb4 <- readRDS("//MyFolder/globaldef/xgb2023/glb_run4.rds")
glb5 <- readRDS("//MyFolder/globaldef/xgb2023/glb_run5.rds")

#regional level

reg1 <- readRDS("//MyFolder/globaldef/xgb2023/contn_run1.rds")
reg2 <- readRDS("//MyFolder/globaldef/xgb2023/contn_run2.rds")
reg3 <- readRDS("//MyFolder/globaldef/xgb2023/contn_run3.rds")
reg4 <- readRDS("//MyFolder/globaldef/xgb2023/contn_run4.rds")
reg5 <- readRDS("//MyFolder/globaldef/xgb2023/contn_run5.rds")


#======================================================
# Check the results and compute means
#======================================================


#global level#

df_glb1 <- data.frame(model = 'run1', rsquared = glb1[["glb"]][["model_summary_glb"]][["rsquared"]], model = glb1[["glb"]][["model_var_importance_glb"]])
df_glb2 <- data.frame(model = 'run2', rsquared = glb2[["glb"]][["model_summary_glb"]][["rsquared"]], model = glb2[["glb"]][["model_var_importance_glb"]])
df_glb3 <- data.frame(model = 'run3', rsquared = glb3[["glb"]][["model_summary_glb"]][["rsquared"]], model = glb3[["glb"]][["model_var_importance_glb"]])
df_glb4 <- data.frame(model = 'run4', rsquared = glb4[["glb"]][["model_summary_glb"]][["rsquared"]], model = glb4[["glb"]][["model_var_importance_glb"]])
df_glb5 <- data.frame(model = 'run5', rsquared = glb5[["glb"]][["model_summary_glb"]][["rsquared"]], model = glb5[["glb"]][["model_var_importance_glb"]])

glob=rbind(df_glb1,df_glb2,df_glb3,df_glb4,df_glb5)
df_glb=data.frame(lapply(glob[-1], mean))%>%
       mutate(level = 'Global')
rowSums(df_glb[,2:(ncol(df_glb)-1)])
rowSums(glob[,3:ncol(glob)])
lapply(glob[-1], mean) 

#regional level#

africa1 <- data.frame(model = 'africa', rsquared = reg1[["africa"]][["model_summary_africa"]][["rsquared"]], model = reg1[["africa"]][["model_var_importance_africa"]])
africa2 <- data.frame(model = 'africa', rsquared = reg2[["africa"]][["model_summary_africa"]][["rsquared"]], model = reg2[["africa"]][["model_var_importance_africa"]])
africa3 <- data.frame(model = 'africa', rsquared = reg3[["africa"]][["model_summary_africa"]][["rsquared"]], model = reg3[["africa"]][["model_var_importance_africa"]])
africa4 <- data.frame(model = 'africa', rsquared = reg4[["africa"]][["model_summary_africa"]][["rsquared"]], model = reg4[["africa"]][["model_var_importance_africa"]])
africa5 <- data.frame(model = 'africa', rsquared = reg5[["africa"]][["model_summary_africa"]][["rsquared"]], model = reg5[["africa"]][["model_var_importance_africa"]])

africa=rbind(africa1,africa2,africa3,africa4,africa5)
df_africa=data.frame(lapply(africa[-1], mean)) %>%
  mutate(level = 'Africa')
rowSums(df_africa[,2:(ncol(df_africa)-1)])
rowSums(africa[,3:ncol(africa)])
lapply(africa[-1], mean) 

asia1 <- data.frame(model = 'asia', rsquared = reg1[["asia"]][["model_summary_asia"]][["rsquared"]], model = reg1[["asia"]][["model_var_importance_asia"]])
asia2 <- data.frame(model = 'asia', rsquared = reg2[["asia"]][["model_summary_asia"]][["rsquared"]], model = reg2[["asia"]][["model_var_importance_asia"]])
asia3 <- data.frame(model = 'asia', rsquared = reg3[["asia"]][["model_summary_asia"]][["rsquared"]], model = reg3[["asia"]][["model_var_importance_asia"]])
asia4 <- data.frame(model = 'asia', rsquared = reg4[["asia"]][["model_summary_asia"]][["rsquared"]], model = reg4[["asia"]][["model_var_importance_asia"]])
asia5 <- data.frame(model = 'asia', rsquared = reg5[["asia"]][["model_summary_asia"]][["rsquared"]], model = reg5[["asia"]][["model_var_importance_asia"]])

asia=rbind(asia1,asia2,asia3,asia4,asia5)
df_asia=data.frame(lapply(asia[-1], mean)) %>%
  mutate(level = 'Asia')
rowSums(df_asia[,2:(ncol(df_asia)-1)])
rowSums(asia[,3:ncol(asia)])
lapply(asia[-1], mean) 

latin1 <- data.frame(model = 'latin', rsquared = reg1[["latin"]][["model_summary_latin"]][["rsquared"]], model = reg1[["latin"]][["model_var_importance_latin"]])
latin2 <- data.frame(model = 'latin', rsquared = reg2[["latin"]][["model_summary_latin"]][["rsquared"]], model = reg2[["latin"]][["model_var_importance_latin"]])
latin3 <- data.frame(model = 'latin', rsquared = reg3[["latin"]][["model_summary_latin"]][["rsquared"]], model = reg3[["latin"]][["model_var_importance_latin"]])
latin4 <- data.frame(model = 'latin', rsquared = reg4[["latin"]][["model_summary_latin"]][["rsquared"]], model = reg4[["latin"]][["model_var_importance_latin"]])
latin5 <- data.frame(model = 'latin', rsquared = reg5[["latin"]][["model_summary_latin"]][["rsquared"]], model = reg5[["latin"]][["model_var_importance_latin"]])

latin=rbind(latin1,latin2,latin3,latin4,latin5)
df_latin=data.frame(lapply(latin[-1], mean)) %>%
  mutate(level = 'LAC')
rowSums(df_latin[,2:(ncol(df_latin)-1)])
rowSums(latin[,3:ncol(latin)])
lapply(latin[-1], mean) 

#create a dataframe for means by level 

df <- rbind(df_glb,df_africa,df_asia,df_latin)



#======================================================
# Generate results graph
#======================================================


# Load libraries 

library(tidyverse)
library(doBy)


# data preparation

datos=df %>%
  pivot_longer(
    cols = starts_with("model.imp_"),
    names_to = "variable",
    values_to = "RI",
  )

datos$variable=gsub('model.imp_', '', datos$variable)

datos=mutate(datos, variable = factor(variable,levels = c("dem_mdn","tmp_change","trv_med","exports_gdp",       
                                                  "food_exports","food_imports","foreing_invest","food_infl",    
                                                  "gdp_growth","gdp_perc","pop_growth","rural_pop")))
datos <- mutate(datos, level = factor(level, levels = c('Africa', 'Asia', 'LAC', 'Global')))

datos=datos[do.call(order,datos[c("variable","level")]),]


#display graph and save

gbar <- ggplot(data=datos, aes(x=variable, y=RI,fill=level)) +  
  geom_bar(stat="identity", position=position_dodge())+
  labs(x="Variable",y="Relative importance (RI)",fill="Level")+
  scale_fill_manual(labels = c("Africa","Asia & Oceania","Latin America &\nthe Caribbean","Global"),values=c("#F7DC6F","#EC7063","#48C9B0","#5DADE2"))+
  theme_light() +
  theme(
    axis.text.y = element_text(size = 6),
    axis.text.x = element_text(size = 6,angle = 90),
    axis.title.x = element_text(size = 8),
    axis.title.y = element_text(size = 8),
    legend.title = element_text(hjust=0.5,size=6),
    legend.text = element_text(size=6)
  )
gbar
ggsave(plot = gbar, filename = '//MyFolder/globaldef/xgb2023/results_xgb2023.png', units = 'in', width = 6, height = 4, dpi = 300)
