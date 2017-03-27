# Script To use the following Decission Tree and Ensemble techniques for simple predictive analytics
# The script is intentionally kept simple to cover the basics of using simple decission tree and ensemble techniques

# Play with data
?mtcars
mtcars
plot(mtcars$mpg,mtcars$carb)
finalResult = list()

#Sample the data set
smp_size <- floor(0.75 * nrow(mtcars))
train_ind <- sample(seq_len(nrow(mtcars)), size = smp_size)
train <- mtcars[train_ind,]
test <- mtcars[-train_ind,]
test_orig <- test$mpg

# Generate formula
f <- as.formula(paste("mpg ~ ",paste(names(train)[-1],collapse = " + ")))
library(rpart)


#####################################################################################
#Decission Tree Refresher: http://snap.stanford.edu/class/cs246-2015/slides/14-dt.pdf
#build a simple decission tree model
carfit_dt <- rpart(f, data = train, method = "anova")

#Predict using Decission Tree
test_pred_dt <- predict(carfit_dt, test[,-1])

#Calculate accuracy
med <- median(test_orig)
RMSE_dt <- sqrt(mean((test_orig-test_pred_dt)^2))
finalResult$DT <- (1-RMSE_dt/med)

#####################################################################################
# Random Forest Simple explanation: http://trevorstephens.com/post/73770963794/titanic-getting-started-with-r-part-5-random
#build a random forest model
set.seed(415)
install.packages('randomForest')
library(randomForest)
carfit_rnd <- randomForest(f,data = train,importance=TRUE,ntree=150)

#See the imnportance of each variable
varImpPlot(carfit_rnd)

#predict using rand forest
test_pred_rnd <- predict(carfit_rnd, test[,-1])

#Calculate accuracy
med <- median(test_orig)
med <- mean(test_orig)
RMSE_rnd <- sqrt(mean((test_orig-test_pred_rnd)^2))
RMSE_rnd
finalResult$Rnd_For <- 1-RMSE_rnd/med

#####################################################################################
# http://trevorstephens.com/post/73770963794/titanic-getting-started-with-r-part-5-random
#Build a Conditional Tree Forest
library(party)
carfit_cforest <- cforest(f,data = train)

#Predict
test_pred_cforest <- predict(carfit_cforest,test[,-1],OOB = TRUE,type = "response")
test_pred_cforest
#Calculate accuracy
med <- median(test_orig)
RMSE_cforest <- sqrt(mean((test_orig-test_pred_cforest)^2))
RMSE_cforest
finalResult$RMSE_cforest <- 1-RMSE_rnd/med
finalResult

#####################################################################################
#
#Build a bagging model
install.packages("ipred")
library(ipred)
carfit_bag <- bagging(f,data = train,control=rpart.control(minsplit=5))

#Predict
test_pred_cforest <- predict(carfit_bag,OOB = TRUE)

#Calculate accuracy
med <- median(test_orig)
RMSE_cforest <- sqrt(mean((test_orig-test_pred_cforest)^2))
RMSE_cforest
finalResult$RMSE_cforest <- 1-RMSE_rnd/med
finalResult

