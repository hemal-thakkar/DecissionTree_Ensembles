# Script To use the following Decission Tree and Ensemble techniques for simple predictive analytics
# Input: dataSet: R data frame, Predictive analytics applied to first column using the remaining ones
# Output list('Technique'=RMSEAccuracy)

# Play with data
?mtcars
mtcars
plot(mtcars$mpg,mtcars$carb)

#Sample the data set
smp_size <- floor(0.75 * nrow(mtcars))
train_ind <- sample(seq_len(nrow(mtcars)), size = smp_size)
train <- mtcars[train_ind,]
test <- mtcars[-train_ind,]
test_orig <- test$mpg

# Generate formula
f <- as.formula(paste("mpg ~ ",paste(names(train)[-1],collapse = " + ")))
library(rpart)

#build a simple decission tree model
carfit_dt <- rpart(f, data = train, method = "anova")

#Predict using Decission Tree
test_pred_dt <- predict(carfit_dt, test[,-1])

#Calculate accuracy
med <- median(test_orig)
RMSE_dt <- sqrt(mean((test_orig-test_pred_dt)^2))
cat("One Decission True : ", (1-RMSE_dt/med))

#build a random forest model
set.seed(415)
install.packages('randomForest')
library(randomForest)
carfit_rnd <- randomForest(f,data = train,importance=TRUE,ntree=150)

#See the imnportance of each variable
varImpPlot(carfit_rnd)

#predict using rand forest
test_pred_rnd <- predict(carfit_rnd, test[,-1])

test_pred_rnd
test_orig

#Calculate accuracy
med <- median(test_orig)
med <- mean(test_orig)
RMSE_rnd <- sqrt(mean((test_orig-test_pred_rnd)^2))
RMSE_rnd
1-RMSE_rnd/med
plot(test_orig-test_pred_rnd)
names(test_orig) <- names (test_pred_rnd)

