library(readr)

# Load the training dataset
train <- read_csv(
  "/app/data/train.csv",
  show_col_types = FALSE
)

# Print information to verify that the script runs
cat("Training data loaded successfully in R!\n")
cat("Number of rows:", nrow(train), "\n")
cat("Number of columns:", ncol(train), "\n")

print(head(train))

# 14.Explore the data
cat("\n--- Data structure ---\n")
str(train)

cat("\n--- Missing values by column ---\n")
print(colSums(is.na(train)))

cat("\n--- Data summary ---\n")
print(summary(train))

cat("\n--- Survival rate by sex ---\n")
print(aggregate(Survived ~ Sex, data = train, FUN = mean))

cat("\n--- Survival rate by passenger class ---\n")
print(aggregate(Survived ~ Pclass, data = train, FUN = mean))

# Create a new feature
train$FamilySize <- train$SibSp + train$Parch + 1

# Select features and target
feature_columns <- c(
  "Pclass",
  "Sex",
  "Age",
  "Fare",
  "Embarked",
  "FamilySize"
)

model_data <- as.data.frame(
  train[, c("Survived", feature_columns)]
)

cat("\n--- Data preparation decisions ---\n")
cat("Added FamilySize = SibSp + Parch + 1.\n")
cat("Selected features:", paste(feature_columns, collapse = ", "), "\n")
cat("Selected target: Survived.\n")
cat("Excluded PassengerId, Name, Ticket, and Cabin from this baseline model.\n")
cat("Missing values and categorical variables will be handled before training.\n")

cat("\n--- Selected model data: first six rows ---\n")
print(head(model_data))

cat("\nNumber of passengers:", nrow(model_data), "\n")
cat("Number of predictors:", length(feature_columns), "\n")


# 15.Train logistic regression
cat("\n--- Data preprocessing ---\n")

# Calculate numerical fill values from the training data
age_median <- median(model_data$Age, na.rm = TRUE)
fare_median <- median(model_data$Fare, na.rm = TRUE)

# Calculate the most common embarkation port
embarked_counts <- table(model_data$Embarked)
embarked_mode <- names(embarked_counts)[which.max(embarked_counts)]

# Fill missing values
model_data$Age[is.na(model_data$Age)] <- age_median
model_data$Fare[is.na(model_data$Fare)] <- fare_median
model_data$Embarked[is.na(model_data$Embarked)] <- embarked_mode

cat("Filled missing Age values with:", age_median, "\n")
cat("Filled missing Fare values with:", fare_median, "\n")
cat("Filled missing Embarked values with:", embarked_mode, "\n")

# Treat passenger class, sex, and port as categorical variables
model_data$Pclass <- factor(model_data$Pclass)
model_data$Sex <- factor(model_data$Sex)
model_data$Embarked <- factor(model_data$Embarked)

cat("Converted Pclass, Sex, and Embarked to factors.\n")
cat("R will encode these factors when fitting the model.\n")

cat("\n--- Remaining missing values in model data ---\n")
print(colSums(is.na(model_data)))

# Train logistic regression using all labeled training records
model <- glm(
  Survived ~ Pclass + Sex + Age + Fare + Embarked + FamilySize,
  data = model_data,
  family = binomial(link = "logit"),
  na.action = na.fail
)

cat("\n--- Logistic regression ---\n")
cat("Model training completed.\n")
cat("Number of training passengers:", nobs(model), "\n")
cat("Model converged:", model$converged, "\n")

print(summary(model))


# 16.Measure accuracy on the training set
train_probabilities <- predict(
  model,
  newdata = model_data,
  type = "response"
)

# Convert survival probabilities into predicted classes
train_predictions <- as.integer(train_probabilities >= 0.5)

# Compare predictions with the true labels
train_accuracy <- mean(
  train_predictions == model_data$Survived
)

cat("\n--- Training accuracy ---\n")
cat("Classification threshold: 0.5\n")
cat(sprintf("Training accuracy: %.2f%%\n", train_accuracy * 100))


# 17.Load the test dataset and make predictions
test <- read_csv("/app/data/test.csv", show_col_types = FALSE)

cat("\n--- Test data preparation ---\n")
cat("Loaded test.csv. Number of passengers:", nrow(test), "\n")

# Create the same feature used in training
test$FamilySize <- test$SibSp + test$Parch + 1
cat("Added FamilySize = SibSp + Parch + 1.\n")

# Select the same predictors used in training
test_data <- as.data.frame(test[, feature_columns])
cat("Selected features:", paste(feature_columns, collapse = ", "), "\n")

# Fill missing values using values calculated from the training set
test_data$Age[is.na(test_data$Age)] <- age_median
test_data$Fare[is.na(test_data$Fare)] <- fare_median
test_data$Embarked[is.na(test_data$Embarked)] <- embarked_mode

cat("Filled missing Age values with training median:", age_median, "\n")
cat("Filled missing Fare values with training median:", fare_median, "\n")
cat("Filled missing Embarked values with training mode:", embarked_mode, "\n")

# Match categorical levels to those used in training
test_data$Pclass <- factor(
  test_data$Pclass,
  levels = levels(model_data$Pclass)
)

test_data$Sex <- factor(test_data$Sex, levels = levels(model_data$Sex))

test_data$Embarked <- factor(test_data$Embarked, levels = levels(model_data$Embarked))

cat("Matched categorical levels to the training data.\n")

# Verify that all selected predictors are available
stopifnot(!anyNA(test_data))

# Predict survival probabilities
test_probabilities <- predict(model, newdata = test_data, type = "response")

# Convert probabilities to predicted classes
test_predictions <- as.integer(test_probabilities >= 0.5)

prediction_results <- data.frame(
  PassengerId = test$PassengerId,
  Survived = test_predictions
)

cat("\n--- Test set predictions ---\n")
cat("Classification threshold: 0.5\n")
cat("0 = predicted not survived; 1 = predicted survived.\n")
print(head(prediction_results, 10))

cat("\n--- Predicted outcome counts ---\n")
print(table(prediction_results$Survived))

cat("\nNumber of predictions:", nrow(prediction_results), "\n")


# 18.(Since there are no labels in the test set, we will skip this step.)
cat("\n--- Test set accuracy ---\n")
cat("Test accuracy is unavailable since test.csv doesn't have Survived labels.\n")