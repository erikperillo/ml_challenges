#packages
library(caret)
library(e1071)

#default values:
#file path of data
data_filepath <- "data1.csv"
#k for external k-fold
ext_k <- 5
#k for internal k-fold
inn_k <- 3
#cost parameters
cost_params <- c(2^-5, 2^-2, 2^0, 2^2, 2^5)
#gamma parameters
gamma_params <- c(2^-15, 2^-10, 2^-5, 2^0, 2^5)

#wrapper for sprintf
printf <- function(...) cat(sprintf(...))

#calculates accuracy of prediction
accuracy <- function(pred_y, y, thresh=0.5)
{
	pred_y <- as.numeric(pred_y >= thresh)
	return (sum(pred_y == y)/length(y))
}

#performs grid search to find best C and gamma for svm
grid_search <- function(x_train, y_train, x_test, y_test, costs, gammas)
{
	best_cost <- cost_params[1]
	best_gamma <- gamma_params[1]
	max_accuracy <- 0.0

	for(cost in costs)
	{
		for(gamma in gammas)
		{
			model <- svm(x_train, y_train, cost=cost, gamma=gamma,
				kernel="radial")

			y_pred <- predict(model, x_test)

			acc <- accuracy(y_pred, y_test)
			if(acc > max_accuracy)
			{
				max_accuracy <- acc
				best_cost <- cost
				best_gamma <- gamma
			}
		}
	}	

	return (c(best_cost, best_gamma, max_accuracy))
}

#gets best svm parameters (C, gamma) within given folds
get_best_svm_params <- function(x, y, num_folds, costs, gammas)
{
	folds <- createFolds(y, k=num_folds)

	count <- 1
	max_accuracy <- 0.0
	best_cost <- costs[1]
	best_gamma <- gammas[1]

	for(fold in folds)
	{
		printf("fold n. %d: ", count)

		x_train <- x[-fold, ]
		x_test <- x[fold, ]
		y_train <- y[-fold]
		y_test <- y[fold]

		params <- grid_search(x_train, y_train, x_test, y_test, costs, gammas)
		cost <- params[1]
		gamma <- params[2]
		acc <- params[3]

		printf("cost=%f, gamma=%f, accuracy=%.2f%% ", cost, gamma, 100*acc)
		if(acc > max_accuracy)
		{
			printf("(best so far!)")
			max_accuracy <- acc
			best_cost <- cost
			best_gamma <- gamma
		}
		printf("\n")

		count <- count + 1
	}

	printf("\t--------\n")
	printf("\tbest cost: %f, best gamma: %f, max accuracy: %.2f%%\n", 
		best_cost, best_gamma, 100*max_accuracy)

	return (max_accuracy)
}

#k-fold inside a k-fold. used to estimate accuracy of svm
mean_accuracy_estimate <- function(x, y, num_external_folds, num_inner_folds)
{
	#preparing data 
	ext_folds <- createFolds(y, k=num_external_folds)
	
	ext_count <- 1
	max_accuracies <- c()

	for(ext_fold in ext_folds)
	{
		printf("external fold n. %d:\n", ext_count)

		x_test <- x[ext_fold, ]
		y_test <- y[ext_fold]
		x_train <- x[-ext_fold, ]
		y_train <- y[-ext_fold]

		max_acc <- get_best_svm_params(x_train, y_train, inn_k, 
			cost_params, gamma_params)	
		max_accuracies <- c(max_accuracies, max_acc)

		ext_count <- ext_count + 1
		printf("\n")
	}

	printf("mean of maximum accuracies: %.2f%%\n", 100*mean(max_accuracies))
}

#main method for whole challenge
main <- function()
{
	#reading data
	data <- read.csv(data_filepath)
	x <- data[, 1:ncol(data)-1]
	y <- data[, ncol(data)]

	printf("ESTIMATING ACCURACY:\n")
	printf("\texternal k-folds: %d\n\tinner k-folds: %d\n\tcosts: ", 
		ext_k, inn_k)
	print(cost_params)
	printf("\tgammas: ")
	print(gamma_params)
	printf("\ttotal iterations: %d\n\n", 
		length(cost_params)*length(gamma_params)*ext_k*inn_k)

	#estimating accuracy for classifier
	mean_accuracy_estimate(x, y, ext_k, inn_k)

	printf("\n-------------------------------------------------------------\n")
	printf("GETTING FINAL CLASSIFIER:\n")
	printf("\tk-folds: %d\n\tcosts: ", inn_k)
	print(cost_params)
	printf("\tgammas:")
	print(gamma_params)
	printf("\ttotal iterations: %d\n\n", 
		length(cost_params)*length(gamma_params)*inn_k)

	#final classifier
	params <- get_best_svm_params(x, y, inn_k, cost_params, gamma_params)
}
