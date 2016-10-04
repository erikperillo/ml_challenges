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

accuracy <- function(pred_y, y, thresh=0.5)
{
	pred_y <- as.numeric(pred_y >= thresh)
	return (sum(pred_y == y)/length(y))
}

grid_search <- function(x_train, y_train, x_test, y_test, costs, gammas)
{
	best_cost <- cost_params[1]
	best_gamma <- gamma_params[1]
	max_accuracy <- 0.0

	for(cost in costs)
	{
		for(gamma in gammas)
		{
			#printf("trying for C=%f, gamma=%f...\n", cost, gamma)
			model <- svm(x_train, y_train, cost=cost, gamma=gamma,
				kernel="radial")

			y_pred <- predict(model, x_test)

			acc <- accuracy(y_pred, y_test)
			#printf("\taccuracy=%f\n", acc)
			if(acc > max_accuracy)
			{
				#printf("\tBEST!\n")
				max_accuracy <- acc
				best_cost <- cost
				best_gamma <- gamma
			}
		}
	}	

	return (c(best_cost, best_gamma, max_accuracy))
}

#main method for whole challenge
main <- function()
{
	#reading data
	data <- read.csv(data_filepath)
	x <- data[, 1:ncol(data)-1]
	y <- data[, ncol(data)]

	#preparing data 
	ext_folds <- createFolds(y, k=ext_k)
	
	ext_count <- 1
	max_accuracies <- c()
	for(ext_fold in ext_folds)
	{
		x_test <- x[ext_fold, ]
		y_test <- y[ext_fold]
		x_train <- x[-ext_fold, ]
		y_train <- y[-ext_fold]

		inn_folds <- createFolds(y_train, k=inn_k)

		inn_count <- 1
		max_accuracy <- 0.0
		for(inn_fold in inn_folds)
		{
			best_cost <- cost_params[1]
			best_gamma <- gamma_params[1]

			printf("ext_k=%d, inn_k=%d...\n", ext_count, inn_count)

			params <- grid_search(x_train[-inn_fold, ], y_train[-inn_fold],
				x_train[inn_fold, ], y_train[inn_fold], 
				cost_params, gamma_params)

			cost <- params[1]
			gamma <- params[2]
			acc <- params[3]
			printf("\tcost=%f, gamma=%f, acc=%f\n", cost, gamma, acc)
			if(acc > max_accuracy)
			{
				printf("\tbest!\n")
				max_accuracy <- acc
				best_cost <- cost
				best_gamma <- gamma
			}

			inn_count <- inn_count + 1
		}
		printf("\n")

		max_accuracies <- c(max_accuracies, max_accuracy)

		ext_count <- ext_count + 1
	}

	printf("max accuracies:\n")
	print(max_accuracies)
	printf("mean:\n")
	print(mean(max_accuracies))
}
