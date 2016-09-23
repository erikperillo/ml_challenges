#package for lda
library(MASS)

#default values:
#file path of data
data_filepath <- "data1.csv"
#minimum variance required
min_var <- 0.80
#number of lines to use in training
train_n_lines <- 200
#number of lines to use in test
test_n_lines <- 276

#wrapper for sprintf
printf <- function(...) cat(sprintf(...))

#gets minimum number of principal components to keep in order to conserve
#min_var of variance
pca_min_pcs <- function(pcs, min_var)
{
	#getting k minimum number of components required for minimum variance
	pcs_var_cumsum <- cumsum(pcs$sdev^2/sum(pcs$sdev^2))
	min_pcs <- which(pcs_var_cumsum >= min_var)[1]
	#printing result
	printf("\t-Number of components to keep %.2f%% variance: %d\n",
		min_var*100, min_pcs)	

	return(min_pcs)
}

#calculates logistic regression and displays accuracy
logit_reg <- function(x_train, y_train, x_test, y_test)
{
	#computing logistic regression
	lr <- glm(y_train ~ ., data=x_train, family=binomial(link="logit"))
	#getting predictions
	pred <- as.matrix(predict(lr, x_test)) >= 1
	#getting score
	score = sum(pred == y_test)/length(y_test)
	#printing accuracy
	printf("accuracy: %.2f%%\n", score*100)
}

#calculates LDA and displays accuracy
lda_reg <- function(x_train, y_train, x_test, y_test)
{
	#computing lda
	ldar <- lda(y_train ~ ., data=x_train)
	#getting predictions
	pred <- predict(ldar, x_test, prior=ldar$prior)
	pred <- pred$posterior[, 2] > pred$posterior[, 1]
	#getting score
	score = sum(pred == y_test)/length(y_test)
	#printing accuracy
	printf("accuracy: %.2f%%\n", score*100)
}

#main method for whole challenge
main <- function()
{
	#reading data
	data <- read.csv(data_filepath)
	x <- data[, 1:ncol(data)-1]

	#scaling data prior to pca. we would have to do it anyway...
	x <- scale(x)

	#getting principal components
	pcs <- prcomp(x, scale=FALSE)

	#1. Faca o PCA dos dados (sem a última coluna). 
	#Se voce quiser que os dados transformados tenham 80% da variância original,
	#quantas dimensões do PCA vc precisa manter?
	#Gere os dados transformados mantendo 80\% da variância. 
	printf("Item 1:\n")
	min_pcs <- pca_min_pcs(pcs, min_var)
	#getting transformation matrix
	transf_mat <- t(pcs$rotation[, 1:min_pcs])
	#transforming data into k dimensions while keeping percentage of variance
	transf_x <- as.matrix(x) %*% t(transf_mat)

	#preparing data for regression
	#train data
	y <- as.matrix(data[, ncol(data)])
	y_train <- y[1:train_n_lines, ]
	x_full_var_train <- x[1:train_n_lines, ]
	x_part_var_train <- transf_x[1:train_n_lines, ]
	#test data
	x_full_var_test <- x[(train_n_lines+1):nrow(x), ]
	x_part_var_test <- as.matrix(x_full_var_test) %*% t(transf_mat)
	y_test <- y[(train_n_lines+1):nrow(y), ]

	#2. Treine uma regressão logística no conjunto de treino dos dados originais
	#e nos dados transformados. 
	#Qual a taxa de acerto no conjunto de teste nas 2 condições (sem e com PCA)?
	printf("\nItem 2:\n")
	#logistic regression on all dimensions
	printf("\t-All dimensions: ")
	logit_reg(as.data.frame(x_full_var_train), y_train,
		as.data.frame(x_full_var_test), y_test)
	#logistic regression on k principal components
	printf("\t-First %d dimensions from PCA: ", min_pcs)
	logit_reg(as.data.frame(x_part_var_train), y_train,
		as.data.frame(x_part_var_test), y_test)

	#3. Treine o LDA nos conjuntos de treino com e sem PCA e teste nos 
	#respectivos conjuntos de testes. Qual a acurácia nas 2 condições?
	printf("\nItem 3:\n")
	#lda on all dimensions 
	printf("\t-All dimensions: ")
	lda_reg(as.data.frame(x_full_var_train), y_train,
		as.data.frame(x_full_var_test), y_test)
	#lda on k principal components
	printf("\t-First %d dimensions from PCA: ", min_pcs)
	lda_reg(as.data.frame(x_part_var_train), y_train,
		as.data.frame(x_part_var_test), y_test)

	printf("\n")
}
