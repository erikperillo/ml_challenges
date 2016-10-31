#packages
library(caret)
library(e1071)
library(class)
library(nnet)
library(randomForest)

#file path of data
data_filepath <- "~/random/secom/secom.data"
labels_filepath <- "~/random/secom/secom_labels.data"

#k for external k-fold
ext_k <- 5
#k for internal k-fold
inn_k <- 3

#knn parameters
knn_ks <- c(1, 5, 11, 15, 21, 25)

#svm parameters
svm_costs <- c(2^-5, 2^0, 2^5, 2^10)
svm_gammas <- c(2^-15, 2^-10, 2^-5, 2^0, 2^5)

#neural net parameters
nn_hidden_layer_sizes <- c(10, 20, 30, 40)

#random forest parameters
rf_nums_features <- c(10, 15, 20, 25)
rf_nums_trees <- c(100, 200, 300, 400, 500)

#gradient boosting machine
gbm_nums_trees <- c(30, 70, 100)
gbm_learning_rates <- c(0.1, 0.05)
gbm_depth <- 5

#wrapper for sprintf
printf <- function(...) cat(sprintf(...))

pre_proc <- function(x, y, empty_ratio_thr=0.3, scale=TRUE)
{
    #filtering out columns with more holes than specified as limit
    empty_num_thr <- floor(empty_ratio_thr*nrow(x))
    col_filter <- apply(x, 2,
        function(col) {sum(is.na(col)) <= empty_num_thr})
    new_x <- x[, col_filter]

    #filtering out rows with holes
    row_filter <- apply(new_x, 1,
        function(row) {!any(is.na(row))})
    new_x <- new_x[row_filter, ]

    new_y <- as.matrix(y[row_filter, 1])

    #scaling x
    if(scale)
    {
        new_x <- scale(new_x, center=TRUE, scale=TRUE)
        #calling pre_proc again to filter out nans since scale produces nans
        ret <- pre_proc(new_x, new_y, empty_ratio_thr, FALSE)
        new_x <- ret$x
        new_y <- ret$y
    }

    return(list("x"=new_x, "y"=new_y))
}

#gets minimum number of principal components to keep in order to conserve
#min_var of variance
pca_min_pcs <- function(pcs, min_var)
{
	#getting k minimum number of components required for minimum variance
	pcs_var_cumsum <- cumsum(pcs$sdev^2/sum(pcs$sdev^2))
	min_pcs <- which(pcs_var_cumsum >= min_var)[1]
	#printing result
	#printf("\t-Number of components to keep %.2f%% variance: %d\n",
	#	min_var*100, min_pcs)

	return(min_pcs)
}

knn_best_params <- function(x_train, y_train, pca_min_var=0.8, ks=knn_ks,
    cross=inn_k)
{
	#getting principal components maintaining minimum variance percentage
	pcs <- prcomp(x_train, scale=FALSE)
    min_pcs <- pca_min_pcs(pcs, pca_min_var)
	#getting transformation matrix
	transf_mat <- t(pcs$rotation[, 1:min_pcs])

	#transforming data into k dimensions while keeping percentage of variance
	x_train <- as.matrix(x_train) %*% t(transf_mat)

    #performing grid search to find best k
    control <- tune.control(sampling=c("cross"), cross=cross)
    tuning <- tune.knn(x_train, y_train, k=ks, tunecontrol=control)

    return(tuning$best.parameters)
}

knn_accuracy <- function(x_train, y_train, x_test, y_test, k)
{
    #getting prediction
    pred <- knn(x_train, x_test, y_train, k)

    #getting accuracy
    acc <- sum(pred == y_test)/length(y_test)

    return(acc)
}

svm_best_params <- function(x_train, y_train, costs=svm_costs,
    gammas=svm_gammas, cross=inn_k)
{
    #converting to numeric values
    y_train[y_train] = 1
    y_train[!y_train] = 0

    #getting best parameters
    control <- tune.control(sampling=c("cross"), cross=cross)
    tuning <- tune.svm(x_train, y_train, cost=costs, gamma=gammas,
       tunecontrol=control)

    return(tuning$best.parameters)
}

svm_accuracy <- function(x_train, y_train, x_test, y_test, cost, gamma)
{
    #converting to numeric values
    y_train[y_train] = 1
    y_train[!y_train] = 0
    y_test[y_test] = 1
    y_test[!y_test] = 0

    #getting model
    model <- svm(x_train, y_train, scale=FALSE, cost=cost, gamma=gamma)

    #getting prediction
    pred <- predict(model, x_test, probability=TRUE)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    #getting accuracy
    acc <- sum(pred == y_test)/length(y_test)

    return(acc)
}

nn_best_params <- function(x_train, y_train,
    hidden_layer_sizes=nn_hidden_layer_sizes, cross=inn_k)
{
    #converting to numeric values
    y_train[y_train] = 1
    y_train[!y_train] = 0

    #getting best parameters
    control <- tune.control(sampling=c("cross"), cross=cross)
    tuning <- tune.nnet(x_train, y_train, size=hidden_layer_sizes,
       tunecontrol=control, MaxNWts=20000)

    return(tuning$best.parameters)
}

nn_accuracy <- function(x_train, y_train, x_test, y_test, size)
{
    #converting to numeric values
    y_train[y_train] = 1
    y_train[!y_train] = 0
    y_test[y_test] = 1
    y_test[!y_test] = 0

    #getting model
    model <- nnet(x_train, y_train, size=size, MaxNWts=20000)

    #getting prediction
    pred <- predict(model, x_test)

    #getting accuracy
    acc <- sum(pred == y_test)/length(y_test)

    return(acc)
}

rf_best_params <- function(x_train, y_train, mtry=rf_nums_features,
    ntree=rf_nums_trees, cross=inn_k)
{
    #converting to categorical values
    y_train <- as.factor(y_train)

    #getting best parameters
    control <- tune.control(sampling=c("cross"), cross=cross)
    tuning <- tune.randomForest(x_train, y_train, mtry=mtry, ntree=ntree,
       tunecontrol=control)

    return(tuning$best.parameters)
}

rf_accuracy <- function(x_train, y_train, x_test, y_test, mtry, ntree)
{
    #converting to categorical values
    y_train <- as.factor(y_train)
    y_test <- as.factor(y_test)

    #getting model
    model <- randomForest(x_train, y=y_train, xtest=x_test, ytest=y_test,
        mtry=mtry, ntree=ntree)

    #getting accuracy
    conf <- ((model$confusion))
    acc <- (conf[1, 1] + conf[2, 2])/sum(conf)

    return(acc)
}

gbm_best_params <- function(x_train, y_train, depth=gbm_depth,
    n_trees=gbm_nums_trees, shrinkage=gbm_learning_rates, cross=inn_k)
{
    #converting to numeric values
    y_train <- as.factor(y_train)

    #getting best parameters
    fitControl <- trainControl(method="repeatedcv", number=cross, repeats=1)
    gbmGrid <- expand.grid(interaction.depth=depth, n.trees=n_trees,
        shrinkage=shrinkage, n.minobsinnode=c(10))
    tuning <- train(x_train, y_train, method = "gbm", trControl = fitControl,
        verbose = FALSE, tuneGrid = gbmGrid)

    return(tuning$bestTune)
}

gbm_accuracy <- function(x_train, y_train, x_test, y_test,
            n_trees, depth, shrinkage, minobs)
{
    #converting to numeric values
    y_train[y_train] <- 1
    y_train[!y_train] <- 0

    #getting model
    model <- gbm.fit(x_train, y_train, n.trees=n_trees, interaction.depth=depth,
        shrinkage=shrinkage, n.minobsinnode=minobs, distribution="adaboost",
        verbose=FALSE)

    #getting prediction
    pred <- predict(model, x_test, n.trees=n_trees, probability=TRUE)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    #getting accuracy
    acc <- sum(pred == y_test)/length(y_test)

    return(acc)
}

#main method for whole challenge
main <- function()
{
    #reading data
    x <- read.csv(data_filepath, header=FALSE, sep=" ")
    x <- as.matrix(x)
    y <- read.csv(labels_filepath, header=FALSE, sep=" ")
    #transforming y into logical matrix
    y <- y == 1
    y <- as.matrix(y[, 1])

    #pre-processing data
    ret <- pre_proc(x, y)
    x <- ret$x
    y <- ret$y

    knn_best_acc <- 0
    svm_best_acc <- 0
    nn_best_acc <- 0
    rf_best_acc <- 0
    gbm_best_acc <- 0

    folds <- createFolds(y, k=ext_k)
    i <- 0
    for(fold in folds)
    {
        printf("on fold n. %d...\n", i)

        #getting train/test folds
        x_train <- x[-fold, ]
        y_train <- as.matrix(y[-fold, ])
        x_test <- x[fold, ]
        y_test <- as.matrix(y[fold, ])

        #KNN
        printf("--- on KNN ---\n")
        printf("\tselecting parameters... ")
        knn_best <- knn_best_params(x_train, y_train)
        knn_k <- knn_best$k
        printf("done. k=%d\n", knn_k)
        printf("\tgetting accuracy... ")
        knn_acc <- knn_accuracy(x_train, y_train, x_test, y_test, knn_k)
        printf("done. accuracy=%.6f", knn_acc)
        if(knn_acc > knn_best_acc)
        {
            printf(" (best so far!)")
            knn_best_acc <- knn_acc
        }
        printf("\n")

        #SVM
        printf("--- on SVM ---\n")
        printf("\tselecting parameters... ")
        svm_best <- svm_best_params(x_train, y_train)
        svm_cost <- svm_best$cost
        svm_gamma <- svm_best$gamma
        printf("done. cost=%.6f, gamma=%.6f\n", svm_cost, svm_gamma)
        printf("\tgetting accuracy... ")
        svm_acc <- svm_accuracy(x_train, y_train, x_test, y_test,
            svm_cost, svm_gamma)
        printf("done. accuracy=%.6f", svm_acc)
        if(svm_acc > svm_best_acc)
        {
            printf(" (best so far!)")
            svm_best_acc <- svm_acc
        }
        printf("\n")

        #NEURAL NETWORK
        printf("--- on NEURAL NETWORK ---\n")
        printf("\tselecting parameters... ")
        nn_best <- nn_best_params(x_train, y_train)
        nn_size <- nn_best$size
        printf("done. size=%d\n", nn_size)
        printf("\tgetting accuracy... ")
        nn_acc <- nn_accuracy(x_train, y_train, x_test, y_test, nn_size)
        printf("done. accuracy=%.6f", nn_acc)
        if(nn_acc > nn_best_acc)
        {
            printf(" (best so far!)")
            nn_best_acc <- nn_acc
        }
        printf("\n")

        #RANDOM FOREST
        printf("--- on RANDOM FOREST ---\n")
        printf("\tselecting parameters... ")
        rf_best <- rf_best_params(x_train, y_train)
        rf_mtry <- rf_best$mtry
        rf_ntree <- rf_best$ntree
        printf("done. mtry=%d, ntree=%d\n", rf_mtry, rf_ntree)
        printf("\tgetting accuracy... ")
        rf_acc <- rf_accuracy(x_train, y_train, x_test, y_test,
            rf_mtry, rf_ntree)
        printf("done. accuracy=%.6f", rf_acc)
        if(rf_acc > rf_best_acc)
        {
            printf(" (best so far!)")
            rf_best_acc <- rf_acc
        }
        printf("\n")

        #GBM
        printf("--- on GBM ---\n")
        printf("\tselecting parameters... ")
        gbm_best <- gbm_best_params(x_train, y_train)
        gbm_n_trees <- gbm_best$n.trees
        gbm_depth <- gbm_best$interaction.depth
        gbm_shrinkage <- gbm_best$shrinkage
        gbm_minobs <- gbm_best$n.minobsinnode
        printf("done.\n")
        print(gbm_best)
        printf("\tgetting accuracy... ")
        gbm_acc <- gbm_accuracy(x_train, y_train, x_test, y_test,
            gbm_n_trees, gbm_depth, gbm_shrinkage, gbm_minobs)
        printf("done. accuracy=%.6f", gbm_acc)
        if(gbm_acc > gbm_best_acc)
        {
            printf(" (best so far!)")
            gbm_best_acc <- gbm_acc
        }
        printf("\n")

        i <- i + 1
        printf("\n")
    }

    printf("FINAL RESULTS:\n")
    printf("\tknn best accuracy: %.6f\n", knn_best_acc)
    printf("\tsvm best accuracy: %.6f\n", svm_best_acc)
    printf("\tnn best accuracy: %.6f\n", nn_best_acc)
    printf("\trf best accuracy: %.6f\n", rf_best_acc)
    printf("\tgbm best accuracy: %.6f\n", gbm_best_acc)
}
