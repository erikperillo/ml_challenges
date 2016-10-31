#packages
library(caret)
library(stats)
library(fpc)

#file path of data
data_filepath <- "./cluster-data.csv"
labels_filepath <- "./cluster-data-class.csv"

#k-means parameters
ks <- seq(2, 10)
n_start <- 5

#wrapper for sprintf
printf <- function(...) cat(sprintf(...))

#main method for whole challenge
main <- function()
{
    #reading data
    x <- read.csv(data_filepath, header=TRUE, sep=",")
    x <- as.matrix(x)
    y <- read.csv(labels_filepath, header=TRUE, sep=",")
    y <- as.matrix(y[, 1])

    #best index of internal metrics and external metrics
    int_best_id <- 1
    ext_best_id <- 1

    #internal and external metrics values
    int_metrics <- c()
    ext_metrics <- c()

    #distances between points in x
    printf("getting distances object for points... ")
    dst <- dist(x)
    printf("done.\n")

    for(i in seq(length(ks)))
    {
        printf("k = %d:\n", ks[i])

        #getting k-means
        printf("\tcomputing k-means... ")
        means <- kmeans(x, ks[i], nstart=n_start)
        printf("done.\n")

        #getting stats
        printf("\tcomputing clustering stats... ")
        stats <- cluster.stats(dst, means$cluster, y)
        printf("done.\n")

        #printing metrics
        printf("\tdunn2: %.6f | corrected.rand: %.6f\n",
            stats$dunn2, stats$corrected.rand)
        printf("\n")

        #appending metric for later plotting
        int_metrics <- c(int_metrics, stats$dunn2)
        ext_metrics <- c(ext_metrics, stats$corrected.rand)

        #checking for best metric
        if(stats$dunn2 > int_metrics[int_best_id])
            int_best_id <- i
        if(stats$corrected.rand > ext_metrics[ext_best_id])
            ext_best_id <- i
    }

    #printing best scores
    printf("best internal metric score: %.6f (k=%d)\n",
        int_metrics[int_best_id], ks[int_best_id])
    printf("best external metric score: %.6f (k=%d)\n",
        ext_metrics[ext_best_id], ks[ext_best_id])

    #plotting metrics
    printf("plotting metrics...\n")
    dev.new()
    plot(int_metrics ~ ks,
         xlab="number of clusters (k)", ylab="dunn2")
    dev.new()
    plot(ext_metrics ~ ks,
         xlab="number of clusters (k)", ylab="corrected.rand")

}
