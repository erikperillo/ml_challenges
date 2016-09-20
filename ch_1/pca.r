printf <- function(...) cat(sprintf(...))

dims <- function(x)
{
	return(c(nrow(x), ncol(x)))
}

pca <- function(data)
{
	#printf("pca on data with shape %dx%d\nstart=%d, end=%d\n",
	#	nrow(data), ncol(data), start, end)

	pcs <- prcomp(data, scale=TRUE)

	return(pcs)
}

pca_k_pcs <- function(pcs, k=-1)
{
	if(k == -1)
		k <- ncol(pcs$rotation)	

	k_pcs <- pcs$rotation[, 1:k]
	transf_mat <- t(k_pcs)

	return(rbind(transf_mat))
}

pca_project <- function(data, k_pcs)
{
	projection <- data %*% t(k_pcs)

	return(projection)
}

pca_recover <- function(proj_data, k_pcs)
{
	recovery <- proj_data %*% k_pcs

	return(recovery)
}

new_plot <- function() x11()

pca_example <- function()
{
	#making random data points
	x <- seq(0, 100, 0.1)
	y <- 1.618*x + runif(length(x), -10, 10)
	#building nx2 matrix
	data <- cbind(x, y)

	#plotting original data
	plot(x ~ y)
	title("y = 1.618x + random")

	printf("getting principal components from data...\n")
	#getting pcs
	pcs <- prcomp(data, scale=TRUE)
	printf("done. stats:\n")
	summary(pcs)

	#getting k principal components
	k <- 1
	printf("getting %d principal components\n", k)
	transf_mat <- rbind(t(pcs$rotation[, 1:k]))
	
	#projecting data
	printf("projecting data...\n")
	proj <- data %*% t(transf_mat)
	x11()
	plot(rep(0:0, each=length(proj)) ~ proj)
	title("projected data")

	#recovering data
	printf("recovering data...\n")
	rec <- proj %*% transf_mat
	x11()
	plot(rec)
	title("recovered data")
}

close <- function() graphics.off()
