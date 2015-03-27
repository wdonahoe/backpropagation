random_weight <- function(n_to,n_from){
  matrix( rnorm( n_to * n_from ), n_to, n_from)
}

weight_list <- function(levels){
  weight_list <- list()
  for( i in seq( 2, length( levels) ) ){
    r <- random_weight( levels[ i ], levels[i - 1] ) 
    weight_list[[i-1]] <- r
  }
  return( weight_list )
}

bias_list <- function(levels){
  bias_list <- list()
  for( i in seq( 2, length( levels ) ) ){
    b <- random_weight( levels[ i ], 1)
    bias_list[[i-1]] <- b
  }
  return( bias_list )
}

activate <- function( x ){
  1/( 1 + exp( -x ) )
}

activate_prime <- function( x ){
  activate( x ) * (1 - activate( x ) )
}

z <- function(w,a_prior,bias){
  ( w %*% a_prior ) + bias
}

`%.%` <- function( X,Y ){ # Be explicit about Hadamard product. 
  X * Y 
}

forward_propagate <- function( x,weights_list,bias_list ){
  zs <- list()
  as <- list()
  x_list <- list()
  levels <- length( weights_list )
  
  zs[[1]] <- z( weights_list[[1]], x, bias_list[[1]] )
  as[[1]] <- activate( zs[[1]] )
  x_list[[1]] <- x
  
  if (levels != 1){
    for( i in seq(2, levels ) ){
        zs[[i]] <- z( weights_list[[i]], as.matrix( as[[i-1]] ), bias_list[[i]] )
        as[[i]] <- activate( zs[[i]] )
    }
  }
  return( list( zs=zs,as=as,x=x_list ) )
}

output_error <- function( z,r ){
  (activate(z) - r) %.% activate_prime( z )
}

output_error_one_layer <- function(z, r){
  activate(z) - r
}

hidden_error <- function( w,delta,z ){
  ( t( w ) %*% delta ) %.% activate_prime( z )
}

backward_propagate <- function( res,weight_list,bias_list,r ){
  L <- length( weight_list )
  deltas <- list()
  if (L == 1){
    deltas[[1]] <- output_error_one_layer( res$zs[[1]],r )
    return( deltas )
  }
  else{
    deltas[[L]] <- output_error( res$zs[[L]],r )
    for( i in seq( L-1,1 ) ){
      deltas[[i]] <- hidden_error( weight_list[[i+1]], deltas[[i+1]], res$zs[[i]] )
    }
  }
  
  return( deltas )
}

gradient_descent <- function( res,deltas,weight_list,bias_list,eta ){
  L <- length( weight_list )
  if (L == 1){
    weight_list[[1]] <- weight_list[[1]] - eta * deltas[[1]] %*% t(res$x[[1]])
    bias_list[[1]] <- bias_list[[1]] - eta * deltas[[1]]
  }
  else {
    for( i in seq( L,2 ) ){
      weight_list[[i]] <- weight_list[[i]] - eta * deltas[[i]] %*% t( res$as[[i-1]] )
      bias_list[[i]] <- bias_list[[i]] - eta * deltas[[i]]
    }
  }
  
  return( list( weight_list=weight_list,bias_list=bias_list ) )
  
}

mini_batch <- function( X,batch_size,n_instances ){
  take <- sample( c( rep( TRUE, batch_size ), rep(FALSE, n_instances - batch_size ) ) ) 
  return( X[,take,drop=FALSE] )                  
}

stochastic_backward_propagation <- function(X,R,mlp_dimensions,eta=1){
  
  W <- weight_list(mlp_dimensions)
  B <- bias_list(mlp_dimensions)
  n_instances <- ncol( X )
  batch_size <- ifelse( floor( n_instances / 10) > 1, floor( n_instances ), 1)
  j <- 0
  converge <- F
  
  while( j < 100000 ){
    
    #batch <- mini_batch( X,batch_size,n_instances )
    
    for( i in seq( 1,n_instances ) ){
      r <- R[, i ,drop=FALSE] 
      
      forward <- forward_propagate( as.matrix( X[,i] ), W, B)
      deltas <- backward_propagate( forward , W, B, r)
      
      backward <- gradient_descent( forward, deltas, W, B, eta)
      W <- backward$weight_list
      B <- backward$bias_list
      
    }
    j <- j + 1
  }
  return( list( W=W,B=B ) )
}


