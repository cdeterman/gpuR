
# This is simply a reproduction of dim_desc from dplyr
# I like the formatted output but it would be silly to 
# require the user to import the package only for this function
dim_desc <- function(x) {
    d <- dim(x)
    d2 <- format(d, big.mark = ",", justify = "none", trim = TRUE)
    d2[is.na(d)] <- "??"
    
    paste0("[", paste0(d2, collapse = " x "), "]")
}



