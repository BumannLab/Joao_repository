library(ggplot2)
library(tidyverse)
library(data.table)
library(dplyr)


read_tsv <- function(tsv_name){
  df_tsv <- fread(file = tsv_name, header = T)
  return (df_tsv)
}


obtain_columns_to_plot <- function(dataframe) {
  
  all_columns <- colnames(dataframe)
  
  columns_to_read <- as.character(grep("Intensity", all_columns, value = TRUE))
  
  return (columns_to_read)
}


combinations_of_columns_for_scatter <- function(channel_of_interest, columns_to_read){
  cols_combinations_to_plot <- list()
  
  for (i in seq_along(columns_to_read)) {
    col_name <- columns_to_read[i]
    
    col_of_interest <- as.character(grep(channel_of_interest, columns_to_read, value = TRUE))
    
    if (col_name != col_of_interest) {
      cols_combinations_to_plot[[length(cols_combinations_to_plot) + 1]] <- c(col_of_interest, col_name)
    }
  }
  return (cols_combinations_to_plot)
}


process_dataframe <- function(dataframe, columns_to_read){
  dataframe <- dataframe |>
    mutate(across(all_of(columns_to_read), ~ .x / Area))
  
  return (dataframe)
}


plot_scatter <- function(dataframe, cols_combinations_to_plot, channel_map, odirname){
  for (col_combn in cols_combinations_to_plot) {
    
    col1 = col_combn[1]
    col2 = col_combn[2]
    
    
    plot <- ggplot(dataframe, aes(x = .data[[col1]], 
                                   y = .data[[col2]])) +
      geom_point(size = 1) +
      theme_minimal() +
      scale_y_log10() +
      scale_x_log10() +
      xlab(col1) +
      ylab(col2)
    
    filename <- paste0(odirname, "/", 
                       col1, 
                       "_vs_", 
                       col2, 
                       "_scatter_plot_test.png")
    
    ggsave(filename, plot, width = 10, height = 10)
  }
}


main <- function(tsv_name, channel_of_interest, channel_map, odirname){
  cluster_df <- read_tsv(tsv_name)
  cols_to_plot <- obtain_columns_to_plot(cluster_df)
  combn_cols <- combinations_of_columns_for_scatter(channel_of_interest, cols_to_plot)
  cluster_df <- process_dataframe(cluster_df, cols_to_plot)
  plot_scatter(cluster_df, combn_cols, channel_map, odirname)
}

tsv_name <- 'Joao_repository/New_Results/CD68AL555_SYBRGold_DAPI_CD68AL647/Pixel_Analysis/clusters_size5plus_intensity_over1000.tsv'
channel_of_interest <- "SYBRGold"
odirname <- 'Joao_repository/New_Results/CD68AL555_SYBRGold_DAPI_CD68AL647/Pixel_Analysis/Plots'

main(tsv_name, channel_of_interest, channel_map, odirname)

