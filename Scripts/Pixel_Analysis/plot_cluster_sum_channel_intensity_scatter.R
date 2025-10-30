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
  
  columns_to_read <- as.character(grep("Channel", all_columns, value = TRUE))
  
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


rename_channel <- function(old_channel_name, channel_map){
  channel_match <- regmatches(tolower(old_channel_name), regexpr("channel_[0-9]+", tolower(old_channel_name)))
  
  col_index <- which(tolower(names(channel_map)) == channel_match)
  
  replacement <- channel_map[[col_index]]
  
  new_name <- gsub(channel_match, replacement, old_channel_name, ignore.case = TRUE)
  
  return(new_name)
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
      xlab(rename_channel(col1, channel_map)) +
      ylab(rename_channel(col2, channel_map))
    
    filename <- paste0(odirname, "/", 
                       rename_channel(col1, channel_map), 
                       "_vs_", 
                       rename_channel(col2, channel_map), 
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

tsv_name <- 'Joao_repository/Results/Pixel_Intensity_Thresholding/New_mTB_Data_4Channel_MAX/clusters_size5plus_intensity_over1000.tsv'
channel_of_interest <- 2
channel_map <- data.frame(
  Channel_1 = "AL-555",
  Channel_2 = "SYBR_GOLD",
  Channel_3 = "DAPI",
  Channel_4 = "AL-647",
  stringsAsFactors = FALSE
)
odirname <- 'Joao_repository/Results/Pixel_Intensity_Thresholding/New_mTB_Data_4Channel_MAX/Images'

main(tsv_name, channel_of_interest, channel_map, odirname)

