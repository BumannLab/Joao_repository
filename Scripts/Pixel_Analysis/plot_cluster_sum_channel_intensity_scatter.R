library(ggplot2)
library(data.table)
library(dplyr)
library(ggprism)


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


filter_rows_no_cluster <- function(dataframe, columns_to_read){
  dataframe <- dataframe |>
    filter(if_any(any_of(c("Area", "Volume")), ~ .x > 0))  
  
  return (dataframe)
}


plot_scatter <- function(dataframe, cols_combinations_to_plot, odirname){
  for (col_combn in cols_combinations_to_plot) {
    
    col1 = col_combn[1]
    col2 = col_combn[2]
    
    
    plot <- ggplot(dataframe, aes(x = .data[[col1]], 
                                   y = .data[[col2]])) +
      geom_point(size = 1) +
      theme_minimal() +
      scale_y_log10() +
      scale_x_log10() +
      xlab(gsub("_", " ", col1)) +
      ylab(gsub("_", " ", col2)) +
      theme_prism(base_size = 16)
    
    filename <- paste0(odirname, "/", 
                       rename_channel(col1, channel_map), 
                       "_vs_", 
                       col2, 
                       "_scatter_plot.png")
    
    ggsave(filename, plot, width = 12, height = 8)
  }
}


main <- function(tsv_name, channel_of_interest, odirname){
  cluster_df <- read_tsv(tsv_name)
  cols_to_plot <- obtain_columns_to_plot(cluster_df)
  combn_cols <- combinations_of_columns_for_scatter(channel_of_interest, cols_to_plot)
  cluster_df_filtered <- filter_rows_no_cluster(cluster_df, cols_to_plot)
  plot_scatter(cluster_df_filtered, combn_cols, odirname)
  
  return (cluster_df)
}

tsv_name <- "/data/joao/Materials/Joao_repository/Results/New_Destaining/Full_scan_2M/SYBRGold_CD68AL555_LAMAL647_DAPI/Voxel_Analysis/voxel_clusters_size_over5_SYBRGold_greater_equal_1000.tsv"
channel_of_interest <- "SYBRGold"
odirname <- "/data/joao/Materials/Joao_repository/Results/New_Destaining/Full_scan_2M/SYBRGold_CD68AL555_LAMAL647_DAPI/Voxel_Analysis/Plots_sybrgold_above1000"

cluster_df_filtered <- main(tsv_name, channel_of_interest, odirname)


base <- ggplot(cluster_df_filtered, aes(x = SYBRGold_Mean_Intensity, 
                              y = DAPI_Mean_Intensity)) +
  geom_point(size = 1) +
  theme_minimal() +
  scale_y_log10() +
  scale_x_log10() +
  xlab(gsub("_", " ", "SYBRGold_Mean_Intensity")) +
  ylab(gsub("_", " ", "DAPI_Mean_Intensity"))

plot <- base +
        theme_prism(base_size = 16)

plot

cluster_df_filtered_2 <- cluster_df_filtered |>
  filter(DAPI_Mean_Intensity <250)  


