library(ggplot2)
library(tidyverse)
library(data.table)
library(dplyr)

tsv_name <- 'Joao_repository/Results/Pixel_Intensity_Thresholding/clusters_size5plus_intensity_over1000.tsv'

cluster_df <- fread(file = tsv_name, header = T)

all_columns <- colnames(cluster_df)

columns_to_read = as.character(grep("Channel", all_columns, value = TRUE))

cols_combinations_to_plot <- combn(columns_to_read, 2, simplify = FALSE)

cols_combinations_to_plot <- Filter(function(x) "Channel_2_Summed_Intensity" %in% x, cols_combinations_to_plot)

for (col_combn in cols_combinations_to_plot) {
  
  plot <- ggplot(cluster_df, aes(x = .data[[col_combn[1]]], 
                                 y = .data[[col_combn[2]]])) +
    geom_point() +
    theme_minimal()
  
  filename <- paste0(col_combn[1], "_vs_", col_combn[2], "_scatter_plot.pdf")
  
  ggsave(filename, plot, width = 10, height = 10)
}




