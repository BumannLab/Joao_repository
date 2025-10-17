library(data.table)
library(ggplot2)
library(tidyverse)

tsv_name <- "Joao_repository/test.tsv"
channel_map <- data.frame(
  Channel_1 = "SYBR_GOLD",
  Channel_2 = "DAPI",
  stringsAsFactors = FALSE
)

odirname <- "Joao_repository/Results/voxel_analysis/New_mTB_Data_4Channel_MAX/Images"

rename_channel <- function(old_channel_name, channel_map){
  channel_match <- regmatches(tolower(old_channel_name), regexpr("channel_[0-9]+", tolower(old_channel_name)))
  
  col_index <- which(tolower(names(channel_map)) == channel_match)
  
  replacement <- channel_map[[col_index]]
  
  new_name <- gsub(channel_match, replacement, old_channel_name, ignore.case = TRUE)
  
  return(new_name)
}


voxel_df <- fread(file = tsv_name,
                  header = T)

col_names <- colnames(voxel_df)

plot <- ggplot(voxel_df, aes(x = .data[[col_names[1]]], y = .data[[col_names[2]]])) +
  geom_point(size = 1) +
  theme_minimal() +
  scale_y_log10(breaks = c(1000, 10000)) +
  scale_x_log10() +
  xlab(rename_channel(col_names[1], channel_map)) +
  ylab(rename_channel(col_names[2], channel_map))

filename <- paste0(odirname,
                  rename_channel(col_names[1], channel_map), 
                   "_vs_", 
                   rename_channel(col_names[2], channel_map), 
                   "_2ch_7img_voxel_scatter.png")

ggsave(filename, plot, width = 10, height = 10)

