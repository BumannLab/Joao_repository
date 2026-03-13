library(ggplot2)
library(tidyverse)
library(data.table)
library(dplyr)
library(ggprism)
library(scales)


find_and_read_tsv <- function(dirname, to_be_included){
  matched_files <- list.files(dirname, pattern = "voxel_clusters", full.names = TRUE)
  
  datalist <- list()
  count = 1
  
  for (regex in to_be_included){
    
    file_path <- grep(regex, matched_files, value = TRUE)
    
    dataframe <- fread(file = file_path, header = T) |>
      mutate(cluster_threshold = paste0(">", regex, " threshold"))
    
    datalist[[count]] <- dataframe
    
    count <- count + 1
  }
  
  dataframe <- bind_rows(datalist)
  
  return (dataframe)
}


process_dataframe <- function(dataframe){
  div_col <- intersect(c("Area", "Volume"), names(dataframe))
  
  dataframe <- dataframe |>
    filter(if_any(any_of(div_col), ~ .x > 0))
  
  return (dataframe)
}


filter_columns <- function(dataframe, filters){

  # Filters columns using a for loop.
  # Each iteration gets the channel name and threshold in filters (e.g. DAPI, 250)
  # Finds the name of the column in the data frame that match the channel name
  # (e.g. DAPI_Summed_Intensity)
  # Filters the matched col using the threshold (e.g. smaller than 250)

  for (i in seq_len(nrow(filters))) {

    channel_name <- filters$Channel[i]
    operator <- filters$Operator[i]
    threshold <- filters$Threshold[i]

    matched_cols <- grep(channel_name, names(dataframe), value = TRUE)

    if (operator == "<"){
      dataframe <- dplyr::filter(dataframe, .data[[matched_cols]] < threshold)
    }
    else if (operator == ">"){
      dataframe <- dplyr::filter(dataframe, .data[[matched_cols]] < threshold)
    }

  }

  return (dataframe)

}

create_plot <- function(dataframe, channel_of_interest, plot_title, odirfilename){
  
  coi <- grep(channel_of_interest, names(dataframe), value = TRUE)
  
  min_sybr <- log10(min(dataframe[[coi]]))
  max_sybr <- log10(max(dataframe[[coi]]))
  
  min_x_lim <- (floor(min_sybr / 0.2)) * 0.2
  max_x_lim <- (ceiling(max_sybr / 0.2)) * 0.2
  
  base <- ggplot(dataframe, aes(x = log10(.data[[coi]]), y = ifelse(after_stat(count) > 0, after_stat(count), NA))) +
    geom_histogram(binwidth = 0.05, boundary = min_x_lim, fill="grey", color="black", linewidth = 0.8) +
    coord_cartesian(xlim = c(min_x_lim, max_x_lim)) +
    scale_x_continuous(n.breaks = 10)
  
  final_plot <- base + 
    facet_wrap(~cluster_threshold, axes = "all" ) +
    ggtitle(plot_title) +
    xlab(paste0("log10(", gsub("_", "", channel_of_interest), " Mean Intensity)")) +
    ylab("Count") +
    theme_prism(
        base_fontface = "plain", 
        base_line_size = 0.7, 
        base_family = "Arial"
      ) +
    guides(y = "prism_offset_minor", x = "prism_offset_minor") + 
    theme_prism(base_size = 14)
    
  
  ggsave(odirfilename, final_plot, width = 12, height = 8)
  
}


main <- function(dirname, to_be_included, channel_of_interest, filters, plot_title, odirfilename){
  
  dataframe <- find_and_read_tsv(dirname, to_be_included)
  dataframe <- process_dataframe(dataframe)
  dataframe <- filter_columns(dataframe, filters)
  create_plot(dataframe, channel_of_interest, plot_title, odirfilename)
  
}



dirname <- "//biopz-jumbo.storage.p.unibas.ch/BZDB01$/_Members/rodrig0012/Documents/Joao_repository/Results/SYBRGold_CD68AL555_LAMAL647_DAPI/Full_scan/Voxel_Analysis"
to_be_included <- c("500", "800")
channel_of_interest <- "SYBRGold"
filters <- data.frame(Channel = c("DAPI"), 
                      Operator = c("<"),
                      Threshold = c(250))
plot_title <- "SYBRGold mean intensity distribution in clusters with DAPI mean intensity < 250"
odirfilename <- 'Joao_repository/Results/SYBRGold_CD68AL555_LAMAL647_DAPI/Full_scan/Voxel_Analysis/Plots/SYBRGold_cluster_mean_intensity_lt250_DAPI.pdf'

main(dirname, to_be_included, channel_of_interest, filters, plot_title, odirfilename)

# b <- ggplot_build(plot)
# print(b$data[[1]], max = 2000)