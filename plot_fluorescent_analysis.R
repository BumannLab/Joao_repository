install.packages("readr")
install.packages("tidyverse")
install.packages("data.table")

library("readr")
library("ggplot2")
library("tidyverse")
library("data.table")

tsv_name <- 'Fluorescent_thresholding_over_2000_updated_normalized.tsv' #"./test.tsv" 

all_columns <- fread(file = tsv_name, header = F ,nrows = 1)

columns_to_read = as.character(grep("/", all_columns, value = TRUE))
columns_to_read

# For some reason it refuses to read the full file, TODO fix
pixel_intensities_threshold <- fread(tsv_name, select = columns_to_read, showProgress = TRUE, header = TRUE, nrows=10000000)

pixel_long <- pivot_longer(pixel_intensities_threshold,
                           cols = everything(),
                           names_to = "Channel",
                           values_to = "Intensity")

pixel_long$Intensity <- as.numeric(pixel_long$Intensity)

plot = ggplot(pixel_long, aes(x = Intensity, fill = Channel)) +
  geom_histogram(color='#e9ecef', alpha=0.6, position='identity', binwidth = 0.1, bins = 20) +
  labs(title = "Pixel Intensity Histograms", x = "Intensity", y = "Count")
plot

