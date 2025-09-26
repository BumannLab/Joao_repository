install.packages("readr")
install.packages("tidyverse")
install.packages("data.table")

library("readr")
library("ggplot2")
library("tidyverse")
library("data.table")

tsv_name <- 'Fluorescent_thresholding_over_2000_updated_normalized.tsv'

all_columns <- fread(file = tsv_name, header = F ,nrows = 1)

columns_to_read = as.character(grep("/", all_columns, value = TRUE))

pixel_intensities_threshold <- fread(tsv_name,
                                     sep = "\t",
                                     header = TRUE,
                                     quote = "",
                                     fill = TRUE,
                                     select = columns_to_read,
                                     showProgress = TRUE)

pixel_intensities_threshold <- pivot_longer(pixel_intensities_threshold,
                                            cols = everything(),
                                            names_to = "Ratios",
                                            values_to = "Intensity")

pixel_intensities_threshold$Intensity <- as.numeric(pixel_intensities_threshold$Intensity)

plot = ggplot(pixel_intensities_threshold, aes(x = Intensity, fill = Ratios)) +
  geom_histogram(color='#e9ecef', alpha=0.6, position='identity', binwidth = 0.06, bins = 50) +
  labs(title = "Pixel Ratios", x = "Intensity", y = "Count")  +
  xlim(0, 3)

plot = ggplot(pixel_intensities_threshold, aes(x = Intensity, fill = Ratios)) +
  geom_histogram(color='#e9ecef', alpha=0.6, position='identity', binwidth = 0.06, bins = 50) +
  labs(title = "Pixel Ratios", x = "Intensity", y = "Count")  +
  xlim(0, 3)

pdf(file="./Unet/Results/Pixel_Intensity_Thresholding/Thresholding_over_2000_intesity_ratios_xlim_0_3.pdf")
plot
dev.off()



