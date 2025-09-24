install.packages("readr")
install.packages("tidyverse")

library("readr")
library("ggplot2")
library("tidyverse")

fluorescent_analysis <- read_tsv("./test.tsv")

column_name = colnames(fluorescent_analysis)

important_columns = regmatches("/", column_name)

plot = ggplot(fluorescent_analysis, aes(important_columns)) +
  geom_histogram()
plot
