library(ggplot2)
library(stringr)
library(plyr)
library(dplyr)
library(lubridate)
library(reshape2)
library(scales)
library(ggthemes)
library(Metrics)

data <- read.csv("scripts/toR.csv")
cor(data$Truth,data$Preds)^2
model <- lm(data$Truth ~ data$Preds)
summary(model) # R = 0.59
summary(model)$r.squared # 0.35
summary(abs(data$Truth - data$Preds))
mean(abs((data$Truth-data$Preds)/data$Truth)) * 100 # 27.82447
