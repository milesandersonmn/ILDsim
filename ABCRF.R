install.packages("abcrf")
library(abcrf)

setwd("~/PhD/ILDsim/ILDsim/")

txt <- read.csv("summary_statistics.csv", header = FALSE)

txt <- txt[ ,-c(2,3,4)]
modindex <- as.factor(txt$V1)
sumsta <- txt[,-1]
data1 <- data.frame(modindex, sumsta)
model.rf1 <- abcrf(modindex~., data = data1, lda = TRUE)
model.rf1

#obs <- txt[c(25000,65000,125000,175000,225000), -1]
obs <- read.csv("obs.csv", header = FALSE)
obs <- obs[, -c(1,2,3,4)]

predict(model.rf1, obs, data1)

plot(model.rf1, data1)
