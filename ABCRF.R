install.packages("abcrf")
library(abcrf)

setwd("~/PhD/ILDsim/")

summary_table <- read.csv("summary_statistics_combined.csv", header = FALSE)

txt <- summary_table[ ,-c(2:11)]
modindex <- as.factor(txt$V1)
sumsta <- txt[,-1]

header <- c( "modindex", "S", "span_S", "mean_pi", "var_pi", "pi", 
            "AFS_q0.1", "AFS_q0.3", "AFS_q0.5", 
            "AFS_q0.7", "AFS_q0.9", "mean_D", "var_D",
            "h_q0.1", "h_q0.3",
            "h_q0.5", "h_q0.7",
            "h_q0.9", "mean_h", 
            "var_h", "r2_q0.1", "r2_q0.3",
            "r2_q0.5", "r2_q0.7", "r2_q0.9",
            "mean_r2", "var_r2", "ILD_q0.1", "ILD_q0.3",
            "ILD_q0.5", "ILD_q0.7", "ILD_q0.9",
            "mean_ILD", "var_ILD")

data1 <- data.frame(modindex, sumsta)

colnames(data1) <- header

#data1 <- data1[ ,-c(2:6,14:20)]
model.rf1 <- abcrf(modindex~., data = data1, lda = FALSE)
model.rf1
quartz(height = 10)
plot(1:10,1:10)
plot(model.rf1, data1, n.var = 25)

#head(summary_table)
#obs <- txt[c(20000,43000,65000,83000,110000, 16000, 45000, 67000, 90000, 120000), -1]
obs <- read.csv("ILDsim/obs.csv", header = FALSE)
obs <- obs[, -c(1:11)]
colnames(obs) <- header[-1]
#obs <- obs[, -c(1:5, 13:19)]

predict(model.rf1, obs, data1)


