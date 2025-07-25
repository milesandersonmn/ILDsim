install.packages("abcrf")
library(abcrf)
install.packages("pheatmap")
library(pheatmap)
install.packages("reshape2")
library(reshape2)

setwd("~/PhD/ILDsim/")

summary_table <- read.csv("summary_statistics_constant.csv", header = FALSE)
summary_table2 <- read.csv("summary_statistics_expansion_scaledLD.csv", header = FALSE)

summary_table2 <- summary_table2[,-70]

names(summary_table2) <- names(summary_table)
summary_table <- rbind(summary_table, summary_table2)
summary_table <- subset(summary_table, select = -V61)
tail(summary_table)
txt <- summary_table[ ,-c(2:11)]
modindex <- as.factor(txt$V1)
sumsta <- txt[,-1]

header <- c( "modindex", 
            "fSFS1", "fSFS2", "fSFS3", 
            "fSFS4", "fSFS5", "fSFS6", "fSFS7", "fSFS8",
            "fSFS9", "fSFS10", "fSFS11", "fSFS12", "fSFS13",
            "fSFS14", "fSFS15", "AFS_q0.1", "AFS_q0.3", "AFS_q0.5",
            "AFS_q0.7", "AFS_q0.9", "mean_D", "var_D", "std_D",
            "mean_F", "std_F", "var_F",
            "ham_q0.1", "ham_q0.3",
            "ham_q0.5", "ham_q0.7",
            "ham_q0.9", "mean_ham", "std_ham",
            "var_ham", "homo_q0.1", "homo_q0.3", "homo_q0.5",
            "homo_q0.7", "homo_q0.9", "mean_homo", "hmean_homo",
            "var_homo", "std_homo", "r2_q0.1", "r2_q0.3",
            "r2_q0.5", "r2_q0.7", "r2_q0.9",
            "mean_r2", "var_r2", "std_r2", "ILD_q0.1", "ILD_q0.3",
            "ILD_q0.5", "ILD_q0.7", "ILD_q0.9",
            "mean_ILD", "var_ILD", "std_ILD", "AndR2_q0.1",
            "AndR2_q0.3", "AndR2_q0.5", "AndR2_q0.7", "AndR2_q0.9",
            "mean_AndR2", "var_AndR2", "std_AndR2")
length(header)
data1 <- data.frame(modindex, sumsta)

colnames(data1) <- header

data1 <- data1[, -c(23,27,35,43,51,59,67)]
#data1 <- data1[ ,-c(2:6,14:20)]
model.rf1 <- abcrf(modindex~., data = data1, lda = FALSE)
model.rf1
conf_mat <- model.rf1$model.rf$confusion.matrix
err.abcrf(model.rf1, data1)

pheatmap(conf_mat[,1:5], 
         display_numbers = TRUE, 
         #color = scales::div_gradient_pal(low = "blue",
          #                                mid = "yellow",
           #                               high="red")(seq(0,1,
            #                                                 length.out = max(conf_mat))),
         cluster_rows = FALSE, 
         cluster_cols = FALSE,
         border_color = FALSE,
         number_color = "black",
        
         main = "Confusion Matrix Heatmap",
         labels_col = c("1.9","1.7","1.5","1.3","1.1"),
         labels_row = c("1.9","1.7","1.5","1.3","1.1"))


quartz(height = 10)
plot(1:10,1:10)
plot(model.rf1, data1, n.var = 17)

#head(summary_table)
#obs <- txt[c(20000,43000,65000,83000,110000, 16000, 45000, 67000, 90000, 120000), -1]

obs <- read.csv("ILDsim/obs.csv", header = FALSE)
obs <- obs[, -c(1:11)]
obs <- obs[, -c(23,27,35,43,51,59,67)]
names(obs) <- names(subset(data1, select = -modindex))
#obs <- obs[, -c(1:5, 13:19)]

predict(model.rf1, obs, data1)

ggplot(data1, aes(x = modindex, y = std_AndR2)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of std_AndR2", x = "model_index", y = "std")

ggplot(data1, aes(x = modindex, y = mean_AndR2)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of mean_AndR2", x = "model_index", y = "mean")

ggplot(data1, aes(x = modindex, y = std_r2)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of std_r2", x = "model_index", y = "std")

ggplot(data1, aes(x = modindex, y = std_ILD)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of std_ILD", x = "model_index", y = "std")

ggplot(data1, aes(x = modindex, y = mean_ILD)) +
  geom_boxplot(fill = "skyblue", outliers = FALSE) +
  scale_x_discrete(labels = c("1" = "1.9", "2" = "1.7", "3" = "1.5", "4" = "1.3", "5" = "1.1")) +
  labs(title = "Distribution of mean_ILD", x = "model_index", y = "mean")

curve(25000*exp(-0.001*x),
      from = 0, to = 100000,
      xlab = "Generations")
      