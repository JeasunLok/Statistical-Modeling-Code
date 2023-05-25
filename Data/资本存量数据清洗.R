# 资本存量数据清洗
library("readxl")
L <- read_excel("D:/大三下/统计建模大赛/data/GDP/clean资本存量78-19.xlsx")
L1 <- L[,order(names(L))]
L1 <- L1[-c(41,42),]
colnames(L1)[1] <- "年"
write.csv(L1,file = "D:/大三下/统计建模大赛/Code/data/clean资本存量78-17.csv")

