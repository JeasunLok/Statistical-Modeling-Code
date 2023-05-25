# 发电量
library("readxl")
ele <- read_excel("D:/大三下/统计建模大赛/Code/data/clean发电量78-17.xls")
ele1 <- ele[,-1]
prvcnm <- colnames(ele1)[2:32]

# 重庆1997
chq <- ele1$重庆市
x <- 1997:2017
plot(x,chq[x-1977])
xx <- 1997:2007
plot(xx,log(chq[xx-1977]))

plot(1978:2017,log(ele1$安徽省))
plot(1978:2017,log(ele1$北京市))
