# 城镇化率
library("readxl")
cit <- read_excel("D:/大三下/统计建模大赛/Code/data/城镇化率.xls")
cit1 <- cit[,order(names(cit))]
colnames(cit1)[1] <- "年"
prvcnm <- colnames(cit1)[2:32]

# 重庆 1996
t <- 1978:2017
chq <- cit1$重庆
x <- 1996:2017
plot(x,log(chq[x-1977]))
fitchq <- lm(log(chq[x-1977])~x)
y <- 1979:1995
chq[y-1977] <- exp(predict(fitchq,data.frame(x=y)))
plot(t,log(chq))
cit1$重庆 <- chq

# 全体 1978
x <- 1979:1999
y = data.frame(x=1978)
cit2 <- cit1
for(i in 2:32){
  z <- as.data.frame(cit1[,i])
  fit <- lm(z[x-1977,]~x)
  z[1,] <- predict(fit,y)
  cit2[,i] <- z
}

plot(t,cit2$安徽)
plot(t,cit2$北京)
plot(t,cit2$福建)
plot(t,cit2$甘肃)
plot(t,cit2$广东)
plot(t,cit2$广西)
plot(t,cit2$贵州)
plot(t,cit2$海南)
plot(t,cit2$河北)

write.csv(cit2,file = "D:/大三下/统计建模大赛/Code/data/clean城镇化率78-17.csv")
