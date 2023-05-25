# 商品零售价格指数(上年＝100)
library("readxl")
rpi <- read_excel("D:/大三下/统计建模大赛/data/GDP/商品零售价格指数.xlsx")
prvcnm <- levels(as.factor(rpi$Prvcnm))
prvcnm
# prvcnm <- prvcnm[-31]

rpi1 <- data.frame(yr=1978:2017)
colnames(rpi1) <- "年"

for(n in 1:length(prvcnm)){
  nc <- as.data.frame(rep(0,40))
  cadd <- as.data.frame(rpi[rpi$Prvcnm==prvcnm[n],'Pi0104'])
  m <- nrow(cadd)
  for(i in 1:m){
    nc[40-m+i,] <- cadd[i,]
  }
  colnames(nc) <- prvcnm[n]
  rpi1 <- cbind(rpi1,nc)
}

# 2016数据
rpi1[rpi1[,"年"]==2016,] <- (rpi1[rpi1[,"年"]==2015,]+rpi1[rpi1[,"年"]==2017,])/2

# 海南1979
hn <- rpi1$海南省
x <- 1979:2017
plot(x,hn[x-1977])
hn[1] <- hn[2]
rpi1$海南省 <- hn

# 西藏 1994
x <- 1994:2017
xz <- rpi1$西藏自治区
xz[1:16] <- mean(xz[x-1977])
plot(1978:2017,xz)
rpi1$西藏自治区 <- xz

# 重庆1997
x <- 1997:2017
chq <- rpi1$重庆市
plot(x,chq[x-1977])
chq[1:19] <- mean(chq[x-1977])
plot(1978:2017,chq)
rpi1$重庆市 <- chq

write.csv(rpi1,file="D:/大三下/统计建模大赛/data/GDP/cleanRPI78-17.csv")
