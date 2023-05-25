# 社会消费品零售总额
library("readxl")
trscg <- read_excel("D:/大三下/统计建模大赛/data/GDP/社会消费品零售总额.xlsx")
prvcnm <- levels(as.factor(trscg$Prvcnm))
prvcnm
prvcnm <- prvcnm[-31]

trscg1 <- data.frame(yr=1978:2021)
colnames(trscg1) <- "年"

for(n in 1:length(prvcnm)){
  nc <- as.data.frame(rep(NA,44))
  cadd <- as.data.frame(trscg[trscg$Prvcnm==prvcnm[n],"社会消费品零售总额"])
  m <- nrow(cadd)
  for(i in 1:m){
    nc[44-m+i,] <- cadd[i,]
  }
  colnames(nc) <- prvcnm[n]
  trscg1 <- cbind(trscg1,nc)
}

# 西藏 1997-1999
xz <- trscg1$西藏自治区
x <- c(1978:1996,2000:2021)
fitxz <- lm(log(xz[x-1977])~x)
xz[20:22] <- exp(predict(fitxz,data.frame(x=c(1997:1999))))
plot(x,log(xz[x-1977]))
plot(1978:2021,log(xz))
trscg1$西藏自治区 <- xz

# 重庆 1978-1996
chq <- trscg1$重庆市
x <- 1997:2021
plot(x,log(chq[x-1977]))
y <- 1978:1997
fitchq <- lm(log(chq[x-1977])~x)
chq[1:20] <- exp(predict(fitchq,data.frame(x=y)))
plot(1978:2021,log(chq))
trscg1$重庆市 <- chq


# 1997数据
x <- c(1978:1996,1998:2021)
ah <- trscg1$安徽省
plot(x,log(ah[x-1977]))

bj <- trscg1$北京市
plot(x,log(bj[x-1977]))

sc <- trscg1$四川省
plot(x,log(sc[x-1977]))

xj <- trscg1$新疆维吾尔自治区
plot(x,log(xj[x-1977]))

# log变换后明显呈线性
trscg2 <- trscg1[,-1]
index <- which(!(prvcnm%in%c("西藏自治区","重庆市")))
ltrscg <- log(trscg1[,-1])
x <- c(15:19,21:25)
trscg2[20,index] <- exp(colMeans(ltrscg[x,index]))
trscg2 <- cbind(data.frame("年"=1978:2021),trscg2)

#for(i in index){
#  y <- as.data.frame(ltrscg[,prvcnm[i]])
#  fit <- lm(y[x-1977,]~x)
#  y[20,] <- predict(fit,data.frame(x=1997))
#  trscg2[,prvcnm[i]] <- exp(y)
#}

plot(1978:2021,log(trscg2$福建省))
plot(1978:2021,log(trscg2$北京市))
plot(1978:2021,log(trscg2$甘肃省))
plot(1978:2021,log(trscg2$河南省))

write.csv(trscg2,file = "D:/大三下/统计建模大赛/data/GDP/cleanTRSCG78-21.csv")
