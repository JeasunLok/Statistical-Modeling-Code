# 乡村恩格尔系数
library("readxl")
eng <- read_excel("D:/大三下/统计建模大赛/Code/data/乡村恩格尔系数.xls")
eng1 <- eng[,order(names(eng))]
colnames(eng1)[1] <- "年"
prvcnm <- colnames(eng1)[2:32]

t <- 1978:2017

# 福建 1985
hn <- eng1$福建
x <- 1985:2017
plot(x,hn[x-1977])
fithn <- lm(hn[x-1977]~x)
y <- 1979:1984
hn[2:7] <- predict(fithn,data.frame(x=y))
eng1$福建 <- hn

# 广西 1980
gs <- eng1$广西
x <- 1980:2017
plot(x,gs[x-1977])
y <- 1979
fitgs <- lm(gs[x-1977]~x)
gs[2] <- predict(fitgs,data.frame(x=y))
eng1$广西 <- gs

# 海南 1985
hn <- eng1$海南
x <- 1985:2017
plot(x,hn[x-1977])
fithn <- lm(hn[x-1977]~x)
y <- 1979:1984
hn[2:7] <- predict(fithn,data.frame(x=y))
plot(t,hn)
eng1$海南 <- hn

# 内蒙古 1980
gs <- eng1$内蒙古
x <- 1980:2017
plot(x,gs[x-1977])
y <- 1979
fitgs <- lm(gs[x-1977]~x)
gs[2] <- predict(fitgs,data.frame(x=y))
eng1$内蒙古 <- gs

# 宁夏 1983
nx <- eng1$宁夏
x <- 1983:2017
plot(x,nx[x-1977])
fitnx <- lm(nx[x-1977]~x)
y <- 1979:1982
nx[y-1977] <- predict(fitnx,data.frame(x=y))
plot(t,nx)
eng1$宁夏 <- nx

# 青海 1984
qh <- eng1$青海
x <- 1984:2017
plot(x,hn[x-1977])
fithn <- lm(hn[x-1977]~x)
y <- 1979:1983
qh[2:6] <- predict(fithn,data.frame(x=y))
plot(t,qh)
eng1$青海 <- qh

# 西藏 1990
xz <- eng1$西藏
x <- 1990:2017
plot(x,xz[x-1977])
fitxz <- lm(xz[x-1977]~x)
y <- 1979:1989
xz[y-1977] <- predict(fitxz,data.frame(x=y))
plot(t,xz)
eng1$西藏 <- xz

# 新疆 1980
gs <- eng1$新疆
x <- 1980:2017
plot(x,gs[x-1977])
y <- 1979
fitgs <- lm(gs[x-1977]~x)
gs[2] <- predict(fitgs,data.frame(x=y))
eng1$新疆 <- gs

# 全体1978
x <- 1979:2017
y = data.frame(x=1978)
eng2 <- eng1
for(i in 2:32){
  z <- as.data.frame(eng1[,i])
  fit <- lm(z[x-1977,]~x)
  z[1,] <- predict(fit,y)
  eng2[,i] <- z
}

plot(t,eng2$安徽)
plot(t,eng2$北京)
plot(t,eng2$福建)
plot(t,eng2$甘肃)
plot(t,eng2$广东)
plot(t,eng2$广西)

write.csv(eng2, file = "D:/大三下/统计建模大赛/Code/data/clean乡村恩格尔系数78-17.csv")
