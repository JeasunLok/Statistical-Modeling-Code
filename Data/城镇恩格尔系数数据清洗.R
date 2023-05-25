# 城镇恩格尔系数
library("readxl")
eng <- read_excel("D:/大三下/统计建模大赛/Code/data/城镇恩格尔系数.xls")
eng1 <- eng[,order(names(eng))]
colnames(eng1)[1] <- "年"
prvcnm <- colnames(eng1)[2:32]

t <- 1978:2017

# 安徽 1981
ah <- eng1$安徽
x <- 1981:2017
plot(x,ah[x-1977])
y <- 1979:1980
fitah <- lm(ah[x-1977]~x)
ah[2:3] <- predict(fitah,data.frame(x=y))
plot(t,ah)
eng1$安徽 <- ah

# 福建 1981
ah <- eng1$福建
x <- 1981:2017
plot(x,ah[x-1977])
y <- 1979:1980
fitah <- lm(ah[x-1977]~x)
ah[2:3] <- predict(fitah,data.frame(x=y))
plot(t,ah)
eng1$福建 <- ah

# 甘肃 1980
gs <- eng1$甘肃
x <- 1980:2017
plot(x,gs[x-1977])
y <- 1979
fitgs <- lm(gs[x-1977]~x)
gs[2] <- predict(fitgs,data.frame(x=y))
eng1$甘肃 <- gs

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
eng1$海南 <- hn

# 河北 1980
gs <- eng1$河北
x <- 1980:2017
plot(x,gs[x-1977])
y <- 1979
fitgs <- lm(gs[x-1977]~x)
gs[2] <- predict(fitgs,data.frame(x=y))
eng1$河北 <- gs

# 黑龙江 1980
gs <- eng1$黑龙江
x <- 1980:2017
plot(x,gs[x-1977])
y <- 1979
fitgs <- lm(gs[x-1977]~x)
gs[2] <- predict(fitgs,data.frame(x=y))
eng1$黑龙江 <- gs

# 湖北 1980
gs <- eng1$湖北
x <- 1980:2017
plot(x,gs[x-1977])
y <- 1979
fitgs <- lm(gs[x-1977]~x)
gs[2] <- predict(fitgs,data.frame(x=y))
eng1$湖北 <- gs

# 湖南 1980
gs <- eng1$湖南
x <- 1980:2017
plot(x,gs[x-1977])
y <- 1979
fitgs <- lm(gs[x-1977]~x)
gs[2] <- predict(fitgs,data.frame(x=y))
eng1$湖南 <- gs

# 吉林 1979，1981—1984，1986-1989
jl <- eng1$吉林
x <- c(1980,1985,1990:2017)
plot(x,jl[x-1977])
fitjl <- lm(jl[x-1977]~x)
y <- c(1979,1981:1984,1986:1989)
jl[y-1977] <- predict(fitjl,data.frame(x=y))
plot(t,jl)
eng1$吉林 <- jl

# 江苏 1980
gs <- eng1$江苏
x <- 1980:2017
plot(x,gs[x-1977])
y <- 1979
fitgs <- lm(gs[x-1977]~x)
gs[2] <- predict(fitgs,data.frame(x=y))
eng1$江苏 <- gs

# 江西 1981
ah <- eng1$江西
x <- 1981:2017
plot(x,ah[x-1977])
y <- 1979:1980
fitah <- lm(ah[x-1977]~x)
ah[2:3] <- predict(fitah,data.frame(x=y))
plot(t,ah)
eng1$江西 <- ah

# 辽宁 1985
hn <- eng1$辽宁
x <- 1985:2017
plot(x,hn[x-1977])
fithn <- lm(hn[x-1977]~x)
y <- 1979:1984
hn[2:7] <- predict(fithn,data.frame(x=y))
eng1$辽宁 <- hn

# 内蒙古 1979-1984，1986-1988
nmg <- eng1$内蒙古
x <- c(1985,1989:2017)
plot(x,nmg[x-1977])
fitnmg <- lm(nmg[x-1977]~x)
y <- c(1979:1984,1986:1988)
nmg[y-1977] <- predict(fitnmg,data.frame(x=y))
eng1$内蒙古 <- nmg

# 青海 1984
qh <- eng1$青海
x <- 1984:2017
plot(x,hn[x-1977])
fithn <- lm(hn[x-1977]~x)
y <- 1979:1983
qh[2:6] <- predict(fithn,data.frame(x=y))
plot(t,qh)
eng1$青海 <- qh

# 山西 1980
gs <- eng1$山西
x <- 1980:2017
plot(x,gs[x-1977])
y <- 1979
fitgs <- lm(gs[x-1977]~x)
gs[2] <- predict(fitgs,data.frame(x=y))
eng1$山西 <- gs

# 陕西 1981
ah <- eng1$陕西
x <- 1981:2017
plot(x,ah[x-1977])
y <- 1979:1980
fitah <- lm(ah[x-1977]~x)
ah[2:3] <- predict(fitah,data.frame(x=y))
plot(t,ah)
eng1$陕西 <- ah

# 天津 1980
gs <- eng1$天津
x <- 1980:2017
plot(x,gs[x-1977])
y <- 1979
fitgs <- lm(gs[x-1977]~x)
gs[2] <- predict(fitgs,data.frame(x=y))
eng1$天津 <- gs

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

# 浙江 1981
ah <- eng1$浙江
x <- 1981:2017
plot(x,ah[x-1977])
y <- 1979:1980
fitah <- lm(ah[x-1977]~x)
ah[2:3] <- predict(fitah,data.frame(x=y))
plot(t,ah)
eng1$浙江 <- ah

# 全体 1978
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

write.csv(eng2, file = "D:/大三下/统计建模大赛/Code/data/clean城镇恩格尔系数78-17.csv")
