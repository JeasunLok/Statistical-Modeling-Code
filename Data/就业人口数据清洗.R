library("readxl")
labour <- read_excel("D:/大三下/统计建模大赛/data/GDP/就业人口78-20.xlsx")
t <- 1978:2020
# 处理空值
# 黑龙江 3  内蒙古自治区 1  青海 1  上海 9  天津 9  重庆 7
# 黑龙江1981
hlj <- labour$黑龙江
x <- 1981:2020
plot(x,hlj[4:43])
y <- 1981:2001
fithlj <- lm((hlj[4:24])~y)
hljn <- predict(fithlj,data.frame(y=1978:1980))
hlj[1:3] <- hljn
plot(t,hlj,type = 'p')
labour$黑龙江 <- hlj

# 内蒙古 1979
nmg <- labour$内蒙古
x <- 1979:2020
plot(x,nmg[2:43])
y <- 1979:1985
fitnmg <- lm(nmg[y-1977]~y)
nmg[1] <- predict(fitnmg,data.frame(y=1978))
plot(t,nmg)
labour$内蒙古 <- nmg

#青海 1979
qh <- labour$青海
x <- 1979:2020
plot(x,qh[2:43])
y <- 1979:1990
fitqh <- lm(qh[y-1977]~y)
qh[1] <- predict(fitqh,data.frame(y=1978))
plot(t,qh)
labour$青海 <- qh

#上海 1987
shh <- labour$上海
x <- 1987:2020
plot(x,shh[x-1977])
y <- 1987:1991
fitshh <- lm(shh[y-1977]~y)
shh[1:9] <- predict(fitshh,data.frame(y=1978:1986))
plot(t,shh)
labour$上海 <- shh

# 天津 1987
tj <- labour$天津
x <- 1987:2020
plot(x,tj[x-1977])
y <- 1987:1995
fittj <- lm(tj[y-1977]~y)
tj[1:9] <- predict(fittj,data.frame(y=1978:1986))
plot(t,tj)
labour$天津 <- tj

# 重庆 1985
chq <- labour$重庆
x <- 1985:2020
plot(x,chq[x-1977])
y <- 1985:1998
fitchq <- lm(chq[y-1977]~y)
chq[1:7] <- predict(fitchq,data.frame(y=1978:1984))
plot(t,chq)
labour$重庆 <- chq

write.csv(labour,file="D:/大三下/统计建模大赛/data/GDP/clean就业人口78-21.csv")
