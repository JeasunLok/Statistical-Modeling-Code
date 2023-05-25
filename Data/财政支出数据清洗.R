library("haven")
library("readxl")
#capital <- read_dta("D:/大三下/统计建模大赛/data/GDP/ChinaCapitalStock-master/2000-2017年中国各省资本存量/province/result_data/capital_stock_1.dta")
spend <- read_excel("D:/大三下/统计建模大赛/data/GDP/财政支出78-21.xlsx")
sc <- spend$四川
cq <- spend$重庆
# 处理四川数据
plot(1978:2021,log(sc))
x <- 1985:2021
fitsc <- lm(log(sc[8:44])~x)
scn <- predict(fitsc,data.frame(x=1978:1984))
sc[1:7] <- exp(scn)
plot(1978:2021,log(sc))

#处理重庆数据
plot(1978:2021,log(cq))
x <- 1997:2010
fitcq <- lm(log(cq[20:33])~x)
cqn <- predict(fitcq,data.frame(x=1978:1996))
cq[1:19] <- exp(cqn)
plot(1978:2021,log(cq))

spend$四川 <- sc
spend$重庆 <- cq
write.csv(spend,file = "D:/大三下/统计建模大赛/data/GDP/clean财政支出78-21.csv")
