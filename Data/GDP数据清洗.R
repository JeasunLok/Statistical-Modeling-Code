library("readxl")
gdp <- read_excel("D:/大三下/统计建模大赛/data/GDP/gdp.xlsx")
prvcnm <- levels(as.factor(gdp$Prvcnm))
prvcnm <- prvcnm[-31]

gdp1 <- data.frame(yr=1978:2021)
colnames(gdp1) <- "年"

for(n in 1:length(prvcnm)){
  nc <- as.data.frame(rep(0,44))
  cadd <- as.data.frame(gdp[gdp$Prvcnm==prvcnm[n],'Gdp0101'])
  m <- nrow(cadd)
  for(i in 1:m){
    nc[44-m+i,] <- cadd[i,]
  }
  colnames(nc) <- prvcnm[n]
  gdp1 <- cbind(gdp1,nc)
}

# 海南数据1978-1986
# 1986年：48.03 1985年：43.26 1984年：37.18 1983年：31.12 1982年：28.86 
# 1981年：22.23 1980年：19.33 1979年：17.45 1978年：16.4
hn <- c(16.4,17.45,19.33,22.23,28.86,31.12,37.18,43.26,48.03)
gdp1$海南省[1:9] <- hn

write.csv(gdp1,file="D:/大三下/统计建模大赛/data/GDP/cleanGDP78-21.csv")
