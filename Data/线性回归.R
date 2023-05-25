# 线性回归
library(R.matlab)
savedata <- 1

data <- readMat("D:/大三下/统计建模大赛/Code/yrprvcdata_z.mat")
R <- data$R
CC <- data$CC
ft <- data$ft

Tno <- nrow(ft)
X1 <- cbind(rep(1,Tno),ft)
fit1 <- lm(R~ft)
Yhat <- fit1$fitted.values
R2 <- 1-sum((Yhat-R)^2)/sum((R-mean(R))^2)
summary(fit1)

X2 <- cbind(ft,CC)
fit2 <- lm(R~X2)
summary(fit2)


# general linear test
SSER <- sum((R-fit1$fitted.values)^2)
SSEF <- sum((R-fit2$fitted.values)^2)
Fstat <- (Tno-2)*(SSER-SSEF)/(8*SSEF)



# 工具变量作解释变量
fit3 <- lm(R~CC)
summary(fit3)

sliced <- readMat("D:/大三下/统计建模大赛/Code/yrprvcdata_sliced.mat")
Rtrn <- sliced$Rtrn
Rtst <- sliced$Rtst
Ctrn <- sliced$Ctrn
Ctst <- sliced$Ctst
fttrn <- sliced$fttrn
fttst <- sliced$fttst

#colnames(fttrn) <- c('K','L')
#colnames(fttst) <- c('fttrnK','fttrnL')
trn <- data.frame(Rtrn=Rtrn,Ctrn=Ctrn,fttrn=fttrn)
tst <- data.frame(Rtst=Rtst,Ctst=Ctst,fttst=fttst)

fs1 <- lm(Rtrn~fttrn,data = trn)
#summary(fs1)
residual1 <- fs1$residuals
R1in <- 1-sum(residual1^2)/sum((Rtrn-mean(Rtrn))^2)
beta1 <- matrix(fs1$coefficients,3,1)
Rpred <- as.matrix(cbind(rep(1,223),fttst),223,3)%*%beta1
error1 <- Rtst-Rpred
R1 <- 1-sum((Rtst-Rpred)^2)/sum((Rtst-mean(Rtst))^2)



fs2 <- lm(Rtrn~fttrn+Ctrn,data = trn)
summary(fs2)
residual2 <- fs2$residuals
R2in <- 1-sum(residual2^2)/sum((Rtrn-mean(Rtrn))^2)
beta2 <- matrix(fs2$coefficients,11,1)
Rpred <- as.matrix(cbind(rep(1,223),fttst,Ctst),223,11)%*%beta2
error2 <- Rtst-Rpred
R2 <- 1-sum((Rtst-Rpred)^2)/sum((Rtst-mean(Rtst))^2)

if(savedata==1){
  error = data.frame(error1=error1,error2=error2)
  residual = data.frame(residual1=residual1,residual2=residual2)
  write.csv(error,file = "D:/大三下/统计建模大赛/Code/linearerror.csv")
  write.csv(residual,file = "D:/大三下/统计建模大赛/Code/linearresidual.csv")
}