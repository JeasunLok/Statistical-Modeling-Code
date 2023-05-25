% option
clear
L = 8;
N = 31;
T = 40;
K = 2;
savedata = 0;

% load data
GDP = csvread('./data/cleanGDP78-17.csv',1,2);
KK   = csvread('./data/clean资本存量78-17.csv',1,2);
LL   = csvread('./data/clean就业人口78-17.csv',1,2);
C0 = cell(L,1);

C0{1,1} = csvread('./data/cleanRPI78-17.csv',1,2); % 零售价格指数RPI
C0{2,1} = csvread('./data/cleanTRSCG78-17.csv',1,2); % 社会消费品零售总额TRSCG
C0{2,1} =log(C0{2,1});
C0{3,1} = csvread('./data/clean乡村恩格尔系数78-17.csv',1,2);
C0{4,1} = csvread('./data/clean产业结构指数78-17.csv',1,2);
C0{5,1} = csvread('./data/clean发电量78-17.csv',1,2);
C0{5,1} =log(C0{5,1});
C0{6,1} = csvread('./data/clean城镇化率78-17.csv',1,2);
C0{7,1} = csvread('./data/clean城镇恩格尔系数78-17.csv',1,2);
C0{8,1} = csvread('./data/clean财政支出78-17.csv',1,2);
C0{8,1} =log(C0{8,1});
%C0{9,1} = ones(T,N);

% 重庆从1995开始
TN = T*(N-1)+23;

CC = nan(TN,L);
R = nan(TN,1);
ft= nan(TN,K);
for l = 1:L
    %C1{l,1} = zscore(C0{l,1},0,'all');
    C1 = C0;
    for n = 1:30
        for t = 1:T
            CC((n-1)*T+t,l) = C1{l,1}(t,n);
            R((n-1)*T+t) = log(GDP(t,n));
            ft((n-1)*T+t,1) = log(KK(t,n));
            ft((n-1)*T+t,2) = log(LL(t,n));
        end
    end
    % 加上重庆数据
    for tt = 18:T
        CC(30*40+tt-17,l) = C1{l,1}(tt,31);
        R(30*40+tt-17) = log(GDP(tt,31));
        ft(30*40+tt-17,1) = log(KK(tt,31));
        ft(30*40+tt-17,2) = log(LL(tt,31));
    end
end
[C1,mu,sigma] = zscore(CC(:,1:8));
CC(:,1:8) = C1;%(CC(:,1:8)-mu)./sigma;

% output
T = TN;
% if savedata == 1
%     filename = 'yrprvcdata.mat';
%     datafile = ['./' erase(filename,'.mat')];
%     save(datafile,'CC','R','ft','N','L','T','K');
% end

ft = [ones(TN,1),ft];

% Slicing data
rng(1);
TT = 1000;
id = sort(randsample(TN,TT));
id1= setdiff(1:TN,id);
Ctrn = CC(id,:);
Ctst = CC(id1,:);
Rtrn = R(id,:);
Rtst = R(id1,:);
fttrn = ft(id,:);
fttst = ft(id1,:);

if savedata == 1
    filename = 'yrprvcdata_sliced_all_log.mat';
    datafile = ['./' erase(filename,'.mat')];
    save(datafile,'Ctrn','Ctst','Rtrn','Rtst','fttrn','fttst','N','L','TT','K');
end


% reshape data
C = nan(1,L,TN);
for l = 1:L
    for t = 1:TN
        C(1,l,t) = CC(t,l);
    end
end

Y = R';

f = ft';

% ------------------------------------------------------------------
% IPCA
% ------------------------------------------------------------------
% Numerical choices
TT          = 1000;
N           = 1;
K           = 3;
W           = nan(L,L,TT);
X           = nan(L,TT);
for t = 1:TT
    W(:,:,t)    = C(:,:,t)'*C(:,:,t)/N;
    X(:,t)      = C(:,:,t)'*Y(:,t)/N;
end
    
% initial guess
[Gamma_Old,s,v]     = svds(X,K); % guess
if K == 1
    Factor_Old          = s*v';
elseif K == 3
    PSF0          = f(1:3,:);
    %PSF           = rankstdize(PSF0);
    %PSF          = PSF0./(sqrt(sum(PSF0.^2)));
    PSF           = PSF0(:,1:TT);
    Factor_Old2         = s*v';
    Factor_Old          = Factor_Old2(1,:);
end

% algorithm
Nts         = N*ones(TT,1);

% Run regressions using Old estimates
[Gamma_New] = IPCA_Gamma(Gamma_Old,W,X,Nts,PSF);
% Calculate change in Old and New estimates
%tol         = max([ abs(Gamma_New(:)-Gamma_Old(:)) ; abs(Factor_New(:)-Factor_Old(:)) ]); % other convergence norms could be used
% Replace Old estimates for the next iteration
%Factor_Old  = Factor_New;

Gamma   = Gamma_New;
Factor  = PSF;
%Factor  = [Factor_New;PSF];

% Goodness-of-fit
% in-sample
Yhat    = nan(1,TT);
IPCAresidual = nan(1,TT);
for t = 1:TT
    Yhat(1,t) = C(1,:,t)*Gamma*Factor(:,t);
    IPCAresidual(1,t) = Y(1,t)-Yhat(1,t);
end

R2      = 1-sum(IPCAresidual.^2)/sum((Y(:,1:TT)-mean(Y(:,1:TT),'all')).^2) % 拟合优度R^2

% out-of-sample
Ypred   = nan(1,TN-TT);
IPCAerror   = nan(1,TN-TT);
%Factor_pred = nan(K,TN-TT);
Factor_pred = f(:,(TT+1):TN);
%Factor_pred(2:3,:) = f(:,(TT+1):TN);
%Factor_pred(1,:) = mean(Factor(1,:))*ones(1,TN-TT);
for t = (TT+1):TN
    Ypred(1,t-TT) = C(1,:,t)*Gamma*Factor_pred(:,t-TT);
    IPCAerror(1,t-TT) = Y(1,t)-Ypred(1,t-TT);
end

R2pred  = 1-sum(IPCAerror.^2)/sum((Y(:,(TT+1):TN)-mean(Y(:,(TT+1):TN),'all')).^2)

% test for Instruments' significance(Bootstrap test)
B = 1000;
N = TT;
cnt = zeros(B,L);
for l = 1:L
    W_old = Gamma(l,:)*Gamma(l,:)';
    for b =1:B
        rng(b+1);
        index = datasample(1:TT,N);
        Gammal = Gamma;
        Gammal(l,:) = 0;
        % draw residuals
        e = trnd(5,1,N);
        %e = 0.6*randn(1,N);
        d = Y(:,index)-Yhat(:,index);
        res = e.*d;
        % resample
%         Cpi = C(:,:,index);
%         fpi = f(:,index);
        Cpi = C;
        fpi = f;
        Ypi = nan(1,N);
        for t = 1:N
            Ypi(1,t) = Cpi(1,:,t)*Gammal*fpi(:,t)+res(1,t);
        end
        % re-fit IPCA
        Nts = N*ones(N,1);
        W           = nan(L,L,N);
        X           = nan(L,N);
        for t = 1:N
            W(:,:,t)    = Cpi(:,:,t)'*Cpi(:,:,t)/N;
            X(:,t)      = Cpi(:,:,t)'*Ypi(:,t)/N;
        end
        Gamma_New = IPCA_Gamma(Gamma_Old,W,X,Nts,fpi);
        W_new = Gamma_New(l,:)*Gamma_New(l,:)';
        if W_new>W_old
            cnt(b,l) = 1;
        end
    end
    l
end
% p-value for each instrument
pval = mean(cnt)

IPCAcoef = [Gamma(:,2),Gamma(:,3),Gamma(:,1),pval'];

if savedata==1
    csvwrite('./data/IPCAcoef.csv',IPCAcoef);
end

% calibrate sigma
e = 0.6*randn(1,TN);
1-sum(e.^2)/sum((Y-mean(Y,'all')).^2)

% draw residual and prediction error plot
linearerr = csvread('./data/linearerror.csv',1,1);
linearres = csvread('./data/linearresidual.csv',1,1);
load("result_prediction_label.mat");
dnnres = residual(1,1:TT);
dnnerr = residual(1,(TT+1):end);
x1 = 1.5;

figdir          = './Figures/';

subplot(4,2,1);
histfit(linearres(:,1));
title(sprintf('线性回归（不含工具变量）残差'));
set(gca,'xlim',[-x1 x1]);

subplot(4,2,2);
histfit(linearerr(:,1));
title(sprintf('线性回归（不含工具变量）预测误差'));
set(gca,'xlim',[-x1 x1]);

subplot(4,2,3);
histfit(linearres(:,1));
title(sprintf('线性回归（含工具变量）残差'));
set(gca,'xlim',[-x1 x1]);

subplot(4,2,4);
histfit(linearerr(:,1));
title(sprintf('线性回归（含工具变量）预测误差'));
set(gca,'xlim',[-x1 x1]);

% residual plot
% mean(residual)
% std(residual)
subplot(4,2,5);
histfit(IPCAresidual);
title(sprintf('IPCA残差'));
set(gca,'xlim',[-x1 x1]);

% prediction error plot
% mean(error)
% std(error)
subplot(4,2,6);
histfit(IPCAerror);
title(sprintf('IPCA预测误差'));
set(gca,'xlim',[-x1 x1]);

subplot(4,2,7);
histfit(dnnres);
title(sprintf('DNN残差'));
set(gca,'xlim',[-x1 x1]);

subplot(4,2,8);
histfit(dnnerr);
title(sprintf('DNN预测误差'));
set(gca,'xlim',[-x1 x1]);
fig=gcf;
saveas(fig,[figdir 'residualanderror_Log'],'eps2c');
%close all
