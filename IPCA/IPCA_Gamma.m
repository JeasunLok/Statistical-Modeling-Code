function [ Gamma_New ] = IPCA_Gamma( Gamma_Old , W , X , Nts , Factor_Old)
% inputs
% - Gamma_Old : LxK matrix of previous iteration's GammaBeta estimate
% - W : LxLxT array of Z(:,:,t)'*Z(:,:,t)/Nts(t)
% - X : LxT array of Z(:,:,t)'*Y(:,t)/Nts(t)
% - Nts : 1xT vector of cross-sectional size [typically found as sum(LOC)]
% - (optional) PSF : Kadd x T Pre-Specified Factors
%
% outputs
% - Gamma_New : LxK matrix of this iteration's GammaBeta estimate
%
% Imposes identification assumption on Gamma_New and F_New: Gamma_New is orthonormal
% matrix and F_New has positive mean (taken across the T columns)

PSF = Factor_Old; % pre-specified factor, 指定潜在因子
%[Kadd,Tadd] = size(PSF);
PSF_version = true;


T = length(Nts);
[L,Ktilde] = size(Gamma_Old);
K = 0;

if K>0 % this could run using only prespecified factors
    
if PSF_version % pre-specified factors in the model
    F_New = nan(K,T);
    for t=1:T % 论文（2）式， 报告（10）式
        F_New(:,t) = ( Gamma_Old(:,1:K)'*W(:,:,t)*Gamma_Old(:,1:K) )\...
            ( Gamma_Old(:,1:K)'*( X(:,t)-W(:,:,t)*Gamma_Old(:,K+1:Ktilde)*PSF(:,t) ) );
    end
else % no pre-specified factors in the model
    F_New = nan(K,T);
    for t=1:T % 论文（2）式， 报告（10）式
        F_New(:,t) = ( Gamma_Old'*W(:,:,t)*Gamma_Old )\( Gamma_Old'*X(:,t) );
    end
end

end


Numer = zeros(L*Ktilde,1);
Denom = zeros(L*Ktilde);
if PSF_version % pre-specified factors in the model
    if K>0
    for t=1:T % 论文（3）式，报告（11）式
        Numer = Numer + kron( X(:,t)        , [F_New(:,t);PSF(:,t)]                            )*Nts(t);
        Denom = Denom + kron( W(:,:,t)      , [F_New(:,t);PSF(:,t)]*[F_New(:,t);PSF(:,t)]'     )*Nts(t);
    end
    else
    for t=1:T % 论文（3）式，报告（11）式
        Numer = Numer + kron( X(:,t)        , [PSF(:,t)]                            )*Nts(t);
        Denom = Denom + kron( W(:,:,t)      , [PSF(:,t)]*[PSF(:,t)]'     )*Nts(t);
    end
    end
else % no pre-specified factors in the model
    for t=1:T
        Numer = Numer + kron( X(:,t)        , F_New(:,t)                )*Nts(t);
        Denom = Denom + kron( W(:,:,t)      , F_New(:,t)*F_New(:,t)'    )*Nts(t);
    end
end
Gamma_New_trans_vec = Denom\Numer;
Gamma_New_trans     = reshape(Gamma_New_trans_vec,Ktilde,L); % 把向量（论文中\gamma）还原成矩阵
Gamma_New           = Gamma_New_trans';

% % GammaBeta orthonormal and F_New Orthogonal （标准化）
% if K>0
% R1                  = chol(Gamma_New(:,1:K)'*Gamma_New(:,1:K),'upper');
% [R2,~,~]            = svd(R1*F_New*F_New'*R1');
% Gamma_New(:,1:K)    = (Gamma_New(:,1:K)/R1)*R2;
% F_New               = R2\(R1*F_New);
% end
% 
% % Sign convention on GammaBeta and F_New
% if K>0
% sg = sign(mean(F_New,2));
% sg(sg==0)=1; % if mean zero, do not flip signs of anything
% Gamma_New(:,1:K) = Gamma_New(:,1:K).*sg';
% F_New = F_New .* sg;
% end

end

