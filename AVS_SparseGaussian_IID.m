% This function determines the performance of the AVS for vectorized
% version
function [x_hat,MSE_out,Sensing_Profile,MSE_old]=AVS_SparseGaussian_IID(r,N,M,sig_0,spa)

K=floor(N/r);
sig=M*sig_0;

x_hat=zeros(N,1);
MSE=1000*ones(N,1);
SumY=zeros(N,1);

L_m=zeros(N,1);

% source
x=randn(N,1).*(rand(N,1)<spa);

% % Initialization
% for ind=1:floor(r)
%     U=Haar(K);
%     col=((ind-1)*K+1):ind*K;
%     A=zeros(K,N);
%     A(:,col)=U;
%     y=A*x+sqrt(sig)*randn(K,1);
%     % receive
%     w=U'*y;
%     SumY(col)=SumY(col)+w;
%     x_hat(col)=estimator(L_m(col),SumY(col));
%     MSE(col)=mse_func(L_m(col),SumY(col));
% end
% index_l2=(MSE==0);
% MSE(index_l2)=Inf;
%% Further Sensing

for ind=1:M
[~,Index_Worst]=sort(MSE);

    U=randn(K)/sqrt(K);
    col=Index_Worst(N-K+1:N);
    A=zeros(K,N);
    A(:,col)=U;
    y=A*x+sqrt(sig)*randn(K,1);
    
    if ind==1
        Input.A = A;
        Input.y = y;
        
    else
        Input.A = [Input.A ; A];
        Input.y = [Input.y ; y];
        
    end

    
    %
    % receive
    %
    w=(U'*U).'*U'*y;
    SumY(col)=SumY(col)+w;
    L_m(col)=L_m(col)+1;
    x_hat(col)=estimator(L_m(col),SumY(col));
    MSE(col)=mse_func(L_m(col),SumY(col));
end



MSE_out=norm(x-x_hat,2)^2/N;
Sensing_Profile=L_m/M;


Input.x = x;
Input.mse = 0.8;
Input.K = 1;
Input.nuw = 0.01*M;
Input.M = K*M;
Input.N = N;
Input.IterNum = 30;
Input.rho = 0.1;
[MSE_old,m0_sub,v0_sub,hatx_plus,vx_plus]=VAMP(Input,1);

%% Locals

    function b=estimator(L,Sy)
        e_term=exp(.5*Sy.^2./(sig*(L+sig)));
        d_term=(1-spa)*sqrt(1+L/sig)./(spa*e_term);
        b=Sy./((L+sig).*(1+d_term));
%         index_l1=(Sy==0);
%         b(index_l1)=NaN;
    end

    function mse=mse_func(L,Sy)     
        e_term=exp(.5*Sy.^2./(sig*(L+sig)));
        d_term=(1-spa)*sqrt(1+L/sig)./(spa*e_term);
        T_1=sig+(d_term.*Sy.^2)./((L+sig).*(1+d_term));
        T_2=1./((1+d_term).*(L+sig));
        mse=T_1.*T_2;
%         index_l2=(Sy==0);
%         mse(index_l2)=Inf;
    end
end


function [MSE_old,m0_sub,v0_sub,hatx_plus,vx_plus]=VAMP(Input,m)

A=Input.A;
KKK = Input.K;
H = [];
for i = 1:m
    H = [H;A(:,:,i)];
end

x=Input.x;
rho=Input.rho;
[M,N]=size(H);
mes=0.8;                            % damping 
IterNum=Input.IterNum;

v1_plus=ones(M,1);
v1_plus_inv=1./v1_plus;
m1_plus=zeros(M,1);
v1_plus_inv_old=v1_plus_inv;
m1_plus_old=m1_plus;

v0_plus=2*ones(N,1);
v0_plus_inv=1./v0_plus;
m0_plus=zeros(N,1);
v0_plus_inv_old=v0_plus_inv;
m0_plus_old=m0_plus;

MSE_error=zeros(IterNum,1);
MSE_old=1;
for ii=1:IterNum
    %% Backward passing
    
    [hatz_sub,vz_sub]=estimator_z(m1_plus./v1_plus_inv,1./v1_plus_inv,Input,m,KKK);
    v1_sub=max(vz_sub./(1-vz_sub.*v1_plus_inv),eps);          %������ָ����Լ�?0
    m1_sub=v1_sub.*(hatz_sub./vz_sub-m1_plus);
        
    Qx_sub=(H'*diag(1./v1_sub)*H+diag(v0_plus_inv))^(-1);
    hatx_sub=Qx_sub*(H'*diag(1./v1_sub)*m1_sub+m0_plus);                     %�����r0_plus
    vx_sub=diag(Qx_sub);
    v0_sub=max(vx_sub./(1-vx_sub.*v0_plus_inv),eps);
    m0_sub=v0_sub.*(hatx_sub./vx_sub-m0_plus);

    %% Forward passing

    [hatx_plus,vx_plus]=estimator_x(rho,m0_sub,v0_sub);
    
    MSE=norm(x-hatx_plus,'fro')^2/N;
    if ii>1 && (isnan(MSE)  || MSE>(MSE_old- (1e-9)))
        MSE_error(ii:IterNum,1)=MSE_old;
        break;
    end
    MSE_old=MSE;
    MSE_error(ii,1)=MSE;
    
    vx_plus=max(vx_plus,5e-13);
    v0_plus_inv=(v0_sub-vx_plus)./vx_plus./v0_sub;
    m0_plus=(hatx_plus.*v0_sub-m0_sub.*vx_plus)./vx_plus./v0_sub;
    
    negldx=v0_plus_inv<eps;
    v0_plus_inv(negldx)=v0_plus_inv_old(negldx);
    m0_plus(negldx)=m0_plus_old(negldx);
    
    % damping
    [v0_plus_inv,v0_plus_inv_old]=damping(v0_plus_inv,v0_plus_inv_old, mes);
    [m0_plus,m0_plus_old]=damping(m0_plus,m0_plus_old, mes);
      
    Qx_plus=(H'*diag(1./v1_sub)*H+diag(v0_plus_inv))^(-1);
    mx_plus=Qx_plus*(H'*diag(1./v1_sub)*m1_sub+m0_plus);
    hatz_plus=H*mx_plus;
    vz_plus=diag(H*Qx_plus*H');

    vz_plus=max(vz_plus,5e-13);
    v1_plus_inv=(v1_sub-vz_plus)./v1_sub./vz_plus;
    m1_plus=(hatz_plus.*v1_sub-m1_sub.*vz_plus)./v1_sub./vz_plus;

    negldx=v1_plus_inv<0;
    v1_plus_inv(negldx)=v1_plus_inv_old(negldx);
    m1_plus(negldx)=m1_plus_old(negldx);
     
    % damping
    [v1_plus_inv,v1_plus_inv_old]=damping(v1_plus_inv,v1_plus_inv_old,mes);
    [m1_plus,m1_plus_old]=damping(m1_plus,m1_plus_old,mes);
      
    
end
end

function [umean,uvar]=estimator_x(hat_rho,v,wvar)
sigma_x=1;

%% Perform MMSE estimator
Gaussian=@(x,a,A) 1./(pi*A).*exp(-1./A.*abs(x-a).^2);
C=(hat_rho*Gaussian(0,v,wvar+sigma_x))./...
    ((1-hat_rho)*Gaussian(0,v,wvar)+hat_rho*Gaussian(0,v,wvar+sigma_x));

umean=C.*(v*sigma_x)./(sigma_x+wvar);
uvar=C.*(abs((v*sigma_x)./(sigma_x+wvar)).^2+(sigma_x*wvar)./(sigma_x+wvar))-abs(umean).^2;

end


function [x,x_old]=damping(x, x_old, mes)
x=mes*x+(1-mes)*x_old;
x_old=x;
end


function [hatz,hatv]=estimator_z(m,v,Input,iii,KKK)

nuw=Input.nuw*KKK;
y = [];
for i = 1:iii
    y = [y;Input.y(:,i)];
end
M=Input.M*iii;

    hatv=1./(1./v+1./nuw);
    hatz=hatv.*(y./nuw+m./v);
end