% This function determines the performance of the AVS for vectorized
% version
function [x_hat,MSE_out,Sensing_Profile,support_X,MSE_old]=AVS_SparseGaussian_IID_Bayes(r,N,M,sig_0,spa,mse_th)

K=floor(N/r);
sig=M*sig_0;
x=randn(N,1).*(rand(N,1)<spa);
support_X = (x~=0);
support_X = double(support_X);

%x_hat=zeros(N,1);

Sa=ones(N,1);
Sy=zeros(N,1);
L=ones(N,1);
% source


% % Initialization
% for ind=1:floor(r)
r_m=N/K;
A=randn(K,N)/sqrt(K);
y=A*x+sqrt(sig)*randn(K,1);
[x_hat,Sy,Sa]=estimator(Sy,Sa,y,A,r_m);
MSE=mse_func(Sy,Sa);
% end
% index_l2=(MSE==0);
% MSE(index_l2)=Inf;
%% Further Sensing
        Input.A = A;
        Input.y = y;


for ind=2:M
    A=randn(K,N)/sqrt(K);
 
%     index_m=(MSE<mse_th);
%     L=L+1-index_m;
%     A(:,index_m)=0;
    index_m=(MSE<mse_th);
    L=L+1;
    
    A(:,index_m)=0;

    r_m=N/K;
    y=A*x+sqrt(sig)*randn(K,1);
    
    Input.A = [Input.A ; A];
    Input.y = [Input.y ; y];
    %
    % receive
    %
[x_hat,Sy,Sa]=estimator(Sy,Sa,y,A,r_m);
MSE=mse_func(Sy,Sa);
end

MSE_out=norm(x-x_hat,2)^2/N;
Sensing_Profile=L/M;


%%  VAMP algorithm
Input.x = x;
Input.K = 1;
Input.nuw = 0.01*M;
Input.M = K*M;
Input.N = N;
Input.IterNum = 30;
Input.rho = 0.1;
[MSE_old,m0_sub,v0_sub,hatx_plus,vx_plus]=VAMP(Input,1);

%% Locals

    function [b,Sy,Sa]=estimator(Sy0,Sa0,y,A,r_m)
        sig_m=r_m*spa+sig;
        Sy=Sy0+A'*y/sig_m;
        Sa=Sa0+diag(A'*A)/sig_m;
        mu_n=Sy./Sa;
        u=sqrt(Sa);
        theta=.5*(Sy.^2)./Sa;
        d=((1-spa)/spa)*u.*exp(-theta);
        b=mu_n./(1+d);
    end

    function mse=mse_func(Sy,Sa)
        u=sqrt(Sa);
        theta=.5*(Sy.^2)./Sa;
        mu_n=Sy./Sa;
        d=((1-spa)/spa)*u.*exp(-theta);
        b=mu_n./(1+d);
        mse_Nom=b.^2.*d+(1./Sa)+(mu_n-b).^2;
        mse=mse_Nom./(1+d);
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
    v1_sub=max(vz_sub./(1-vz_sub.*v1_plus_inv),eps);          %??????????????????????????????????????0
    m1_sub=v1_sub.*(hatz_sub./vz_sub-m1_plus);
        
    Qx_sub=(H'*diag(1./v1_sub)*H+diag(v0_plus_inv))^(-1);
    hatx_sub=Qx_sub*(H'*diag(1./v1_sub)*m1_sub+m0_plus);                     %???????????????r0_plus
    vx_sub=diag(Qx_sub);
    v0_sub=max(vx_sub./(1-vx_sub.*v0_plus_inv),eps);
    m0_sub=v0_sub.*(hatx_sub./vx_sub-m0_plus);

    %% Forward passing

    [hatx_plus,vx_plus]=estimator_x(rho,m0_sub,v0_sub);
    
    MSE=norm(x-hatx_plus,2)^2/N;
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