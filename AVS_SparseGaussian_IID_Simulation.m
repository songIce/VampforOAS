clear
clc

%% Common Parameters
sparsity=.1;
sigma_noise=.01;
% r: variable
%% Parameters for AVS
ITR_AVS=1e3;
N=100;
M=40;
%r=2;
Sense_out=zeros(N,1);
supp_out=zeros(N,1);

%% Simulations
r_vec=1:.5:5;
MSE_AVS_dB=zeros(size(r_vec));
MSE_VAMP_dB=zeros(size(r_vec));
D_th=-26.5;
%% AVS
parfor_progress(ITR_AVS*length(r_vec));

for l=1:length(r_vec)
   % Sense=zeros(N,1);
   % supp=zeros(N,1);
    r=r_vec(l)
    MSE_out=zeros(ITR_AVS,1);
    parfor itr=1:ITR_AVS
        itr;
        %[~,MSE_out(itr),Sensing_Profile]=AVS_SparseGaussian_Haar(r,N,M,sigma_noise,sparsity);
        [~,MSE_out(itr),Sensing_Profile,support_X,MSE_old(itr)]=AVS_SparseGaussian_IID_Bayes(r,N,M,sigma_noise,sparsity,10^(D_th/10));
        %
%         [~,MSE_out(itr),Sensing_Profile,MSE_old(itr)]=AVS_SparseGaussian_IID(r,N,M,sigma_noise,sparsity);
        
        parfor_progress;  
   %     Sense=Sense+Sensing_Profile;
   %     supp=supp+support_X;
    end
    %Sense=Sense/ITR_AVS;
    %Sense=Sense-sum(Sense)/N;
    %supp=supp/ITR_AVS;
    %supp=supp-sum(supp)/N;
    
    %Sense_out=Sense_out+Sense;
    %supp_out=supp_out+supp;
    MSE_vamp=sum(MSE_old)/ITR_AVS;
    MSE_avg=sum(MSE_out)/ITR_AVS;
    MSE_AVS_dB(l)=10*log10(MSE_avg);
    MSE_VAMP_dB(l)=10*log10(MSE_vamp);
    %%
    %     figure(l)
    %     plot(1:N,Sense)
    %     hold on
    %     plot(1:N,supp)
    %     pause(.1)
end
parfor_progress(0); 
%Sense_out=Sense_out/length(r_vec);
%supp_out=supp_out/length(r_vec);
%%
figure(1)
% plot(1:N,Sense_out)
% hold on
% plot(1:N,supp_out)
% figure(2)
hold on
plot(r_vec,MSE_AVS_dB,'-o')
plot(r_vec,MSE_VAMP_dB,'-o')
legend('AVS','VAMP')
hold on