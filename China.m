%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Regression on Chinese Gini
%% -----------Setting Environment-----------
clear all;
clc;
addpath('functions');
rng(100)
%% -----------LOAD DATA-----------
% load data used to extract factors
[data, name] = xlsread('China_reg.xlsx');
xdata = data(:,[2:10]);
ydata = data(:,[1,11]);
slowcode = xlsread('China_index.xls');
% Define X and Y matrices
X = xdata;   % X contains the 'xdata' which are used to extract factors.
Y = ydata; % Y contains inflation, unemployment and interest rate
% Number of observations and dimension of X and Y
T = size(Y,1); % T time series observations
N = size(X,2); % N series from which we extract factors
M = size(Y,2); % and M(=1) series for the federal funds rate

%% -----------PRELIMINARIES-----------
% Setting Factor Numbers and VAR Lags 
K = 2;               % number of factors
tau = 0;             % tau is the size of the training sample
p = K+M;             % p is the dimensionality of [Factors, Y]
plag = 3;            % plag is number of lags in the VAR part
numa = p*(p-1)/2;    % numa is the number of elements of At
i_count = 0;         % just a counter for saved draws
% Extracting Factors via Principal Component Analysis
[~,F0,pvar] = pca(xdata);
F0 = F0(:,1:K);
% Set some Gibbs - related preliminaries
nrep = 50000;  % Number of replications
nburn = 10000;   % Number of burn-in-draws
nthin = 10;   % Consider every thin-th draw (thin value)
it_print = 1000;  %Print in the screen every "it_print"-th iteration
constant = 1;  % Set 0:no constant on the FAVAR equation, 1:include constant
% Set which parameters are time-varying. The order is:
TVP_Beta = 1;   % Set 0:constant, 1:time-varying
TVP_Sigma = 1;  % Set 0:constant, 1:time-varying
TVP_Alpha = 1;  % Set 0:constant, 1:time-varying
% Now rotate the factor space as in Bernanke, Boivin and Eliasz (2005)
slowindex = find(slowcode==1)';
xslow = X(:,slowindex);
[~,Fslow0,pvarslow] = pca(xslow);
Fr0 = facrot(F0,Y(:,end),Fslow0);
% Put it all in state-space representation, write obs equ as XY=FY*L+e
XY=[X,Y];   %Tx(N+M)
FY=[Fr0,Y];
% Obtain L (the loadings matrix)
L = (olssvd(XY,FY))';
% Obtain R (the error varaince in the factor equation)
e = XY - FY*L';
R = e'*e./T;
R = diag(diag(R));
R = diag([diag(R);zeros(M,1)]);   %(N+M)x(N+M)
L_RE = L;
R_RE = R;

%% -----------VAR EQUATION-----------
% Generate lagged FY matrix.
ylag = mlag2(FY,plag);
% Form RHS matrix X_t = [1 y_t-1 y_t-2 ... y_t-k] for t=1:T
ylag = ylag(plag+tau+1:T,:);
m = p*constant + plag*(p^2); % m is the number of elements in the state vector
Z = zeros((T-tau-plag)*p,m);
for i = 1:T-tau-plag
    if constant
        ztemp = eye(p);
    else
        ztemp = []; %#ok<UNRCH>
    end
    for j = 1:plag
        xtemp = ylag(i,(j-1)*p+1:j*p);
        xtemp = kron(eye(p),xtemp);
        ztemp = [ztemp xtemp];  %#ok<AGROW>
    end
    Z((i-1)*p+1:i*p,:) = ztemp;
end
% Redefine FAVAR variables y
y = FY(tau+plag+1:T,:)';
% Time series observations
T=size(y,2);

%% -----------PRIORS-----------
% ========= PRIORS ON FACTOR EQUATION:
% Prior on loadings L_i ~ N(0, I), where i = 1,...,N
Li_prvar = 4*eye(p);
% Prior on covariance SIGMA_i ~ iG(a, b), where i = 1,...,N 
alpha = 0.01;
beta = 0.01;
% ========= PRIORS ON VAR EQUATION:
% % To set up training sample prior a la primiceri, use the following subroutine:
% [B_OLS,VB_OLS,A_OLS,sigma_OLS,VA_OLS]= ts_prior(FY,tau,p,plag);
% Otherwise, use non-informative starting values
A_OLS = zeros(numa,1);
B_OLS = zeros(m,1);
VA_OLS = eye(numa);
VB_OLS = eye(m);
sigma_OLS = ones(p,1);
% Set some hyperparameters here (see page 831, end of section 4.1)
k_Q = 0.007;
k_S = 0.1;
k_W = 1;
% We need the sizes of some matrices as prior hyperparameters (see page
% 831 again, lines 2-3 and line 6)
sizeQ = m; % Size of matrix Q
sizeW = p; % Size of matrix W 
sizeS = 1:p; % Size of matrix S
%-------- Now set prior means and variances (_prmean / _prvar)
% B_0 ~ N(B_OLS, 4Var(B_OLS))
B_0_prmean = B_OLS;
B_0_prvar = 4*VB_OLS;
% A_0 ~ N(A_OLS, 4Var(A_OLS))
A_0_prmean = A_OLS;
A_0_prvar = 4*VA_OLS;
% log(sigma_0) ~ N(log(sigma_OLS),I_n)
sigma_prmean = sigma_OLS;
sigma_prvar = 4*eye(p);
% Note that for IW distribution I keep the _prmean/_prvar notation,
% but these are scale and shape parameters...
% Q ~ iWishart(k2_Q*size(subsample)*Var(B_OLS),size(subsample))
Q_prmean = ((k_Q).^2)*(1+sizeQ)*VB_OLS;
Q_prvar = 1 + sizeQ;
% W ~ IG(k2_W*(1+dimension(W))*I_n,(1+dimension(W)))
W_prmean = ((k_W)^2)*ones(p,1);
W_prvar = 2;
% S ~ iWishart(k2_S*(1+dimension(S)*Var(A_OLS),(1+dimension(S)))
S_prmean = cell(p-1,1);
S_prvar = zeros(p-1,1);
ind = 1;
for ii = 2:p
    % S is block diagonal as in Primiceri (2005)
    S_prmean{ii-1} = ((k_S)^2)*(1 + sizeS(ii-1))*VA_OLS(((ii-1)+(ii-3)*(ii-2)/2):ind,((ii-1)+(ii-3)*(ii-2)/2):ind);
    S_prvar(ii-1) = 1 + sizeS(ii-1);
    ind = ind + ii;
end

%% -----------INITIALIZE MATRICES-----------
% Specify covariance matrices for measurement and state equations*/
consQ = 0.0001;
consS = 0.0001;
consH = 0.01;
consW = 0.0001;
Qdraw = consQ*eye(m);
Qchol = sqrt(consQ)*eye(m);
Ht = kron(ones(T,1),consH*eye(p));
Htsd = kron(ones(T,1),sqrt(consH)*eye(p));
Sdraw = consS*eye(numa);
Sblockdraw = cell(p-1,1);
ijc = 1;
for jj=2:p
    Sblockdraw{jj-1} = Sdraw(((jj-1)+(jj-3)*(jj-2)/2):ijc,((jj-1)+(jj-3)*(jj-2)/2):ijc);
    ijc = ijc + jj;
end
Wdraw = consW*ones(p,1);
Btdraw = 0.5*ones(m,T);
Atdraw = zeros(numa,T);
Sigtdraw = zeros(T,p);
sigt = kron(ones(T,1),0.01*eye(p));
mixdrawS = 5*ones(T,p);
Zs = kron(ones(T,1),eye(p));
% Storage matrices for posteriors and stuff
Bt_postmean = zeros(m,T);
At_postmean = zeros(numa,T);
Sigt_postmean = zeros(T,p);
Qmean = zeros(m,m);
Smean = zeros(numa,numa);
Wmean = zeros(p,1);
sigmean = zeros(T,p);
cormean = zeros(T,numa);
kpmean = zeros(1,2+N);
sig2mo = zeros(T,p);
cor2mo = zeros(T,numa);
kp2mo = zeros(1,2+N);
% Storage draws
Btdraw_save = zeros(m,T,nrep/nthin);
Atdraw_save = zeros(numa,T,nrep/nthin);
Sigtdraw_save = zeros(T,p,nrep/nthin);
Qdraw_save = zeros(m,m,nrep/nthin);
Sdraw_save = zeros(numa,numa,nrep/nthin);
Wdraw_save = zeros(p,1,nrep/nthin);
Hdraw_save = zeros([size(Ht), nrep/nthin]);
Ldraw_save = zeros([size(L), nrep/nthin]);

%% -----------IMPULSE RESPONSES:
% Note that impulse response and related stuff involves a lot of storage
% and, hence, put istore=0 if you do not want them
nhor = 41;
%shock = diag([1/100 1/100 1/100 1/100 ]') % diag([zeros(1,p-1) 1/100]');
shock = -diag([zeros(1,p-1) 1/100]'); % in terms of standard deviation, identification is recursive
bigj = zeros(p,p*plag);
bigj(1:p,1:p) = eye(p);
impresp_save = zeros(p,p*nhor, T, nrep/nthin);

%% -----------START SAMPLING -----------
tic; % This is just a timer
disp('Number of iterations');
for irep = 1:nrep + nburn    %  GIBBS iterations starts here
    % Print iterations
    if mod(irep,it_print) == 0
        disp(irep); toc;
    end    
    %=========================================FACTOR (MEASUREMENT) EQUATION: 
    % -----------------------------------------------------------------------------------------
    %  Sample L and R
    % ----------------------------------------------------------------------------------------- 
    % Since the covariance matrix of the error (SIGMA) in this equation is
    % diagonal, we can estimate the parameters equation-by-equation
    for i=1:N
        % Sample L from a Normal distribution. The upper KxK block of L is 
        % the identity matrix, so we sample the rest N-K rows only:
        if i > K
            Li_postvar = inv(inv(Li_prvar) + inv(R(i,i))*FY'*FY);
            Li_postmean = Li_postvar*(inv(R(i,i))*FY'*X(:,i));
            Lidraw = Li_postmean' + randn(1,p)*chol(Li_postvar);
            L(i,1:p) = Lidraw;
        end        
        ed = X(:,i) - FY*L(i,:)';
        % Sample SIGMA(i,i) from iGamma
		R_1 = alpha/2 + T/2;
        R_2 = beta/2 + ((X(:,i) - FY*L(i,:)')'*(X(:,i) - FY*L(i,:)'))/2;
        Ridraw = inv(gamrnd(R_1,1/R_2));
        R(i,i) = Ridraw;
    end  
    %=========================================VAR (STATE) EQUATION:
    % -----------------------------------------------------------------------------------------
    %   STEP I: Sample B(T) from p(B|y,A,Sigma,V) (Drawing coefficient states, pp. 844-845)
    % -----------------------------------------------------------------------------------------
    [Btdrawc,log_likB] = carter_kohn(y,Z,Ht,Qdraw,m,p,T,B_0_prmean,B_0_prvar,TVP_Beta*ones(T,1));
     Btdraw = Btdrawc;
        
    % Now draw the covariance of B(t) which is called Qdraw
    Btemp = Btdraw(:,2:T)' - Btdraw(:,1:T-1)';
    sse_2 = zeros(m,m);
    for i = 1:T-1
        sse_2 = sse_2 + Btemp(i,:)'*Btemp(i,:);
    end

    Qinv = inv(sse_2 + Q_prmean);
    Qinvdraw = wish(Qinv,T - 1 + Q_prvar);
    if TVP_Beta
        Qdraw = inv(Qinvdraw); %covariance of time-varying parameters B(t)
        Qchol = chol(Qdraw);
    else
        Qdraw = zeros(m,m); %#ok<UNRCH>
    end
    %-------------------------------------------------------------------------------------------
    %   STEP II: Draw At from p(At|y,B,Sigma,V) (Drawing coefficient states, p. 845)
    %-------------------------------------------------------------------------------------------
    % Substract from the data y(t), the mean Z x B(t)
    yhat = zeros(p,T);
    for i = 1:T
        yhat(:,i) = y(:,i) - Z((i-1)*p+1:i*p,:)*Btdraw(:,i);
    end
    
    % This part is more tricky, check Primiceri
    % Zc is a [p x p(p-1)/2] matrix defined in (A.2) page 845, Primiceri
    Zc = - yhat(:,:)';
    sigma2temp = exp(Sigtdraw);
    
    Atdraw = [];
    ind = 1;
    for ii = 2:p
        % Draw each block of A(T)
        [Atblockdraw,log_lik2a] = carter_kohn(yhat(ii,:),Zc(:,1:ii-1),sigma2temp(:,ii),...
            Sblockdraw{ii-1},sizeS(ii-1),1,T,A_0_prmean(((ii-1)+(ii-3)*(ii-2)/2):ind,:),A_0_prvar(((ii-1)+(ii-3)*(ii-2)/2):ind,((ii-1)+(ii-3)*(ii-2)/2):ind),TVP_Alpha*ones(T,1));
        Atdraw = [Atdraw ; Atblockdraw]; %#ok<AGROW> % Atdraw is the final matrix of draws of A(t)
        ind = ind + ii;
    end
    
    %=====| Draw S, the covariance of A(t) (from iWishart)
    Attemp = Atdraw(:,2:T)' - Atdraw(:,1:T-1)';
    sse_2 = zeros(numa,numa);
    for i = 1:T-1
        sse_2 = sse_2 + Attemp(i,:)'*Attemp(i,:);
    end
    
    ijc = 1;
    for jj=2:p
        Sinv = inv(sse_2(((jj-1)+(jj-3)*(jj-2)/2):ijc,((jj-1)+(jj-3)*(jj-2)/2):ijc) + S_prmean{jj-1});
        Sinvblockdraw = wish(Sinv,T - 1 + S_prvar(jj-1));
        Sblockdraw{jj-1} = inv(Sinvblockdraw);
        ijc = ijc + jj;
    end

    %------------------------------------------------------------------------------------------
    %   STEP III: Draw diagonal VAR covariance matrix "S_t" elements
    %------------------------------------------------------------------------------------------
    capAt = zeros(p*T,p);
    for i = 1:T
        capatemp = eye(p);
        aatemp = Atdraw(:,i);
        ic=1;
        for j = 2:p
            capatemp(j,1:j-1) = aatemp(ic:ic+j-2,1)';
            ic = ic + j - 1;
        end
        capAt((i-1)*p+1:i*p,:) = capatemp;
    end

    y2 = [];
    for i = 1:T
        ytemps = capAt((i-1)*p+1:i*p,:)*yhat(:,i);
        y2 = [y2  (ytemps.^2)]; %#ok<AGROW>
    end   

    yss = log(y2 + 1e-6)';
    for j=1:p
        [Sigtdraw(:,j) , statedraw(:,j)] = SVRW2(yss(:,j),Sigtdraw(:,j),Wdraw(j,:),sigma_prmean(j),sigma_prvar(j,j),TVP_Sigma);
    end
    sigt = exp(.5*Sigtdraw);
    
    e2 = Sigtdraw(2:end,:) - Sigtdraw(1:end-1,:);
    W1 = W_prvar + T - plag - 1;
    W2 = W_prmean + sum(e2.^2)';
    Winvdraw = gamrnd(W1./2,2./W2);
    Wdrawc = 1./Winvdraw;
    if TVP_Sigma 
        Wdraw = Wdrawc;
    else
        Wdraw = zeros(p,p); %#ok<UNRCH>
    end

    Ht = zeros(p*T,p);
    Htsd = zeros(p*T,p);
    for i = 1:T
        inva = inv(capAt((i-1)*p+1:i*p,:));
        stem = diag(sigt(i,:));
        Hsd = inva*stem;
        Hdraw = Hsd*Hsd';
        Ht((i-1)*p+1:i*p,:) = Hdraw;
        Htsd((i-1)*p+1:i*p,:) = Hsd;
    end
    %------------------------SAVE AFTER-BURN-IN DRAWS-----------------
    if irep > nburn && mod((irep-nburn),nthin)==0 %&& sum(counter)==0
        i_count = i_count + 1;
        Bt_postmean = Bt_postmean + Btdraw;
        At_postmean = At_postmean + Atdraw;
        Sigt_postmean = Sigt_postmean + Sigtdraw;
        Qmean = Qmean + Qdraw;
        ikc = 1;
        for kk = 2:p
            Sdraw(((kk-1)+(kk-3)*(kk-2)/2):ikc,((kk-1)+(kk-3)*(kk-2)/2):ikc)=Sblockdraw{kk-1};
            ikc = ikc + kk;
        end
        Smean = Smean + Sdraw;
        Wmean = Wmean + Wdraw;
        stemp6 = zeros(p,1);
        stemp5 = [];
        stemp7 = [];
        for i = 1:T
            stemp8 = corrvc(Ht((i-1)*p+1:i*p,:));
            stemp7a = [];
            ic = 1;
            for j = 1:p
                if j>1;
                    stemp7a = [stemp7a ; stemp8(j,1:ic)']; %#ok<AGROW>
                    ic = ic+1;
                end
                stemp6(j,1) = sqrt(Ht((i-1)*p+j,j));
            end
            stemp5 = [stemp5 ; stemp6']; %#ok<AGROW>
            stemp7 = [stemp7 ; stemp7a']; %#ok<AGROW>
        end
        sigmean = sigmean + stemp5;
        cormean =cormean + stemp7; 
        sig2mo = sig2mo + stemp5.^2;
        cor2mo = cor2mo + stemp7.^2;
        % save draws
        Btdraw_save(:,:,i_count) = Btdraw;
        Atdraw_save(:,:,i_count) = Atdraw;
        Sigtdraw_save(:,:,i_count) = Sigtdraw;
        Qdraw_save(:,:,i_count) = Qdraw;
        Sdraw_save(:,:,i_count) = Sdraw;
        Wdraw_save(:,:,i_count) = Wdraw;
        Hdraw_save(:,:,i_count) = Ht;
        Ldraw_save(:,:,i_count) = L;
        %------------------------SAVE IMPULSE RESPONSES -----------------
        % Note that Htsd contains the
        % structural error cov matrix
        % Set up things in VAR(1) format as in Lutkepohl page 11
        k = size(Btdraw,1);
        biga = zeros(p*plag,p*plag);
        for j = 1:plag-1
            biga(j*p+1:p*(j+1),p*(j-1)+1:j*p) = eye(p);
        end
        for i = 1:T % Get impulses recurssively for each time period
            bbtemp = Btdraw(p*constant + 1:k,i);
            splace = 0;
            for ii = 1:plag
                for iii = 1:p
                    biga(iii,(ii-1)*p+1:ii*p) = bbtemp(splace+1:splace+p,1)';
                    splace = splace + p;
                end
            end
            % now get impulse responses for 1 through nhor future periods@
            impresp = zeros(p,p*nhor);
            impresp(1:p,1:p) = shock;
            bigai = biga;
            for j = 1:nhor-1
                impresp(:,j*p+1:(j+1)*p) = bigj*bigai*bigj'*shock;
                bigai = bigai*biga;
            end
            impresp_save(:,:,i,i_count) = impresp;
        end
        
    end % END saving after burn-in results 
end %END main Gibbs loop (for irep = 1:nrep+nburn)
%=============================GIBBS SAMPLER ENDS HERE==================================

nrep = i_count;
Bt_postmean = Bt_postmean./nrep;
At_postmean = At_postmean./nrep;
Sigt_postmean = Sigt_postmean./nrep;
Qmean = Qmean./nrep;
Smean = Smean./nrep;
Wmean = Wmean./nrep;
sigmean = sigmean./nrep;
cormean = cormean./nrep;
sig2mo = sig2mo./nrep;
cor2mo = cor2mo./nrep;

toc; % Stop timer and print total time
save("China_gini.mat")

%% -----------impluse response draw figures
% interested variable
interest = 4;
response_v = 3;
[variable_c, nhor_c, Time, mcmc_c] = size(impresp_save);
impluse_interest = impresp_save(response_v, (nhor_c/nhor)*(0:nhor-1)+interest,:,:);

irf=squeeze(prctile(impluse_interest,[50],4));
a = cumsum(irf,1);

x = 1999:0.25:2019.75;
y = 1:21;
[Y,X] = meshgrid(y,x);

mesh(Y,X,cumsum(irf(1:21,:),1)');
