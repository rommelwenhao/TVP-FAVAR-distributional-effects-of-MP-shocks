%% Setting Environment
clear all;
clc;
addpath('1. functions');
rng(50)

clear 
randn('state',sum(100*clock)); %#ok<*RAND>
rand('twister',sum(100*clock));
%-----------------------------LOAD DATA------------------------------------
% Load Korobilis (2008) quarterly data
% load data used to extract factors
load xdata.dat;
% load data on inflation, unemployment and interest rate 
load ydata.dat;
% load transformation codes (see file transx.m)
load tcode.dat;
% load the slow or fast moving variables codes (see Bernanke, Boivin and
% Eliasz, 2005, QJE)
load slowcode.dat;
% load the file with the dates of the data (quarters)
load yearlab.dat;
% load the file with the names of the variables. 
% namesX.mat contains only the short codes of the data. For a complete
% description see the attached excel file.
load namesX.mat;

% Transform data to be approximately stationary
for i_x = 1:size(xdata,2)   % Transform "X"
    xtempraw(:,i_x) = transx(xdata(:,i_x),tcode(i_x)); %#ok<AGROW>
end

% Correct size after stationarity transformation
xdata = xtempraw(3:end,:);
ydata = ydata(3:end,:);
yearlab = yearlab(3:end);

% Demean and standardize data (needed to extract Principal Components)
stddata = [std(xdata) ones(1,3)];
t1 = size(xdata,1);    % time series observations of xdata
t2 = size(ydata,1);    % time series dimension of ydata
stdffr = std(ydata(:,3));      % standard deviation of the Fed Funds Rate
xdata = (xdata - repmat(mean(xdata,1),t1,1))./repmat(std(xdata),t1,1);
ydata = (ydata - repmat(mean(ydata,1),t2,1))./repmat(std(ydata),t2,1);

% Define X and Y matrices
X = xdata;   % X contains the 'xdata' which are used to extract factors.
Y = ydata(:,:); % Y contains inflation, unemployment and interest rate
% NamesXY has the short codes of X and Y
namesXY = [namesX ; 'Inflation' ; 'Unemployment'; 'Fed_funds'];

% Number of observations and dimension of X and Y
T=size(Y,1); % T time series observations
N=size(X,2); % N series from which we extract factors
M=size(Y,2); % and M(=1) series for the federal funds rate
%----------------------------PRELIMINARIES---------------------------------
% Set some Gibbs - related preliminaries
nrep = 20000;  % Number of replications
nburn = 10000;   % Number of burn-in-draws
nthin = 1;   % Consider every thin-th draw (thin value)
it_print = 100;  %Print in the screen every "it_print"-th iteration
constant = 1;  % Set 0:no constant on the FAVAR equation, 1:include constant

% Set which parameters are time-varying. The order is:
% 1. Beta: mean (auto)-regression coefficients
% 2. Sigma: Log volatilities
% 3. Alpha: Covariance elements
TVP_Beta = 1;   % Set 0:constant, 1:time-varying
TVP_Sigma = 1;  % Set 0:constant, 1:time-varying
TVP_Alpha = 1;  % Set 0:constant, 1:time-varying

% Number of factors & lags:
K = 2;               % Number of Factors
tau = 0;             % tau is the size of the training sample
p = K+M;             % p is the dimensionality of [Factors, Y]
plag = 2;            % plag is number of lags in the VAR part
numa = p*(p-1)/2;    % numa is the number of elements of At
i_count = 0;         % just a counter for saved draws
% ============================| FACTOR EQUATION |==========================
% first step - extract PC from X
[F0,Lf]=extract(X,K);

% Now rotate the factor space as in Bernanke, Boivin and Eliasz (2005)
slowindex = find(slowcode==1)';
xslow = X(:,slowindex);
[Fslow0,Lfslow0] = extract(xslow,K);
Fr0 = facrot(F0,Y(:,end),Fslow0);

% % regress X on F0 and Y, obtain loadings
% Lfy=olssvd(X(:,K+1:N),[F0 Y])';     % upper KxM block of Ly set to zero
% Lf=[Lf(1:K,:);Lfy(:,1:K)];
% Ly=[zeros(K,M);Lfy(:,K+1:K+M)];
% 
% % transform factors and loadings for LE normalization
% [ql,rl]=qr(Lf');
% Lf=rl;  % do not transpose yet, is upper triangular
% F0=F0*ql;
% 
% % need identity in the first K columns of Lf, call them A for now
% A=Lf(:,1:K);
% Lf=[eye(K),inv(A)*Lf(:,(K+1):N)]';
% Fr0=F0*A;

% Put it all in state-space representation, write obs equ as XY=FY*L+e
XY=[X,Y];   %Tx(N+M)
FY=[Fr0,Y];
% Obtain L (the loadings matrix)
L = (olssvd(XY,FY))';
%L=[Lf Ly;zeros(M,K),eye(M)]; 

% Obtain R (the error varaince in the factor equation)
e = XY - FY*L';
R = e'*e./T;
R = diag(diag(R));
R = diag([diag(R);zeros(M,1)]);   %(N+M)x(N+M)

L_RE=L;
R_RE=R;
% ================================| VAR EQUATION |=========================
% Generate lagged FY matrix.
ylag = mlag2(FY,plag);
% Form RHS matrix X_t = [1 y_t-1 y_t-2 ... y_t-k] for t=1:T
ylag = ylag(plag+tau+1:T,:);

m = p*constant + plag*(p^2); % m is the number of elements in the state vector
% Create X_t matrix as in Primiceri equation (4). Of course I have reserved
% "X" for the data I use to extract factors, hence name this matrix "Z".
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
yearlab = yearlab(tau+plag+1:T);
% Time series observations
T=size(y,2);
%-------------------------------- PRIORS ----------------------------------
%========= PRIORS ON FACTOR EQUATION:
% Prior on loadings L_i ~ N(0, I), where i = 1,...,N
Li_prvar = 4*eye(p);

% Prior on covariance SIGMA_i ~ iG(a, b), where i = 1,...,N 
alpha = 0.01;
beta = 0.01;

%========= PRIORS ON VAR EQUATION:
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


%========= INITIALIZE MATRICES:
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

%========= IMPULSE RESPONSES:
% Note that impulse response and related stuff involves a lot of storage
% and, hence, put istore=0 if you do not want them
istore = 1;
if istore == 1;
    % Impulse response horizon
    nhor = 21;
    shock_init = diag([zeros(1,p-1) 1/stdffr]'); % in terms of standard deviation, identification is recursive
    imp75 = zeros(nrep,N+M,nhor);
    imp81 = zeros(nrep,N+M,nhor);
    imp96 = zeros(nrep,N+M,nhor);
    imp06 = zeros(nrep,N+M,nhor);
    bigj = zeros(p,p*plag);
    bigj(1:p,1:p) = eye(p);
end
%--------------------------- END OF PRELIMINARIES -------------------------

%====================================== START SAMPLING ========================================
%==============================================================================================
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

    % 1. Accept draw   
    Btdraw = Btdrawc;
    
%     % or 2. check for stationarity
%     ctemp1 = zeros(p,p*plag);
%     counter = [];
%     restviol=0;
%     for i = 1:T;
%         BBtempor = Btdrawc(p*constant+1:end,i);
%         BBtempor = reshape(BBtempor,p*plag,p)';
%         ctemp1 = [BBtempor; eye(p*(plag-1)) zeros(p*(plag-1),p)];
%         if max(abs(eig(ctemp1)))>0.99999;
%             restviol=1;
%             counter = [counter ; restviol]; %#ok<AGROW>
%         end
%     end
%     %if they haven't been rejected, then accept them, else keep old draw
%     if sum(counter)==0
%         Btdraw = Btdrawc;
%         disp('I found a keeper!');
%     end
    
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
    %------------------------SAVE AFTER-BURN-IN DRAWS AND IMPULSE RESPONSES -----------------
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
        
        %----------------- Impulse response analysis:
        if istore==1;
            % Note that Htsd contains the
            % structural error cov matrix
            % Set up things in VAR(1) format as in Lutkepohl page 11
            k = size(Btdraw,1);
            biga = zeros(p*plag,p*plag);
            for j = 1:plag-1
                biga(j*p+1:p*(j+1),p*(j-1)+1:j*p) = eye(p);
            end

            for i = 1:T %Get impulses recurssively for each time period
                bbtemp = Btdraw(p*constant + 1:k,i);
                splace = 0;
                for ii = 1:plag
                    for iii = 1:p
                        biga(iii,(ii-1)*p+1:ii*p) = bbtemp(splace+1:splace+p,1)';
                        splace = splace + p;
                    end
                end
                
                % st dev matrix for structural VAR
                Hsd = chol(Ht((i-1)*p+1:i*p,1:p))';
                d = diag(diag(Hsd));
                shock = inv(d)*Hsd;
                %shock = Hsd*shock_init;
                %shock=eye(p);
                
                %now get impulse responses for 1 through nhor future periods@
                impresp = zeros(p,p*nhor);
                impresp(1:p,1:p) = shock;
                bigai = biga;
                for j = 1:nhor-1
                    impresp(:,j*p+1:(j+1)*p) = bigj*bigai*bigj'*shock;
                    bigai = bigai*biga;
                end
                if yearlab(i,1) == 1975;
                    imp75_m = zeros(p,nhor);
                    jj=0;
                    for ij = 1:nhor
                        jj = jj + p;
                        imp75_m(:,ij) = impresp(:,jj);
                    end
                    imp75(i_count,:,:) = L_RE*imp75_m;                    
                end
                if yearlab(i,1) == 1981.75;
                    imp81_m = zeros(p,nhor);
                    jj=0;
                    for ij = 1:nhor
                        jj = jj + p;
                        imp81_m(:,ij) = impresp(:,jj);
                    end
                    imp81(i_count,:,:) = L_RE*imp81_m;
                end                
                if yearlab(i,1) == 1996;
                    imp96_m = zeros(p,nhor);
                    jj = 0;
                    for ij = 1:nhor
                        jj = jj + p;
                        imp96_m(:,ij) = impresp(:,jj);
                    end
                    imp96(i_count,:,:) = L_RE*imp96_m;
                end
                if yearlab(i,1) == 2006.5;
                    imp06_m = zeros(p,nhor);
                    jj = 0;
                    for ij = 1:nhor
                        jj = jj + p;
                        imp06_m(:,ij) = impresp(:,jj);
                    end
                    imp06(i_count,:,:) = L_RE*imp06_m;
                end
            end %END geting impulses for each time period
        end %END the impulse response calculation section   

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

if istore == 1;       
    scale=repmat(stddata',[1 3 21]);
    scale=permute(scale,[2 1 3]);
    
    imp75 = permute(imp75,[2 3 1]);    
    imp81 = permute(imp81,[2 3 1]);    
    imp96 = permute(imp96,[2 3 1]); 
    imp06 = permute(imp06,[2 3 1]);
    
    for i=1:N
        if tcode(i)==4
            imp75(i,:,:) = exp(imp75(i,:,:))-ones(1,nhor,nrep/nthin);
            imp81(i,:,:) = exp(imp81(i,:,:))-ones(1,nhor,nrep/nthin);
            imp96(i,:,:) = exp(imp96(i,:,:))-ones(1,nhor,nrep/nthin);
            imp06(i,:,:) = exp(imp06(i,:,:))-ones(1,nhor,nrep/nthin);
        elseif tcode(i)==5
            imp75(i,:,:)=exp(cumsum(imp75(i,:,:),2))-ones(1,nhor,nrep/nthin);
            imp81(i,:,:)=exp(cumsum(imp81(i,:,:),2))-ones(1,nhor,nrep/nthin);
            imp96(i,:,:)=exp(cumsum(imp96(i,:,:),2))-ones(1,nhor,nrep/nthin);
            imp06(i,:,:)=exp(cumsum(imp06(i,:,:),2))-ones(1,nhor,nrep/nthin);
        end
    end
    imp75 = permute(imp75,[3 1 2]);
    imp81 = permute(imp81,[3 1 2]);    
    imp96 = permute(imp96,[3 1 2]); 
    imp06 = permute(imp06,[3 1 2]);
    
    % Set quantiles from the posterior density of the impulse responses
    qus = [.10, .5, .90];
    imp75XY=squeeze(quantile(imp75,qus));
    imp81XY=squeeze(quantile(imp81,qus));
    imp96XY=squeeze(quantile(imp96,qus));
    imp06XY=squeeze(quantile(imp06,qus));
    
    
    %============================| PLOTS |=================================
    %---Plot I: Standard deviations of residuals of Inflation, Unemployment and Interest Rates
    figure
    set(0,'DefaultAxesColorOrder',[0 0 0],...      
        'DefaultAxesLineStyleOrder','-|.|-')    
    subplot(4,1,1)
    plot(yearlab,sigmean(:,1:K))
    title(['Posterior mean of the standard deviation of residuals of the ' num2str(K) ' Factors'])
    xlim([yearlab(1) yearlab(end)])
    legend('Factor 1','Factor 2') 
    subplot(4,1,2)
    plot(yearlab,sigmean(:,K+1))
    title('Posterior mean of the standard deviation of residuals in Inflation equation')
    xlim([yearlab(1) yearlab(end)])    
    subplot(4,1,3)
    plot(yearlab,sigmean(:,K+2))
    title('Posterior mean of the standard deviation of residuals in Unemployment equation')
    xlim([yearlab(1) yearlab(end)])    
    subplot(4,1,4)
    plot(yearlab,sigmean(:,K+3))
    title('Posterior mean of the standard deviation of residuals in Interest Rate equation')
    xlim([yearlab(1) yearlab(end)])
    %-------------------------
    
    %---Plot II: impulse responses of inflation, unemployment, interest
    figure       
    set(0,'DefaultAxesColorOrder',[0 0 0],...
        'DefaultAxesLineStyleOrder','--|-|--')
    subplot(3,3,1)
    plot(1:nhor,squeeze(imp75XY(:,end-2,:)))
    hold;
    plot(zeros(1,nhor),'-')
    title('Impulse response of inflation, 1975:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)
    subplot(3,3,2)
    plot(1:nhor,squeeze(imp75XY(:,end-1,:)))
    hold;
    plot(zeros(1,nhor),'-')
    title('Impulse response of unemployment, 1975:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,3)
    plot(1:nhor,squeeze(imp75XY(:,end,:)))
    hold;
    plot(zeros(1,nhor),'-')
    title('Impulse response of interest rate, 1975:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,4)
    plot(1:nhor,squeeze(imp81XY(:,end-2,:)))
    hold;
    plot(zeros(1,nhor),'-')
    title('Impulse response of inflation, 1981:Q3')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,5)
    plot(1:nhor,squeeze(imp81XY(:,end-1,:)))
    hold;
    plot(zeros(1,nhor),'-')
    title('Impulse response of unemployment, 1981:Q3')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,6)
    plot(1:nhor,squeeze(imp81XY(:,end,:)))
    hold;
    plot(zeros(1,nhor),'-')
    title('Impulse response of interest rate, 1981:Q3')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,7)
    plot(1:nhor,squeeze(imp96XY(:,end-2,:)))
    hold;
    plot(zeros(1,nhor),'-')
    title('Impulse response of inflation, 1996:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,8)
    plot(1:nhor,squeeze(imp96XY(:,end-1,:)))
    hold;
    plot(zeros(1,nhor),'-')
    title('Impulse response of unemployment, 1996:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)
    subplot(3,3,9)
    plot(1:nhor,squeeze(imp96XY(:,end,:)))
    hold;
    plot(zeros(1,nhor),'-')
    title('Impulse response of interest rate, 1996:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)
    %-------------------------
    
    %---Plot III: impulse responses of other variables
    figure       
    set(0,'DefaultAxesColorOrder',[0 0 0],...
        'DefaultAxesLineStyleOrder','-d|-*|-o')
    % I will plot only 12 out of the 115 variables here
    var_numbers = [2  9 10 28 42 46 77 91 92 108 109 111];
    % These variables have the following short codes:
    var_names = namesXY(var_numbers);
    

    % Each figure has the median response of each variable in three
    % different periods. No space to plot quantiles of responses. Note that
    % as the number of factors increases, estimation error also increases
    % and the credible intervals of the responses become larger.
    for i=1:12
        subplot(4,3,i)   
        plot(1:nhor,squeeze(imp75XY(2,var_numbers(i),:)),1:nhor,squeeze(imp81XY(2,var_numbers(i),:)),1:nhor,squeeze(imp96XY(2,var_numbers(i),:)))
        hold;
        plot(zeros(1,nhor),'-')
        title(['Impulse response of ' var_names(i)])   
        xlim([1 nhor])
        set(gca,'XTick',0:3:nhor)   
        legend('1975:Q1','1981:Q3','1996:Q1')   
    end
    %-------------------------
    
end

clc;
toc; % Stop timer and print total time