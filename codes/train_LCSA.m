function [B] = train_LCSA(L1,XKTrain,YKTrain,LTrain,param)
    % X,Y=n*d,  L,L1=n*c
    % parameters
    nbits = param.nbits;
    n = size(LTrain,1);
    c = size(LTrain,2);

    % initization
    B = sign(randn(nbits, n)); B(B==0) = -1;    %r*n
    V = randn(nbits, n);                        %r*n
    %S = L1*L1';     %n*n
    X = XKTrain';   %d*n
    Y = YKTrain';   %d*n
    L = LTrain';    %c*n
    L1 = L1';       %c*n
    C = L1*L1';     %c*c
    Lnew = L;       %c*n
    %Lnew = randn(c,n);
    A = randn(c, nbits);
    D = randn(c, nbits);
    Z = randn(nbits, nbits);

    %[~,~,M] = cLLR_k(L,param.k);
    N = param.k;                            % top-N 数量（复用 k 参数，或改名 topN）

    [~, idx] = sort(C, 2, 'descend');       % 每行降序排列
    mask = false(c, c);
    for i = 1:c
        mask(i, idx(i, 1:N)) = true;        % 每行保留前 N 个
     end
    G = C .* mask;                          % 稀疏化
    G = (G + G') / 2;                       % 对称化
    P = diag(sum(G, 2));                    % 度矩阵
    M = P - G;                              % Laplacian

    %%iteration start
    for i = 1:param.max_iter
        fprintf('iteration %3d\n', i);
        
        % update U
        Ux = (X*V')/n;
        Uy = (Y*V')/n;

        %update Lnew
        temp1 = param.alpha*M + (param.beta + param.theta)*eye(c);
        temp2 = param.beta*L + param.theta*A*Z*V;
        Lnew = temp1\temp2;
        clear temp1 temp2

        %update A
        temp1 = param.delta*C*D + param.omega*D + param.theta*Lnew*V'*Z';
        temp2 = param.delta*(D'*D) + param.omega*eye(nbits) + n*param.theta*(Z*Z');
        A = temp1/temp2;
        clear temp1 temp2

        %update D
        temp1 = param.delta*C*A + param.omega*A;
        temp2 = param.delta*(A'*A) + param.omega*eye(nbits);
        D = temp1/temp2;
        clear temp1 temp2
        
        % update V
        H = ((B*L1')*L1) + param.gamma1*(Ux'*X) + param.gamma2*(Uy'*Y) + param.theta*Z'*A'*Lnew;
        %Temp = Z - 1/n*ones(n,1)*(ones(1,n)*Z);
        %[P,Lmd,Q] = svd(Temp);
        %idx = (diag(Lmd)>1e-6);
        %Q = Q(:,idx); Q_ = orth(Q(:,~idx));
        %P = P(:,idx); P_ = orth(P(:,~idx));
        %V = sqrt(n)*[P P_]*[Q Q_]';
        Temp = H*H'-1/n*(H*ones(n,1)*(ones(1,n)*H'));
        [~,Lmd,QQ] = svd(Temp); clear Temp
        idx = (diag(Lmd)>1e-6);
        Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
        P = (H'-1/n*ones(n,1)*(ones(1,n)*H')) *  (Q / (sqrt(Lmd(idx,idx))));
        P_ = orth(randn(n,nbits-length(find(idx==1))));
        V = sqrt(n)*[Q Q_]*[P P_]';
        
        %update Z
        temp1 = n*param.theta*(A'*A) + param.eta*eye(nbits);
        temp2 = param.theta*A'*Lnew*V';
        Z = temp1\temp2;
        clear temp1 temp2

        % update B
        B = sign(((V*L1')*L1));
        

    end

    final_B = sign(B);
    final_B(final_B==0) = -1;
    B = final_B;

end