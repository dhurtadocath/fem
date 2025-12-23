function [u,m,iter] = NEWTON(FUN,u,fr)
    tol=1e-10;
    maxiter = 100;

    [m,f,K] = FUN(u);
    
    condK = condest(K);

    upre=u;
    res=10^5;
    iter = 1;
    di = setdiff(1:numel(f),fr)';
    while res>tol && iter<maxiter && condK<1e20
        dua = u(di,1)-upre(di);
        Kba = K(fr,di);
        Kbb = K(fr,fr);
        finb= f(fr,1);

        dub = linsolve(Kbb,-finb-Kba*dua);
        u(fr) = u(fr) + dub;
        [m,f,K] = FUN(u);
        upre=u;
        condK = condest(K);

        iter=iter+1;
        res=norm(f);
    end

    if condK>1e20
        fprintf("Matrix is ILL!!! (cond = %d) \n",condK);
        u = NaN;
    end
    % if iter>7
    %     iter
    % end
end