function [s_opt,cnt] = CG_mf_constr(u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,s_opt,sh)
    s_size = size(s_opt);
    s = reshape(s_opt , prod(s_size),1);

    k_op = 1e6;         % Penalty to inforce constraint in the outer minimization
    fd_step = 1e-8;
    fd_method = 'forward';


    alpha_init = 1.0;
    c = 0.1;
    r = 0.5;
    nreset = 25;
    tol = 1e-13;
    
    mnew = mm(s,u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh,k_op);

    % alpha=alpha_init;         % Apparently unnecessary
    mold = 10^100;
    fnew = zeros(prod(s_size),1);     %vectorize?
    hnew = zeros(prod(s_size),1);     %vectorize?
    cnt=0;
    tot_lnsrch = 0;
    while (mold-mnew)>tol*abs(mold)
        mold = mnew;
        fold = fnew;
        hold = hnew;
        
        fnew = numericalDerivativeParallel(@(sv) mm(sv,u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh,k_op), s, fd_method, fd_step);
        
        if mod(cnt,nreset)==0
            hnew = -fnew;
        else
            beta = (fnew'*fnew)/(fold'*fold);
            hnew = -fnew + max(beta,0)*hold;
        end

        cnt_lnsrch = 0;
        % fprintf('SARTING LINESEARCH')


        % Classic line-search with Armijo rule

        alpha = 1/r*alpha_init;
        mx = 10^100;
        sx = s;
        while mx > mnew + c*alpha*(hnew'*fnew) && norm(sx-s+r*alpha*hnew) > 1e-15
            alpha = r*alpha;
            sx = s + alpha*hnew;
            mx = mm(sx,u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh,k_op);
            if alpha<1e-15
                0;
            end
            cnt_lnsrch = cnt_lnsrch + 1;
        end



        cnt_lnsrch
        mnew = mx;
        s = sx;
        cnt = cnt + 1;
        tot_lnsrch = tot_lnsrch + cnt_lnsrch;

    end

    
    fprintf('iters = %d \t tot_lnsrch = %d \n', cnt, tot_lnsrch);
    s_opt = reshape(s,s_size);

end

function m_constr = mm(s_opt,u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh,k_op)
    s_opt = reshape(s_opt,[],2);
    [c_ineq,c_eq] = nonlcon(s_opt,u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh);
    c_in = max(c_ineq,0);
    obj = mgt2(s_opt,u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh);
    m_constr = obj + 1/2*k_op*(c_eq'*c_eq) + 1/2*k_op*(c_in'*c_in);
end

function dF = numericalDerivative(func, x, method, h)
    % numericalDerivative: Compute the numerical derivative of a function
    %
    % Inputs:
    %   func - function handle (should take a vector as input)
    %   x - vector at which to evaluate the derivative
    %   method - 'central', 'forward', or 'backward' for finite difference
    %   h - step size for finite differences
    %
    % Output:
    %   dF - numerical derivative (Jacobian matrix) of func at x

    if nargin < 4
        h = 1e-7; % default step size
    end

    n = numel(x); % number of elements in x
    dF = zeros(n, 1); % initialize Jacobian matrix

    if ismember(method,{'forward','backward'})
        f0 = func(x);
    end

    for i = 1:n
        xTemp = x;
        
        if strcmp(method, 'central')
            xTemp(i) = x(i) + h;
            f1 = func(xTemp);
            xTemp(i) = x(i) - h;
            f2 = func(xTemp);
            dF(i) = (f1 - f2) / (2*h);
        elseif strcmp(method, 'forward')
            f1 = f0;
            xTemp(i) = x(i) + h;
            f2 = func(xTemp);
            dF(i) = (f2 - f1) / h;
        elseif strcmp(method, 'backward')
            xTemp(i) = x(i) - h;
            f1 = func(xTemp);
            f2 = f0;
            dF(i) = (f2 - f1) / h;
        else
            error('Invalid method. Choose central, forward, or backward.');
        end
    end
end

function dF = numericalDerivativeParallel(func, x, method, h)
    % numericalDerivativeParallel: Compute the numerical derivative of a function in parallel
    %
    % Inputs:
    %   func - function handle (should take a vector as input)
    %   x - vector at which to evaluate the derivative
    %   method - 'central', 'forward', or 'backward' for finite difference
    %   h - step size for finite differences
    %
    % Output:
    %   dF - numerical derivative (Jacobian matrix) of func at x

    if nargin < 4
        h = 1e-7; % default step size
    end

    n = numel(x); % number of elements in x
    dF = zeros(n, 1); % initialize Jacobian matrix

    if ismember(method,{'forward','backward'})
        f0 = func(x);
    end

    parfor i = 1:n
        xTemp = x;
        % fprintf('started %s\n',int2str(i));
        if strcmp(method, 'central')
            xTemp(i) = x(i) + h;
            f1 = func(xTemp);
            xTemp(i) = x(i) - h;
            f2 = func(xTemp);
            dF(i) = (f1 - f2) / (2*h);
        elseif strcmp(method, 'forward')
            f1 = f0;
            xTemp(i) = x(i) + h;
            f2 = func(xTemp);
            dF(i) = (f2 - f1) / h;
        elseif strcmp(method, 'backward')
            xTemp(i) = x(i) - h;
            f1 = func(xTemp);
            f2 = f0;
            dF(i) = (f2 - f1) / h;
        else
            error('Invalid method. Choose central, forward, or backward.');
        end
        % fprintf('finished %s\n',int2str(i));

    end
end