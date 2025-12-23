function [s_opt,cnt,unew,m_out,f_out] = CG_mf_constr_ParLnSrch(u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,s_opt,sh,k_op,tol,fd_step)
    s_size = size(s_opt);
    s = reshape(s_opt , prod(s_size),1);


    % k_op = 1e6;         % Penalty to inforce constraint in the outer minimization
    % k_op = 1e3;         % Penalty to inforce constraint in the outer minimization
    % k_op = 1;         % Penalty to inforce constraint in the outer minimization
    % fd_step = 1e-12;
    % fd_method = 'forward';
    fd_method = 'central';


    % alpha_init = 0.001;
    c = 0.5;
    r = 0.5;
    nreset = 10;
    % tol = 1e-13;
    
    [mnew,unew] = mm(s,u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh,k_op,false);

    % alpha=alpha_init;         % Apparently unnecessary
    mold = 10^100;
    % fnew = zeros(prod(s_size),1);     %vectorize?
    fnew = 10^50*ones(prod(s_size),1);     %vectorize?
    hnew = zeros(prod(s_size),1);     %vectorize?
    cnt=0;
    tot_lnsrch = 0;
    fold = 10^100;

    alpha=1;

    reset = false;
    steepest_descent_counter = 0;
    while (mold-mnew)>tol*abs(mold) || reset==true
    % while (norm(fold-fnew)>tol*norm(fold) && norm(fold)>1e-5) && (mold-mnew)>tol*abs(mold)
        mold = mnew;
        fold = fnew;
        h_old = hnew;
        
        fprintf('Computing Derivative\n')

        % [c_in,c_eq,~] = nonlcon(reshape(s,[],2),unew,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh);
        % % act_constr = (c_in>0).*(c_eq>0);  % Array with 1's for nodes with some active constraint
        % act_constr = (c_in>0)|(c_eq>0);  % Array with 1's for nodes with some active constraint
        % ina_constr = ~act_constr;
        % % fd_actin = 1e-10*act_constr + 1e-12*ina_constr;
        % fd_actin = fd_step*act_constr + fd_step/100*ina_constr;

        % dFst = study_numericalDerivative(@(sv) mm(sv,unew,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh,k_op), s, fd_method, cnt);
        fnew = numericalDerivativeParallel(@(sv) mm(sv,unew,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh,k_op,false), s, fd_method, fd_step);
        % fnew = numericalDerivativeParallel(@(sv) mm(sv,unew,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh,k_op), s, fd_method, fd_actin);
        
        % if (cnt>1 && norm(fnew)/norm(fold)>1e3)
        %     break
        % end

        if reset
            hnew = -fnew;
            reset=false;
        else
            % beta = (fnew'*fnew)/(fold'*fold);                   % (  I)
            beta = (fnew'*(fnew-fold))/(fold'*fold);            % ( II)
            % beta = (fnew'*(fnew-fold))/(hold'*(fnew-fold));     % (III)
            % beta = (fnew'*fnew)/(hold'*(fnew-fold));            % ( IV)

            hnew = -fnew + max(beta,0)*h_old;
        end
        norm(hnew)

        cnt_lnsrch = 0;
        fprintf('SARTING LINESEARCH\n')
        alpha_init = min( [1.0 , 0.001/max(abs(hnew)),max(-mnew/(c*(hnew'*fnew)),1e-10)])    % numerator is the maximum a node should move in each linesearch
        % alpha_init = min( [10*alpha , 0.001/max(abs(hnew)),-mnew/(c*(hnew'*fnew))])    % numerator is the maximum a node should move in each linesearch


        % ALPHAS = [];
        % MMMS = [];
        % ARMJ = [];
        % ARMJ1 = [];
        % SX= zeros(100,6,2);

        % Classic line-search with Armijo rule
        alpha = 1/r*alpha_init;
        mx = 10^100;
        sx = s;
        % while mx > mnew + c*alpha*(hnew'*fnew) && norm(sx-s+r*alpha*hnew) > 1e-15
        ux = unew;
        % while mx > mnew + c*alpha*(hnew'*fnew) && norm(sx-s+r*alpha*hnew) > 1e-15

        while mx > mnew + c*alpha*(hnew'*fnew) && alpha>1e-25
        % while mx > mnew + c*alpha*(hnew'*fnew)
            alpha = r*alpha;
            sx = s + alpha*hnew;
            [mx,ux] = mm(sx,ux,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh,k_op,true);
            
            % for plotting purposes onyl! don't use the minimum of these.
            % Trust Armijo instead.
            % ALPHAS = [ALPHAS,alpha];
            % MMMS = [MMMS,mx];
            % ARMJ = [ARMJ,max(mnew + c*alpha*(hnew'*fnew),0.0)];
            % ARMJ1 = [ARMJ1,mnew + c*alpha*(hnew'*fnew)];

            cnt_lnsrch = cnt_lnsrch + 1;

            % SX(cnt_lnsrch,:,:) = reshape(sx,6,2);


        end
        fprintf('ALPHA: %0.4e\n',alpha)

        % if cnt_lnsrch>0
        %     figure(2)
        %     plot(SX(1:cnt_lnsrch,:,1),SX(1:cnt_lnsrch,:,2))
        %     hold on,
        %     scatter(sh(actives,1),sh(actives,2),"x")
        %     scatter(SX(cnt_lnsrch,:,1),SX(cnt_lnsrch,:,2),'o')
        %     text(SX(cnt_lnsrch,:,1),SX(cnt_lnsrch,:,2),int2str(cnt))
        %     hold on,
        %     xlim([0.45,0.55])
        %     ylim([0.45,0.55])
        %     drawnow
        % 
        %     figure(3)
        %     loglog(ALPHAS,MMMS,ALPHAS,ARMJ)
        %     drawnow
        %     figure(4)
        %     plot(ALPHAS,MMMS,ALPHAS,ARMJ1)
        %     drawnow
        % end

        s = sx;
        unew = ux;
        mnew = mx;


        % if ((mold-mnew)<=1e-6*abs(mold)) && (k_op<1e3)
        %     k_op=k_op*2
        % end



        % alpha_a = 0.0;
        % alpha_b = 1.0;
        % % exclude = [];
        % ms = zeros(ndpls+2,1);
        % alphas = zeros(ndpls+2,1);
        % m_min = 10^100;
        % us = zeros(ndpls+2,length(u));
        % % while norm((alpha_b-alpha_a)*hnew)>1e-8
        % if norm(fnew)>1e-6
        %     tol_ls = 1e-8;
        % else
        %     tol_ls = 1e-10;
        % end
        % 
        % alphas(2,end-1) = linspace(alpha_a,alpha_b,ndpls);
        % d_alpha = alphas(3)-alphas(2);
        % alphas(1)   = alpha_a-d_alpha;
        % alphas(end) = alpha_b+d_alpha;
        % while norm((alpha_b-alpha_a)*hnew)>tol_ls || m_min>mnew
        % % while m_min>mnew
        %     % parfor i = setdiff(1:ndpls,exclude)
        %     parfor i = 1:ndpls
        %         fprintf('started lnSrch: %s\n',int2str(i))
        %         alpha = alphas(i+1)
        %         [ms(i+1),us(i,:)] = mm( s+alpha*hnew ,u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh,k_op);
        %         fprintf('ended lnSrch: %s\n',int2str(i))
        % 
        %     end
        %     [m_min,idx_min] = min(ms);
        %     u_min = us(idx_min,:);
        %     if idx_min == 1
        %         alpha_a = alphas(1);
        %         alpha_b = alphas(2);
        %         ms(ndpls) = ms(2);
        %         us(ndpls,:) = us(2,:);
        %     elseif idx_min == ndpls
        %         alpha_a = alphas(ndpls-1);
        %         alpha_b = alphas(ndpls);
        %         ms(1) = ms(ndpls-1);
        %         us(1,:) = us(ndpls-1,:);
        % 
        %     else
        %         alpha_a = alphas(idx_min-1);
        %         alpha_b = alphas(idx_min+1);
        %         ms(1) = ms(idx_min-1);
        %         ms(ndpls) = ms(idx_min+1);
        %         us(1,:) = us(idx_min-1,:);
        %         us(ndpls,:) = us(idx_min+1,:);
        % 
        %     end
        %     cnt_lnsrch = cnt_lnsrch + 1;
        %     % exclude = [1,ndpls];
        %     alpha_full = linspace(alpha_a,alpha_b,ndpls+2);
        %     alphas = alpha_full(2:end-1);
        % 
        % end
        % 
        % cnt_lnsrch
        % mnew = m_min;
        % unew = u_min';
        % s = s+alphas(idx_min)*hnew;
        % cnt = cnt + 1;
        % tot_lnsrch = tot_lnsrch + cnt_lnsrch;


        cnt = cnt + 1;
        tot_lnsrch = tot_lnsrch + cnt_lnsrch;

        if ((mold-mnew)<=tol*abs(mold) || mod(cnt,nreset)==0)&&steepest_descent_counter<3
            if norm(fnew)>1
            reset = true;
            % [c_in,c_eq,~] = nonlcon(reshape(s,[],2),unew,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh);
            % act_constr = (c_in>0).*(c_eq);  % Array with 1's for nodes with some active constraint
            % ina_constr = ~act_constr;
            % fd_actin = 1e-10*act_constr + 1e-12*ina_constr;
            % fd_step = fd_actin


            steepest_descent_counter = steepest_descent_counter + 1;
            fprintf("mnew==mold but |f| too high. Trying reset h=-f \n")
            end
        end
        


    end

    if ~(norm(fold-fnew)>tol*norm(fold))
        fprintf('cond1: norm(fold-fnew)>tol*norm(fold) no longer fulfilled\n')
    end
    if ~(norm(fold)>1e-5)
        fprintf('cond2: norm(fold)>1e-5  no longer fulfilled\n')
    end
    if ~((mold-mnew)>tol*abs(mold))
        fprintf('cond3: (mold-mnew)>tol*abs(mold)  no longer fulfilled\n')
    end
    fprintf('iters = %d \t tot_lnsrch = %d \n', cnt, tot_lnsrch);
    s_opt = reshape(s,s_size);

    m_out = mnew;
    f_out = norm(hnew);

end

function [m_constr,u_constr] = mm(s_opt,u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh,k_op,print_stuff)
    c_bound = max(abs(s_opt-0.5)-0.5,0);
    s_opt = reshape(s_opt,[],2);
    [c_ineq,c_eq0,u_constr] = nonlcon(s_opt,u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh,print_stuff);
    % c_ineq'
    c_in = max(c_ineq,0);
    c_eq = max(c_eq0,0);
    % c_min=-1e-3;
    % c_max= 1e-3;
    % c_eq = max(abs(c_eq0-(c_max+c_min)/2)-(c_max-c_min)/2,0);
    obj = mgt2(s_opt,u_constr,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh);
    % m_constr = obj + 1/2*k_op*(c_eq'*c_eq) + 1/2*k_op*(c_in'*c_in)+1e6*(c_bound'*c_bound);
    m_constr = obj + 1/2*k_op*norm(c_eq) + 1/2*k_op*(c_in'*c_in)+1e6*(c_bound'*c_bound);
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
    elseif length(h)> 1
        h_all = [h;h];  % duplicates the list of active/inactive values
    else 
        h_all = h;
    end

    n = numel(x); % number of elements in x
    dF = zeros(n, 1); % initialize Jacobian matrix

    if ismember(method,{'forward','backward'})
        f0 = func(x);
    end

    % delete(gcp('nocreate'))
    % % c = parcluster;
    % c = parpool('Threads');
    % % c.NumWorkers = 2;
    % % c.NumThreads = 2;
    % % parpool(c);
    % 

    % parfor( i = 1:n,c)
    for i = 1:n
        if length(h_all)>1
            h = h_all(i);
        end
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

function dF = study_numericalDerivative(func, x, method, iter)

    EXPS = -12:-8;
    DSs0 = 10.^EXPS;
    DSs1 = linspace(DSs0(1),DSs0(end),10);
    DSs = sort([DSs0,DSs1(2:end-1)]);

    nds = length(DSs);
    n = numel(x); % number of elements in x

    if strcmp(method, 'central')
        SS = zeros(n,2*nds+1);
        MM = zeros(n,2*nds+1);
    else
        SS = zeros(n,nds+1);
        MM = zeros(n,nds+1);
    end
    
    f0 = func(x);
    screenSize = [0, 0, 1920, 1080];
    widthInches = 12; % Example width
    heightInches = 6.5; % Example height

    dF = zeros(n, nds); % initialize Jacobian matrix
    ids = 1;
    for h=DSs
    
        for i = 1:n
            xTemp = x;

            if strcmp(method, 'central')
                xTemp(i) = x(i) + h;
                f1 = func(xTemp);
                xTemp(i) = x(i) - h;
                f2 = func(xTemp);
                dF(i,ids) = (f1 - f2) / (2*h);
                SS(i,[ids,nds+ids]) = [x(i) - h,x(i) + h];
                MM(i,[ids,nds+ids]) = [f2,f1];
            elseif strcmp(method, 'forward')
                f1 = f0;
                xTemp(i) = x(i) + h;
                f2 = func(xTemp);
                dF(i) = (f2 - f1) / h;
                SS(i,ids) = x(i) + h;
                MM(i,ids) = f2;

            elseif strcmp(method, 'backward')
                xTemp(i) = x(i) - h;
                f1 = func(xTemp);
                f2 = f0;
                dF(i) = (f2 - f1) / h;
                SS(i,ids) = x(i) - h;
                MM(i,ids) = f1;
            else
                error('Invalid method. Choose central, forward, or backward.');
            end

            MM(i,end) = f0;
            SS(i,end) = x(i);

        end


        ids=ids+1;

    end
    figure(7)
    % set(gcf, 'Position', screenSize); % Maximize figure
    set(gcf, 'Units', 'inches', 'Position', [0, 0, widthInches, heightInches]); % Set figure size
    set(gcf, 'PaperPositionMode', 'auto'); % Ensure the size is respected when printing/saving

    sgtitle (strcat('Derivatives iteration ',int2str(iter)))    % main title
    for i_sp=1:n/2
        subplot(2,n/4,i_sp)
        semilogx(DSs,dF(i_sp,:),DSs,dF(n/2+i_sp,:))
        legend('s1','s2')
        title(strcat("node ",int2str(i_sp)))
        xlabel('$\Delta s$','Interpreter','latex')
        ylabel('$\frac{d m}{d s_{1,2}}$','Interpreter','latex')
    end
    saveas(gcf, strcat('Deriv_comp_iter_', int2str(iter), '.png'))
    
    figure(8)
    % set(gcf, 'Position', screenSize); % Maximize figure
    set(gcf, 'Units', 'inches', 'Position', [0, 0, widthInches, heightInches]); % Set figure size
    set(gcf, 'PaperPositionMode', 'auto'); % Ensure the size is respected when printing/saving
    sgtitle (strcat('Objective fcn iter',int2str(iter))) % main title
    pos_sp = [1,2,5,6,9,10,3,4,7,8,11,12];
    node = [1:6,1:6];
    for i_sp=1:n
        subplot(3,n/3,pos_sp(i_sp))
        if i_sp<7
            color='blue';
        else
            color='red';
        end
        scatter(SS(i_sp,:),MM(i_sp,:),color)
        which_s = (i_sp>6) + 1;
        title(strcat("node ",int2str(node(i_sp)),", s",int2str(which_s)))
        xlabel(strcat('$s_{',int2str(which_s),'}$'),'Interpreter','latex')
        ylabel('$m$','Interpreter','latex')
    end
    saveas(gcf, strcat('Objective_fcn_iter_', int2str(iter), '.png'))


end

