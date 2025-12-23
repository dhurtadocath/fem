function [u,m_new,iter] = new_fullBFGStoCompileToMex(u,X,conn,is_free,actives,mu,ratio,k_pen,Cx,Cy,Cz,R,slaves,dofs,s,sh)
%Here is a fully expanded version of the BFGS algorithm that solves the
%mechanical subproblem with a specific list of constrained slave nodes

% old_constraints = slaves(actives==true);
old_constraints = find(actives==1);
free_ind = dofs(is_free==true);

delta_alpha=0.1;
c_par=1e-4;
c_par2=0.9;

tol=1e-13;
tol2=1e-15;

nfr = length(free_ind);
[m_new,f]=mf_with_constr(u,X,conn,free_ind,old_constraints,mu,ratio,k_pen,Cx,Cy,Cz,R,slaves,dofs,s,sh);
f_2=f(free_ind);
K_new_inv=inv(eye(nfr));
f_new=zeros(nfr,1);
h_new=zeros(nfr,1);
m_old = 1e100;
iter=0;

delta_ufr=zeros(length(free_ind));
delta_f=zeros(length(free_ind));
m_2 = 0.0;
ux = u;

scalar1 = 0.0;
scalar2 = 0.0;

% while norm(f_2-f_new)>tol && norm(f_2)>0
while (m_old-m_new)>tol*abs(m_old)
    iter=iter+1;

    m_old=m_new;
    f_old=f_new;

    f_new=f_2;

    K_old_inv=K_new_inv;
    delta_f=f_new-f_old;

    if iter==1
        h_new=-K_old_inv*f_new;
    else
        % K_new_inv=K_old_inv+(delta_u'*delta_f+delta_f'*K_old_inv*delta_f)*(delta_u*delta_u')/(delta_u'*delta_f)^2-(K_old_inv*delta_f*delta_u'+delta_u*delta_f'*K_old_inv)/(delta_u'*delta_f);

        scalar1(1) = delta_ufr'*delta_f+delta_f'*K_old_inv*delta_f;
        scalar2(1) = delta_ufr'*delta_f;

        assert(isscalar(scalar1));
        assert(isscalar(scalar2));
        % K_new_inv=K_old_inv+scalar1*(delta_u*delta_u')/scalar2^2-(K_old_inv*delta_f*delta_u'+delta_u*delta_f'*K_old_inv)/scalar2;
    
        % Step 1: Calculate matrix products separately
        product1 = delta_ufr * delta_ufr';
        product2 = K_old_inv * delta_f * delta_ufr';
        product3 = delta_ufr * delta_f' * K_old_inv;
        
        % Step 2: Perform operations with scalars
        part1 = (scalar1/scalar2^2) * product1;
        part2 = (product2 + product3) / scalar2;
        
        % Step 3: Combine the parts
        K_new_inv = K_old_inv + part1 - part2;

        
        h_new=-K_new_inv*f_new;

    end

    alpha_1=0;
    m_1 = m_new;
    alpha_2 = delta_alpha;
    ux(free_ind) = u(free_ind)+ alpha_2*h_new;

    m_2=mf_with_constr(ux,X,conn,free_ind,old_constraints,mu,ratio,k_pen,Cx,Cy,Cz,R,slaves,dofs,s,sh);
    alpha_3 = 2*delta_alpha;
    ux(free_ind) = u(free_ind) + alpha_3*h_new;
    m_3=mf_with_constr(ux,X,conn,free_ind,old_constraints,mu,ratio,k_pen,Cx,Cy,Cz,R,slaves,dofs,s,sh);

    % if m_3>m_1 && m_2>m_1
    %     ux = u;
    %     m_new=m_1;
    %     f_2 = f_new;
    % else
        lcnt = 0;
        mx = m_new;
        fx = f_new;
        alpha_x= 0.1;
        while alpha_x>0 && (mx > m_new+c_par*alpha_x*h_new'*f_new || h_new'*fx < c_par2*h_new'*f_new)
            lcnt=lcnt+1;
            pars=-[alpha_1^2,alpha_1,1 ; alpha_2^2,alpha_2,1 ; alpha_3^2,alpha_3,1  ]\[m_1;m_2;m_3];
            alpha_x = -pars(2)/2/pars(1);
            ux(free_ind) = u(free_ind) + alpha_x*h_new;
            [mx,f_tot]=mf_with_constr(ux,X,conn,free_ind,old_constraints,mu,ratio,k_pen,Cx,Cy,Cz,R,slaves,dofs,s,sh);
            fx = f_tot(free_ind);
            if alpha_x>alpha_3
                alpha_1 = alpha_2;
                alpha_2 = alpha_3;
                alpha_3 = alpha_x;
                m_1 = m_2;
                m_2 = m_3;
                m_3 = mx;
            elseif alpha_x < alpha_1
                alpha_3 = alpha_2;
                alpha_2 = alpha_1;
                alpha_1 = alpha_x;
                m_3 = m_2;
                m_2 = m_1;
                m_1 = mx;
            elseif alpha_x < alpha_3 && alpha_x > alpha_2
                if alpha_x - alpha_1 < alpha_3 - alpha_2
                    alpha_3 = alpha_x;
                    m_3 = mx;
                else
                    alpha_1 = alpha_2;
                    alpha_2 = alpha_x;
                    m_1 = m_2;
                    m_2 = mx;
                end
            elseif alpha_x < alpha_2 && alpha_x > alpha_1
                if alpha_3 - alpha_x < alpha_2 - alpha_1
                    alpha_1 = alpha_x;
                    m_1 = mx;
                else
                    alpha_3 = alpha_2;
                    alpha_2 = alpha_x;
                    m_3 = m_2;
                    m_2 = mx;
                end
            else
                error('wrong alpha_x or alpha_x==alpha_2');
            end
        end

        if alpha_x <= 0 
            f_2 = f_new;
            ux = u;
        else
            f_2 = fx;
            m_new = mx;
        end
    % end

    delta_ufr = ux(free_ind) - u(free_ind);
    u = ux;



end

    % sprintf('fullBFGS: %s',int2str(iter))







end




function [m,f] = mf_with_constr(u,X,conn,free_ind,old_constraints,mu,ratio,k_pen,Cx,Cy,Cz,R,slaves,dofs,s,sh)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    len_el=length(conn);
    len_u=length(u);
    % m=mp('0');  % it will adapt to datatype in the 'for' loop
    m=0.0;  % it will adapt to datatype in the 'for' loop
    % f=zeros(len_u,1,'like',u);  
    % f=zeros(len_u,1,'mp');

    if nargout==2
        f=zeros(len_u,1);
    end

    for i=1:len_el

        nr1=conn(i,1);
        nr2=conn(i,2);
        dofi=[nr1*3-2;nr1*3-1;nr1*3;nr2*3-2;nr2*3-1;nr2*3];

        x1 = X(nr1,:)'+u(3*nr1-2:3*nr1);
        x2 = X(nr2,:)'+u(3*nr2-2:3*nr2);

        L0 = norm(X(nr1,:)-X(nr2,:));
        L  = norm(x1-x2);
        if L>L0
            if size(conn,2)>2
                k=1.0*conn(i,3);
            else
                k=1.0;
            end
        else
            if size(conn,2)>2
                k=ratio*conn(i,3);
            else
                k=ratio;
            end
        end
        
        m = m + k*L0*(0.5*(log(L/L0))^2);
        
        if nargout==2
            a = [x1-x2;x2-x1];
            dLdu = a/L;
    
            f(dofi)=f(dofi)+k*(L0/L)*log(L/L0)*dLdu ;
        end
    end

    if nargout==2
        % Sort 'free_ind' in ascending order
        free_ind_sorted = sort(free_ind);
        % Use the sorted array with 'setdiff'
        f(setdiff(1:numel(f), free_ind_sorted)) = 0;
        % f(setdiff(1:numel(f),free_ind))=0;
    end

    Cxy = [Cx;Cy;Cz];
    xs = X(slaves,:) + u(dofs(slaves,:));

    cnt_act = 0;
    for i_cnstr=1:length(old_constraints)
    
        ids = old_constraints(i_cnstr);

        xsi = xs(ids,:)';
        nr1 = slaves(ids);

        if all(sh(ids,:)==[-1,-1])
            g =norm(xsi - Cxy) - R;
            if nargout == 2
                dgdu = (xsi-Cxy)/norm(xsi-Cxy);
            end
        else
            cnt_act = cnt_act + 1;
            theta = pi*(1-s(ids,2));
            phi = pi*s(ids,1);
            nor = -[cos(phi)*sin(theta); cos(theta);sin(theta)*sin(phi)];
            xci = Cxy+nor*R;
            g = norm(xsi-xci);
            if nargout == 2
                dgdu = (xsi-xci)/g;
            end
        end

        if nargout ==2 
        a=dofs(nr1,:)';
        % f(a)=f(a)+[dEdu1;dEdv1;dEdw1];
        dEdg=k_pen*g;
        f(a)=f(a)+dEdg*dgdu;
        end

        E=0.5*k_pen*g^2;
        m=m+E;
    
    end
end