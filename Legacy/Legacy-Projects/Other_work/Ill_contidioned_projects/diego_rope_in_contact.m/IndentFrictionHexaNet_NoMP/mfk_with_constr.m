function [m,f,K] = mfk_with_constr(u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,s,sh)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    len_el=length(conn);
    idxs_hooked = find(all(sh ~= -1, 2));
    idxs_fless = setdiff(actives,idxs_hooked);
    len_fless = length(idxs_fless);
    
    len_u=length(u);
    m=0;  % it will adapt to datatype in the for loop
    % f=zeros(len_u,1,'like',u);  
    % f=zeros(len_u,1,'mp');
    f=zeros(len_u,1);

    r = zeros(36*len_el+9*len_fless,1);
    c = zeros(36*len_el+9*len_fless,1);
    v = zeros(36*len_el+9*len_fless,1);

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

        a = [x1-x2;x2-x1];
        dLdu = a/L;

        f(dofi)=f(dofi)+k*(L0/L)*log(L/L0)*dLdu ;


        dadu=   [[1.0,0.0,0.0,-1.0,0.0,0.0];...
                [0.0,1.0,0.0,0.0,-1.0,0.0];...
                [0.0,0.0,1.0,0.0,0.0,-1.0];...
                [-1.0,0.0,0.0,1.0,0.0,0.0];...
                [0.0,-1.0,0.0,0.0,1.0,0.0];...
                [0.0,0.0,-1.0,0.0,0.0,1.0]];
        d2Ldu2 = dadu/L - (a*a')/(L^3);

        kij=k*L0*((1-log(L/L0))/L^2*(dLdu*dLdu')+log(L/L0)/L*d2Ldu2);
        r(36*(i-1)+1:36*i) = repmat(dofi,1,6);
        c(36*(i-1)+1:36*i) = repelem(dofi,1,6);
        v(36*(i-1)+1:36*i) = reshape(kij,1,36);

    end


    f(setdiff(1:numel(f),free_ind))=0;


    if ~isempty(actives)
        Cxy = cell2mat(sph{1})';
        R  = sph{2};
        xs = X(slaves,:) + u(dofs(slaves,:));

    end

    if ~isempty(idxs_fless)
        ind_act = 0;
        for ids=idxs_fless
    
            xsi = xs(ids,:)';
            nr1 = slaves(ids);
    
            dis = norm(xsi - Cxy);
            g = dis - R;
            dgdu = (xsi-Cxy)/norm(xsi-Cxy);
            d2gdu2=(eye(3) - dgdu*dgdu')/dis;
    
            dofi=dofs(nr1,:)';
            dEdg=k_pen*g;
            f(dofi)=f(dofi)+dEdg*dgdu;
    
            kij=k_pen*(dgdu*dgdu'+g*d2gdu2);
    
            E=0.5*k_pen*g^2;
            m=m+E;
    
            desde = 36*len_el+9*ind_act+1;
            hasta = 36*len_el+9*(ind_act+1);
            r(desde:hasta) = repmat(dofi',1,3);
            c(desde:hasta) = repelem(dofi',1,3);
            v(desde:hasta) = reshape(kij,1,9);
            ind_act=ind_act+1;
    
        end
    end

    K = sparse(r,c,v);

end