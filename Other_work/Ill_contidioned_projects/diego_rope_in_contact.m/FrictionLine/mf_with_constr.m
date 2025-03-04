function [m,f] = mf_with_constr(u,X,conn,free_ind,old_constraints,mu,ratio,k_pen,sph,slaves,dofs,s,sh)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    len_el=length(conn);
    len_u=length(u);
    m=0;  % it will adapt to datatype in the for loop
    % f=zeros(len_u,1,'like',u);  
    f=zeros(len_u,1,'mp');
    % f=zeros(len_u,1);


    for i=1:len_el

        nr1=conn(i,1);
        nr2=conn(i,2);
        dofi=[nr1*3-2;nr1*3-1;nr1*3;nr2*3-2;nr2*3-1;nr2*3];

        x1 = X(nr1,:)'+u(3*nr1-2:3*nr1);
        x2 = X(nr2,:)'+u(3*nr2-2:3*nr2);

        L0 = norm(X(nr1,:)-X(nr2,:));
        L  = norm(x1-x2);
        if L>L0
            k=1.0;
        else
            k=ratio;
        end
        
        m = m + k*L0*(0.5*(log(L/L0))^2);
        

        a = [x1-x2;x2-x1];
        dLdu = a/L;

        f(dofi)=f(dofi)+k*(L0/L)*log(L/L0)*dLdu ;
        
    end


    f(setdiff(1:numel(f),free_ind))=0;

    if ~isempty(old_constraints)
        Cxy = cell2mat(sph{1})';
        R  = sph{2};
        xs = X(slaves,:) + u(dofs(slaves,:));

    end

    cnt_act = 0;
    for ids=old_constraints
    
        xsi = xs(ids,:)';
        nr1 = slaves(ids);

        if all(sh(ids,:)==[-1,-1])
            g =norm(xsi - Cxy) - R;
            dgdu = (xsi-Cxy)/norm(xsi-Cxy);
        else
            cnt_act = cnt_act + 1;
            theta = pi*(1-s(cnt_act,2));
            phi = pi*s(cnt_act,1);
            nor = -[cos(phi)*sin(theta); cos(theta);sin(theta)*sin(phi)];
            xci = Cxy+nor*R;
            g = norm(xsi-xci);
            dgdu = (xsi-xci)/g;

        end

        a=dofs(nr1,:)';
        % f(a)=f(a)+[dEdu1;dEdv1;dEdw1];
        dEdg=k_pen*g;
        f(a)=f(a)+dEdg*dgdu;

        E=0.5*k_pen*g^2;
        m=m+E;
    
    end
end