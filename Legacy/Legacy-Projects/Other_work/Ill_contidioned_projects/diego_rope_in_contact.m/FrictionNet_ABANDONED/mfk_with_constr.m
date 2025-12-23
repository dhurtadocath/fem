function [m,f,K] = mfk_with_constr(u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,s,sh)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    len_el=length(conn);
    len_u=length(u);
    m=0;  % it will adapt to datatype in the for loop
    % f=zeros(len_u,1,'like',u);  
    % f=zeros(len_u,1,'mp');
    f=zeros(len_u,1,'mp');

    K=zeros(len_u,len_u,'mp');



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


        dadu=   [[1.0,0.0,0.0,-1.0,0.0,0.0];...
                [0.0,1.0,0.0,0.0,-1.0,0.0];...
                [0.0,0.0,1.0,0.0,0.0,-1.0];...
                [-1.0,0.0,0.0,1.0,0.0,0.0];...
                [0.0,-1.0,0.0,0.0,1.0,0.0];...
                [0.0,0.0,-1.0,0.0,0.0,1.0]];
        d2Ldu2 = dadu/L - (a*a')/(L^3);

        K(dofi,dofi)=K(dofi,dofi)+k*L0*((1-log(L/L0))/L^2*(dLdu*dLdu')+log(L/L0)/L*d2Ldu2);

        
    end
    
    f(setdiff(1:numel(f),free_ind))=0;


    if ~isempty(actives)
        Cxy = cell2mat(sph{1})';
        R  = sph{2};
        xs = X(slaves,:) + u(dofs(slaves,:));

    end

    cnt_act = 0;
    for ids=actives
    
        xsi = xs(ids,:)';
        nr1 = slaves(ids);

        if all(sh(ids,:)==[-1,-1])
            dis = norm(xsi - Cxy);
            g = dis - R;
            dgdu = (xsi-Cxy)/norm(xsi-Cxy);
        else
            cnt_act = cnt_act + 1;
            theta = pi*(1-s(cnt_act,2));
            phi = pi*s(cnt_act,1);
            nor = -[cos(phi)*sin(theta); cos(theta);sin(theta)*sin(phi)];
            xci = Cxy+nor*R;
            dis = norm(xsi-xci);
            g = dis;
            dgdu = (xsi-xci)/g;

        end

        a=dofs(nr1,:)';
        dEdg=k_pen*g;
        f(a)=f(a)+dEdg*dgdu;

        d2gdu2=(eye(3) - dgdu*dgdu')/dis;
        K(a,a)=K(a,a)+k_pen*(dgdu*dgdu'+g*d2gdu2);

        E=0.5*k_pen*g^2;
        m=m+E;
    
    end
end