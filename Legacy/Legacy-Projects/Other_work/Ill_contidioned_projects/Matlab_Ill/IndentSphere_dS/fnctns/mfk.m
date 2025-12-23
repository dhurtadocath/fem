function [m,ff,K]=mfk(u,X,conn,free_ind,ratio)
    len_u=length(u);
    len_el=length(conn);
    % m=mp('0');
    % f=zeros(len_u,1,'mp');
    m=0.0;
    f=zeros(len_u,1);
    if nargout>2
        K=zeros(len_u,len_u);
    end
    if isempty(free_ind)==0
        free_ind=1:length(u);
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
        

        a = [x1-x2;x2-x1];
        dLdu = a/L;

        f(dofi)=f(dofi)+k*(L0/L)*log(L/L0)*dLdu ;

        if nargout>2
            dadu=   [[1.0,0.0,0.0,-1.0,0.0,0.0];...
                    [0.0,1.0,0.0,0.0,-1.0,0.0];...
                    [0.0,0.0,1.0,0.0,0.0,-1.0];...
                    [-1.0,0.0,0.0,1.0,0.0,0.0];...
                    [0.0,-1.0,0.0,0.0,1.0,0.0];...
                    [0.0,0.0,-1.0,0.0,0.0,1.0]];
            d2Ldu2 = dadu/L - (a*a')/(L^3);
    
            K(dofi,dofi)=K(dofi,dofi)+k*L0*((1-log(L/L0))/L^2*(dLdu*dLdu')+log(L/L0)/L*d2Ldu2);
        end
    end
    % ff=zeros(len_u,1,'mp');
    ff=zeros(len_u,1);
    ff(free_ind)=f(free_ind);
end
