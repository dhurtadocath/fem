function [m,f] = mf_with_constr(u,FE_node_coord,FE_node_nrs,free_ind,old_constraints,ratio,k_pen,sph)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    
    len_el=length(FE_node_nrs);
    len_u=length(u);
    m=0;  % it will adapt to datatype in the for loop
    % f=zeros(len_u,1,'like',u);  
    % f=zeros(len_u,1,'mp');
    f=zeros(len_u,1);

    

    for i=1:len_el

        nr1=FE_node_nrs(i,1);
        nr2=FE_node_nrs(i,2);

        x1=FE_node_coord(nr1,1);
        y1=FE_node_coord(nr1,2);
        z1=FE_node_coord(nr1,3);
        x2=FE_node_coord(nr2,1);
        y2=FE_node_coord(nr2,2);
        z2=FE_node_coord(nr2,3);

        u1=u(nr1*3-2);
        v1=u(nr1*3-1);
        w1=u(nr1*3);
        u2=u(nr2*3-2);
        v2=u(nr2*3-1);
        w2=u(nr2*3);

        L0=norm([x2-x1;y2-y1;z2-z1]);
        L=norm([x2+u2-x1-u1;y2+v2-y1-v1;z2+w2-z1-w1]);

        if L>L0
            k=1.0;
        else
            k=ratio;
        end
        
        E=0.5*k*(L-L0)^2;

        dEdu1=-(x2+u2-x1-u1)/L*(L-L0);
        dEdv1=-(y2+v2-y1-v1)/L*(L-L0);
        dEdw1=-(z2+w2-z1-w1)/L*(L-L0);

        a=[nr1*3-2;nr1*3-1;nr1*3;nr2*3-2;nr2*3-1;nr2*3];
        f(a)=f(a)+k*[dEdu1;dEdv1;dEdw1;-dEdu1;-dEdv1;-dEdw1];

        m=m+E;
    end
    f(setdiff(1:numel(f),free_ind))=0;

    for i=1:length(old_constraints)
    
        xR = sph(1);
        yR = sph(2);
        zR = sph(3);
        R  = sph(4);

        nr1=old_constraints(i);
    
        if nr1==0
            break
        end

        x1=FE_node_coord(nr1,1);
        y1=FE_node_coord(nr1,2);
        z1=FE_node_coord(nr1,3);
    
        u1=u(nr1*3-2);
        v1=u(nr1*3-1);
        w1=u(nr1*3);
    
        L=norm([x1+u1-xR;y1+v1-yR;z1+w1-zR]);
        dis=L-R;
    
        E=0.5*k_pen*dis^2;
    
        dEdL=k_pen*dis;
    
        dEdu1=(x1+u1-xR)/L*dEdL;
        dEdv1=(y1+v1-yR)/L*dEdL;
        dEdw1=(z1+w1-zR)/L*dEdL;
    
        a=[nr1*3-2;nr1*3-1;nr1*3];
        f(a)=f(a)+[dEdu1;dEdv1;dEdw1];
    
        m=m+E;
    
    end
end