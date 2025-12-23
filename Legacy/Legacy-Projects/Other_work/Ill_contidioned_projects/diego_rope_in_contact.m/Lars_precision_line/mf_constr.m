function [m,f] = mf_constr(u,FE_node_coord,old_constraints,k_pen,sph)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    len_u=length(u);
    m=mp('0');
    f=mp(zeros(len_u,1));
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