clear all
% close all
clc

digits(100)

addpath('~/AdvanpixMCT-5.0.0.15222')
format longG


nrx=11;

tot_incr=100;

Utot=mp(10);

k_pen=mp(1);

alpha_init=mp(1);
c_par=mp(1e-4);
c_par2=mp(0.9);
r_par=mp(0.5);

tol=mp(0);
tol2=mp(1e-15);

ui = zeros(33,1);
ui(2:3:end)=0.1;

xR=mp((nrx-1)/2);
yR=mp(4);
zR=mp(4);
R=mp(5);

% THETA=0:2*pi/15:2*pi;
% OMEGA=-pi/2:pi/15:pi/2;
% figure(1)
% hold on
% for j=1:length(OMEGA)
%     omega=OMEGA(j);
% 
%     for i=1:length(THETA)
% 
%         theta=THETA(i);
% 
%         x=xR+R*cos(theta)*cos(omega);
%         y=yR+R*sin(theta)*cos(omega);
%         z=zR+R*sin(omega);
% 
%         plot3(x,y,z,'bo')
% 
%     end
% end


FE_node_coord=mp((0:1:nrx-1)');

FE_node_nrs=mp([(1:nrx-1)' (2:nrx)']);
len_el=nrx-1;
len_u=nrx*3;

Digs = 30:50;
ntry=length(Digs);
r_f=zeros(ntry,1);
r_u=zeros(ntry,1);
r_m=zeros(ntry,1);
r_i=zeros(ntry,1);

icnt = 0;

for ndig=Digs

    mp.Digits(ndig);
    
    u=mp(zeros(nrx*3,1));
    % load u u
    % u=mp(u);
    
    free_ind=(4:1:(nrx-1)*3)';
    fixed_ind=[1;2;3; (nrx*3-2:1:nrx*3)'];
    
    
    % figure(1)
    % hold on
    % 
    % plot3(FE_node_coord,zeros(nrx,1),zeros(nrx,1),'bo')
    % for i=1:len_el
    % 
    %     nr1=FE_node_nrs(i,1);
    %     nr2=FE_node_nrs(i,2);
    % 
    %     x1=FE_node_coord(nr1);
    %     x2=FE_node_coord(nr2);
    % 
    %     u1=u(nr1*3-2);
    %     v1=u(nr1*3-1);
    %     w1=u(nr1*3);
    %     u2=u(nr2*3-2);
    %     v2=u(nr2*3-1);
    %     w2=u(nr2*3);
    % 
    %     plot3([x1+u1;x2+u2],[v1;v2],[w1;w2],'b')
    % 
    % end
    
    
    
    % m=0;
    % f=zeros(len_u,1);
    % for i=1:len_el
    %
    %     nr1=FE_node_nrs(i,1);
    %     nr2=FE_node_nrs(i,2);
    %
    %     x1=FE_node_coord(nr1);
    %     x2=FE_node_coord(nr2);
    %
    %     u1=u(nr1*3-2);
    %     v1=u(nr1*3-1);
    %     w1=u(nr1*3);
    %     u2=u(nr2*3-2);
    %     v2=u(nr2*3-1);
    %     w2=u(nr2*3);
    %
    %     L=norm([x2+u2-x1-u1;v2-v1;w2-w1]);
    %
    %     E=0.5*(L-1)^2;
    %
    %     dEdu1=-(x2+u2-x1-u1)/L*(L-1);
    %     dEdv1=-(v2-v1)/L*(L-1);
    %     dEdw1=-(w2-w1)/L*(L-1);
    %
    %     a=[nr1*3-2;nr1*3-1;nr1*3;nr2*3-2;nr2*3-1;nr2*3];
    %     f(a)=f(a)+[dEdu1;dEdv1;dEdw1;-dEdu1;-dEdv1;-dEdw1];
    %
    %     m=m+E;
    %
    % end
    
    new_constraints=zeros(2,1);
    new_constraints(1:2)=[];
    tic
    % for incr=1:tot_incr
    for incr=1:1
    
        u0=u;
    
        YN=1;
        iter_out=0;
        while YN==1
    
            iter_out=iter_out+1;
    
            old_constraints=new_constraints;
    
            u=u0;
    
            incr
    
            u(2)=mp(incr/tot_incr*Utot);
            u(nrx*3-1)=mp(incr/tot_incr*Utot);
    
    
    
            m_new=mp(0);
            f=mp(zeros(len_u,1));
            for i=1:len_el
    
                nr1=FE_node_nrs(i,1);
                nr2=FE_node_nrs(i,2);
    
                x1=FE_node_coord(nr1);
                x2=FE_node_coord(nr2);
    
                u1=u(nr1*3-2);
                v1=u(nr1*3-1);
                w1=u(nr1*3);
                u2=u(nr2*3-2);
                v2=u(nr2*3-1);
                w2=u(nr2*3);
    
                L=norm([x2+u2-x1-u1;v2-v1;w2-w1]);
    
                if L>1
                    E=0.5*(L-1)^2;
    
                    dEdu1=-(x2+u2-x1-u1)/L*(L-1);
                    dEdv1=-(v2-v1)/L*(L-1);
                    dEdw1=-(w2-w1)/L*(L-1);
    
                    a=[nr1*3-2;nr1*3-1;nr1*3;nr2*3-2;nr2*3-1;nr2*3];
                    f(a)=f(a)+[dEdu1;dEdv1;dEdw1;-dEdu1;-dEdv1;-dEdw1];
    
                    m_new=m_new+E;
                end
            end
            for i=1:length(old_constraints)
    
                nr1=old_constraints(i);
    
                x1=FE_node_coord(nr1);
    
                u1=u(nr1*3-2);
                v1=u(nr1*3-1);
                w1=u(nr1*3);
    
                L=norm([x1+u1-xR;v1-yR;w1-zR]);
                dis=L-R;
    
                E=0.5*k_pen*dis^2;
    
                dEdL=k_pen*dis;
    
                dEdu1=(x1+u1-xR)/L*dEdL;
                dEdv1=(v1-yR)/L*dEdL;
                dEdw1=(w1-zR)/L*dEdL;
    
                a=[nr1*3-2;nr1*3-1;nr1*3];
                f(a)=f(a)+[dEdu1;dEdv1;dEdw1];
    
                m_new=m_new+E;
    
            end
            f_2=f(free_ind);
            f_res=norm(f(fixed_ind));
    
            tic
            K_new_inv=inv(eye(len_u-6));
            f_new=zeros(len_u-6,1);
            f_old=ones(len_u-6,1);
            m_old=1e100;
            iter=0;
            while norm(f_2-f_new)>0 && norm(f_2)>0
    %         while (m_old-m_new)>tol*abs(m_old)
    
                iter=iter+1;
    
                m_old=m_new;
                f_old=f_new;
    
                f_new=f_2;
    
                K_old_inv=K_new_inv;
                delta_f=f_new-f_old;
    
                if iter==1
                    h_new=-K_old_inv*f_new;
                else
                    K_new_inv=K_old_inv+(delta_u'*delta_f+delta_f'*K_old_inv*delta_f)*delta_u*delta_u'/(delta_u'*delta_f)^2-(K_old_inv*delta_f*delta_u'+delta_u*delta_f'*K_old_inv)/(delta_u'*delta_f);
                    h_new=-K_new_inv*f_new;
                end
    
                alpha3=alpha_init;
                ux=u;
                ux(free_ind)=u(free_ind)+alpha3*h_new;
                m_3=mp(0);
                f=mp(zeros(len_u,1));
                for i=1:len_el
    
                    nr1=FE_node_nrs(i,1);
                    nr2=FE_node_nrs(i,2);
    
                    x1=FE_node_coord(nr1);
                    x2=FE_node_coord(nr2);
    
                    u1=ux(nr1*3-2);
                    v1=ux(nr1*3-1);
                    w1=ux(nr1*3);
                    u2=ux(nr2*3-2);
                    v2=ux(nr2*3-1);
                    w2=ux(nr2*3);
    
                    L=norm([x2+u2-x1-u1;v2-v1;w2-w1]);
    
                    if L>1
                        E=0.5*(L-1)^2;
    
                        dEdu1=-(x2+u2-x1-u1)/L*(L-1);
                        dEdv1=-(v2-v1)/L*(L-1);
                        dEdw1=-(w2-w1)/L*(L-1);
    
                        a=[nr1*3-2;nr1*3-1;nr1*3;nr2*3-2;nr2*3-1;nr2*3];
                        f(a)=f(a)+[dEdu1;dEdv1;dEdw1;-dEdu1;-dEdv1;-dEdw1];
    
                        m_3=m_3+E;
                    end
    
                end
                for i=1:length(old_constraints)
    
                    nr1=old_constraints(i);
    
                    x1=FE_node_coord(nr1);
    
                    u1=ux(nr1*3-2);
                    v1=ux(nr1*3-1);
                    w1=ux(nr1*3);
    
                    L=norm([x1+u1-xR;v1-yR;w1-zR]);
                    dis=L-R;
    
                    E=0.5*k_pen*dis^2;
    
                    dEdL=k_pen*dis;
    
                    dEdu1=(x1+u1-xR)/L*dEdL;
                    dEdv1=(v1-yR)/L*dEdL;
                    dEdw1=(w1-zR)/L*dEdL;
    
                    a=[nr1*3-2;nr1*3-1;nr1*3];
                    f(a)=f(a)+[dEdu1;dEdv1;dEdw1];
    
                    m_3=m_3+E;
    
                end
                f_3=f(free_ind);
    
                signal1=0;
                if m_3<=m_new+c_par*alpha3*h_new'*f_new && abs(h_new'*f_3)<=c_par2*abs(h_new'*f_new)
                    signal1=1;
                end
    
                iter2a=0;
                while m_3<m_new+c_par*alpha3*h_new'*f_new && h_new'*f_3<-c_par2*h_new'*f_new && signal1==0
    
                    alpha3=alpha3/r_par;
    
                    ux=u;
                    ux(free_ind)=u(free_ind)+alpha3*h_new;
                    m_3=mp(0);
                    f=mp(zeros(len_u,1));
                    for i=1:len_el
    
                        nr1=FE_node_nrs(i,1);
                        nr2=FE_node_nrs(i,2);
    
                        x1=FE_node_coord(nr1);
                        x2=FE_node_coord(nr2);
    
                        u1=ux(nr1*3-2);
                        v1=ux(nr1*3-1);
                        w1=ux(nr1*3);
                        u2=ux(nr2*3-2);
                        v2=ux(nr2*3-1);
                        w2=ux(nr2*3);
    
                        L=norm([x2+u2-x1-u1;v2-v1;w2-w1]);
    
                        if L>1
                            E=0.5*(L-1)^2;
    
                            dEdu1=-(x2+u2-x1-u1)/L*(L-1);
                            dEdv1=-(v2-v1)/L*(L-1);
                            dEdw1=-(w2-w1)/L*(L-1);
    
                            a=[nr1*3-2;nr1*3-1;nr1*3;nr2*3-2;nr2*3-1;nr2*3];
                            f(a)=f(a)+[dEdu1;dEdv1;dEdw1;-dEdu1;-dEdv1;-dEdw1];
    
                            m_3=m_3+E;
                        end
    
                    end
                    for i=1:length(old_constraints)
    
                        nr1=old_constraints(i);
    
                        x1=FE_node_coord(nr1);
    
                        u1=ux(nr1*3-2);
                        v1=ux(nr1*3-1);
                        w1=ux(nr1*3);
    
                        L=norm([x1+u1-xR;v1-yR;w1-zR]);
                        dis=L-R;
    
                        E=0.5*k_pen*dis^2;
    
                        dEdL=k_pen*dis;
    
                        dEdu1=(x1+u1-xR)/L*dEdL;
                        dEdv1=(v1-yR)/L*dEdL;
                        dEdw1=(w1-zR)/L*dEdL;
    
                        a=[nr1*3-2;nr1*3-1;nr1*3];
                        f(a)=f(a)+[dEdu1;dEdv1;dEdw1];
    
                        m_3=m_3+E;
    
                    end
                    f_3=f(free_ind);
    
    
                    iter2a=iter2a+1;
    
                    if m_3<=m_new+c_par*alpha3*h_new'*f_new && abs(h_new'*f_3)<=c_par2*abs(h_new'*f_new)
    
                        signal1=1;
    
                    end
    
                end
    
                if signal1==0
                    alpha1=0;
                    m_1=m_new;
                    f_1=f_new;
    
                    alpha2=alpha3/2;
                    ux=u;
                    ux(free_ind)=u(free_ind)+alpha2*h_new;
                    m_2=mp(0);
                    f=mp(zeros(len_u,1));
                    for i=1:len_el
    
                        nr1=FE_node_nrs(i,1);
                        nr2=FE_node_nrs(i,2);
    
                        x1=FE_node_coord(nr1);
                        x2=FE_node_coord(nr2);
    
                        u1=ux(nr1*3-2);
                        v1=ux(nr1*3-1);
                        w1=ux(nr1*3);
                        u2=ux(nr2*3-2);
                        v2=ux(nr2*3-1);
                        w2=ux(nr2*3);
    
                        L=norm([x2+u2-x1-u1;v2-v1;w2-w1]);
    
                        if L>1
                            E=0.5*(L-1)^2;
    
                            dEdu1=-(x2+u2-x1-u1)/L*(L-1);
                            dEdv1=-(v2-v1)/L*(L-1);
                            dEdw1=-(w2-w1)/L*(L-1);
    
                            a=[nr1*3-2;nr1*3-1;nr1*3;nr2*3-2;nr2*3-1;nr2*3];
                            f(a)=f(a)+[dEdu1;dEdv1;dEdw1;-dEdu1;-dEdv1;-dEdw1];
    
                            m_2=m_2+E;
                        end
    
                    end
                    for i=1:length(old_constraints)
    
                        nr1=old_constraints(i);
    
                        x1=FE_node_coord(nr1);
    
                        u1=ux(nr1*3-2);
                        v1=ux(nr1*3-1);
                        w1=ux(nr1*3);
    
                        L=norm([x1+u1-xR;v1-yR;w1-zR]);
                        dis=L-R;
    
                        E=0.5*k_pen*dis^2;
    
                        dEdL=k_pen*dis;
    
                        dEdu1=(x1+u1-xR)/L*dEdL;
                        dEdv1=(v1-yR)/L*dEdL;
                        dEdw1=(w1-zR)/L*dEdL;
    
                        a=[nr1*3-2;nr1*3-1;nr1*3];
                        f(a)=f(a)+[dEdu1;dEdv1;dEdw1];
    
                        m_2=m_2+E;
    
                    end
                    f_2=f(free_ind);
    
                    signal2=0;
                    iter2=0;
                    while signal2==0
    
                        iter2=iter2+1;
    
                        if alpha3-alpha1<tol2
    
                            signal2=1;
                            m_2=m_new;
                            f_2=f_new;
    
                        elseif m_2>m_new+c_par*alpha2*h_new'*f_new
    
                            alpha3=alpha2;
                            m_3=m_2;
                            f_3=f_2;
    
                            alpha2=0.5*(alpha1+alpha2);
    
                            ux=u;
                            ux(free_ind)=u(free_ind)+alpha2*h_new;
                            m_2=mp(0);
                            f=mp(zeros(len_u,1));
                            for i=1:len_el
    
                                nr1=FE_node_nrs(i,1);
                                nr2=FE_node_nrs(i,2);
    
                                x1=FE_node_coord(nr1);
                                x2=FE_node_coord(nr2);
    
                                u1=ux(nr1*3-2);
                                v1=ux(nr1*3-1);
                                w1=ux(nr1*3);
                                u2=ux(nr2*3-2);
                                v2=ux(nr2*3-1);
                                w2=ux(nr2*3);
    
                                L=norm([x2+u2-x1-u1;v2-v1;w2-w1]);
    
                                if L>1
                                    E=0.5*(L-1)^2;
    
                                    dEdu1=-(x2+u2-x1-u1)/L*(L-1);
                                    dEdv1=-(v2-v1)/L*(L-1);
                                    dEdw1=-(w2-w1)/L*(L-1);
    
                                    a=[nr1*3-2;nr1*3-1;nr1*3;nr2*3-2;nr2*3-1;nr2*3];
                                    f(a)=f(a)+[dEdu1;dEdv1;dEdw1;-dEdu1;-dEdv1;-dEdw1];
    
                                    m_2=m_2+E;
                                end
    
                            end
                            for i=1:length(old_constraints)
    
                                nr1=old_constraints(i);
    
                                x1=FE_node_coord(nr1);
    
                                u1=ux(nr1*3-2);
                                v1=ux(nr1*3-1);
                                w1=ux(nr1*3);
    
                                L=norm([x1+u1-xR;v1-yR;w1-zR]);
                                dis=L-R;
    
                                E=0.5*k_pen*dis^2;
    
                                dEdL=k_pen*dis;
    
                                dEdu1=(x1+u1-xR)/L*dEdL;
                                dEdv1=(v1-yR)/L*dEdL;
                                dEdw1=(w1-zR)/L*dEdL;
    
                                a=[nr1*3-2;nr1*3-1;nr1*3];
                                f(a)=f(a)+[dEdu1;dEdv1;dEdw1];
    
                                m_2=m_2+E;
    
                            end
                            f_2=f(free_ind);
    
                        else
                            if h_new'*f_2<c_par2*h_new'*f_new
    
                                alpha1=alpha2;
                                m_1=m_2;
                                f_1=f_2;
    
                                alpha2=0.5*(alpha2+alpha3);
    
                                ux=u;
                                ux(free_ind)=u(free_ind)+alpha2*h_new;
                                m_2=mp(0);
                                f=mp(zeros(len_u,1));
                                for i=1:len_el
    
                                    nr1=FE_node_nrs(i,1);
                                    nr2=FE_node_nrs(i,2);
    
                                    x1=FE_node_coord(nr1);
                                    x2=FE_node_coord(nr2);
    
                                    u1=ux(nr1*3-2);
                                    v1=ux(nr1*3-1);
                                    w1=ux(nr1*3);
                                    u2=ux(nr2*3-2);
                                    v2=ux(nr2*3-1);
                                    w2=ux(nr2*3);
    
                                    L=norm([x2+u2-x1-u1;v2-v1;w2-w1]);
    
                                    if L>1
                                        E=0.5*(L-1)^2;
    
                                        dEdu1=-(x2+u2-x1-u1)/L*(L-1);
                                        dEdv1=-(v2-v1)/L*(L-1);
                                        dEdw1=-(w2-w1)/L*(L-1);
    
                                        a=[nr1*3-2;nr1*3-1;nr1*3;nr2*3-2;nr2*3-1;nr2*3];
                                        f(a)=f(a)+[dEdu1;dEdv1;dEdw1;-dEdu1;-dEdv1;-dEdw1];
    
                                        m_2=m_2+E;
                                    end
    
                                end
                                for i=1:length(old_constraints)
    
                                    nr1=old_constraints(i);
    
                                    x1=FE_node_coord(nr1);
    
                                    u1=ux(nr1*3-2);
                                    v1=ux(nr1*3-1);
                                    w1=ux(nr1*3);
    
                                    L=norm([x1+u1-xR;v1-yR;w1-zR]);
                                    dis=L-R;
    
                                    E=0.5*k_pen*dis^2;
    
                                    dEdL=k_pen*dis;
    
                                    dEdu1=(x1+u1-xR)/L*dEdL;
                                    dEdv1=(v1-yR)/L*dEdL;
                                    dEdw1=(w1-zR)/L*dEdL;
    
                                    a=[nr1*3-2;nr1*3-1;nr1*3];
                                    f(a)=f(a)+[dEdu1;dEdv1;dEdw1];
    
                                    m_2=m_2+E;
    
                                end
                                f_2=f(free_ind);
    
                            elseif h_new'*f_2>-c_par2*h_new'*f_new
    
                                alpha3=alpha2;
                                m_3=m_2;
                                f_3=f_2;
    
                                alpha2=0.5*(alpha1+alpha2);
    
                                ux=u;
                                ux(free_ind)=u(free_ind)+alpha2*h_new;
                                m_2=mp(0);
                                f=mp(zeros(len_u,1));
                                for i=1:len_el
    
                                    nr1=FE_node_nrs(i,1);
                                    nr2=FE_node_nrs(i,2);
    
                                    x1=FE_node_coord(nr1);
                                    x2=FE_node_coord(nr2);
    
                                    u1=ux(nr1*3-2);
                                    v1=ux(nr1*3-1);
                                    w1=ux(nr1*3);
                                    u2=ux(nr2*3-2);
                                    v2=ux(nr2*3-1);
                                    w2=ux(nr2*3);
    
                                    L=norm([x2+u2-x1-u1;v2-v1;w2-w1]);
    
                                    if L>1
                                        E=0.5*(L-1)^2;
    
                                        dEdu1=-(x2+u2-x1-u1)/L*(L-1);
                                        dEdv1=-(v2-v1)/L*(L-1);
                                        dEdw1=-(w2-w1)/L*(L-1);
    
                                        a=[nr1*3-2;nr1*3-1;nr1*3;nr2*3-2;nr2*3-1;nr2*3];
                                        f(a)=f(a)+[dEdu1;dEdv1;dEdw1;-dEdu1;-dEdv1;-dEdw1];
    
                                        m_2=m_2+E;
                                    end
    
                                end
                                for i=1:length(old_constraints)
    
                                    nr1=old_constraints(i);
    
                                    x1=FE_node_coord(nr1);
    
                                    u1=ux(nr1*3-2);
                                    v1=ux(nr1*3-1);
                                    w1=ux(nr1*3);
    
                                    L=norm([x1+u1-xR;v1-yR;w1-zR]);
                                    dis=L-R;
    
                                    E=0.5*k_pen*dis^2;
    
                                    dEdL=k_pen*dis;
    
                                    dEdu1=(x1+u1-xR)/L*dEdL;
                                    dEdv1=(v1-yR)/L*dEdL;
                                    dEdw1=(w1-zR)/L*dEdL;
                                    a=[nr1*3-2;nr1*3-1;nr1*3];
                                    f(a)=f(a)+[dEdu1;dEdv1;dEdw1];
    
                                    m_2=m_2+E;
    
                                end
                                f_2=f(free_ind);
    
                            else
    
                                signal2=1;
    
                            end
                        end
                    end
                end
                delta_u=ux(free_ind)-u(free_ind);
    
                u=ux;
    
                if signal1==1
                    m_new=m_3;
                    f_2=f_3;
                else
                    m_new=m_2;
                end
    
    %             norm(f_2)
                %m_new
    
            end
    
            norm(f_2-f_new)
            norm(f_2)
    
            new_constraints=zeros(nrx,1);
            cnt=1;
            for i=1:nrx
                x1=FE_node_coord(i);
    
    
                u1=u(i*3-2);
                v1=u(i*3-1);
                w1=u(i*3);
    
                if norm([x1+u1-xR;v1-yR;w1-zR])<R
                    new_constraints(cnt)=i;
                    cnt=cnt+1;
                end
    
            end
            new_constraints(cnt:nrx)=[];
    
            old_constraints
            new_constraints
            
            YN=0;
            if length(old_constraints)~=length(new_constraints)
                YN=1;
            else
                if isequal(old_constraints,new_constraints)
                else
                    YN=1;
                end
            end
    
        end
        toc
    
        if incr==1
            figure(1)
            hold on
            plot3(FE_node_coord+u(1:3:nrx*3-2),u(2:3:nrx*3-1),u(3:3:nrx*3),'rx')
            for i=1:len_el
    
                nr1=FE_node_nrs(i,1);
                nr2=FE_node_nrs(i,2);
    
                x1=FE_node_coord(nr1);
                x2=FE_node_coord(nr2);
    
                u1=u(nr1*3-2);
                v1=u(nr1*3-1);
                w1=u(nr1*3);
                u2=u(nr2*3-2);
                v2=u(nr2*3-1);
                w2=u(nr2*3);
    
                L=norm([x2+u2-x1-u1;v2-v1;w2-w1]);
    
    
                plot3([x1+u1;x2+u2],[v1;v2],[w1;w2],'r')
    
            end
            % asdf
        end
    
    end
    icnt = icnt+1;

    r_m(icnt)=m_2;
    r_f(icnt)=norm(f_2);
    r_u(icnt)=norm(u-ui);
    r_i(icnt)=iter;
end

figure(2)
hold on
subplot(2,2,1)
semilogy(Digs,r_m)
title('m')
subplot(2,2,2)
semilogy(Digs,r_f)
title('|f|')
subplot(2,2,3)
semilogy(Digs,r_u)
title('|du|')
subplot(2,2,4)
plot(Digs,r_i)
title('iters')

    err
