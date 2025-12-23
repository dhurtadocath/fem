clear all
% close all
clc

digits(100)

addpath('~/AdvanpixMCT-5.0.0.15222')
mp.Digits(100);
format longG

ratio = 0;

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


FE_node_coord=mp([(0:1:nrx-1)' zeros(nrx,2)]);

FE_node_nrs=mp([(1:nrx-1)' (2:nrx)']);
len_el=nrx-1;
len_u=nrx*3;

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
for incr=1:tot_incr

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



        [m_new,f]=m_and_f(u,FE_node_coord,FE_node_nrs,free_ind,ratio);
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

        % % finite difference verification
        % [m1,f1,K1] = mfk(u,FE_node_coord,FE_node_nrs,free_ind,ratio);
        % nvars = length(u);
        % f2=zeros(nvars);
        % for i=1:nvars
        %     u2=u;
        %     u2(i)=u2(i)+eps;
        %     [m2,f2,K2] = mfk(u2,FE_node_coord,FE_node_nrs,free_ind,ratio);
        % end
        % 
        % der = (f2-f1)/eps;
     
        
        tic



        options = optimoptions('fminunc','Algorithm','trust-region' ...
                                        ,'FunctionTolerance',0.0...
                                        ,'HessianFcn','objective'...
                                        ,'SpecifyObjectiveGradient',true...
                                        ,'OptimalityTolerance',0.0...
                                        ,'StepTolerance',0.0...
                                        ,'UseParallel',true...
                                        ,'SubproblemAlgorithm','factorization'...
                                        ,'PlotFcn',{'optimplotfirstorderopt','optimplotfval','optimplotstepsize'}...
                                        ,'MaxIterations',100000,'Display','iter-detailed'); % indicate gradient is provided
        % options = optimoptions('fminunc','SpecifyObjectiveGradient',true...
        %                                 ,'OptimalityTolerance',1e-22...
        %                                 ,'StepTolerance',1e-15...
        %                                 ,'MaxIterations',100000 ...
        %                                 ,'MaxFunctionEvaluations',20000); % indicate gradient is provided
        % [u,FVAL,EXITFLAG,OUTPUT,GRAD,HESSIAN] = fminunc(@(x) m_and_f(x,FE_node_coord,FE_node_nrs,free_ind,ratio),u,options);
        [u,FVAL,EXITFLAG,OUTPUT,GRAD,HESSIAN] = fminunc(@(x) mfk(x,FE_node_coord,FE_node_nrs,free_ind,ratio),u,options);


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
        figure(2)
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
        asdf
    end

end
err
