clear all
close all
clc


addpath('~/AdvanpixMCT-5.0.0.15222')
mp.Digits(50);
format longG

tic
nrx=10;

tot_incr=100;

Utot=10;

k_pen=1;
ratio=0;


xR=3;
yR=0;
zR=1;
R=3.1;

len_el=nrx-1;
len_u =nrx*3;

x=(0:1:nrx-1)';

%% Coordinates
FE_node_coord=[zeros(nrx,1,'mp') linspace(-5,5,nrx)' zeros(nrx,1,'mp')];

% FE_node_coord=[zeros(nrx,1) linspace(-5,5,nrx)' zeros(nrx,1)];

%% Connectivity
FE_node_nrs=[1:nrx-1; 2:nrx]';
cnt=1;



%% Init
u=zeros(len_u,1,'mp');
% u=zeros(len_u,1);

fixed_ind=[1;2;3; (nrx*3-2:1:nrx*3)'];
free_ind=(1:len_u)';
free_ind(fixed_ind)=[];

new_constraints=zeros(2,1);
new_constraints(1:2)=[];

%% Increments
for incr=1:tot_incr

    u0=u;

    YN=1;
    iter_out=0;
    while YN==1

        iter_out=iter_out+1;

        old_constraints=new_constraints;

        u=u0;

        incr

        u(1)=incr/tot_incr*Utot;
        u(nrx*3-2)=incr/tot_incr*Utot;

%         u
%         
%         [m,f]=m_and_f(u,FE_node_coord,FE_node_nrs,free_ind,ratio)
        
        % first approx
        tic
        options = optimoptions('fminunc','SpecifyObjectiveGradient',true...
                                        ,'OptimalityTolerance',1e-15...
                                        ,'MaxIterations',1000); % indicate gradient is provided
        [u,FVAL,EXITFLAG,OUTPUT,GRAD,HESSIAN] = fminunc(@(x) m_and_f(x,FE_node_coord,FE_node_nrs,free_ind,ratio),u,options);
        
        u=mp(u);
        % good approx
        options = optimoptions('fminunc','SpecifyObjectiveGradient',true...
                                        ,'OptimalityTolerance',0.0...
                                        ,'StepTolerance',0.0...
                                        ,'PlotFcn',{'optimplotfirstorderopt','optimplotfval','optimplotstepsize'}...
                                        ,'MaxIterations',100000,'Display','iter'); % indicate gradient is provided
        % options = optimoptions('fminunc','SpecifyObjectiveGradient',true...
        %                                 ,'OptimalityTolerance',1e-22...
        %                                 ,'StepTolerance',1e-15...
        %                                 ,'MaxIterations',100000 ...
        %                                 ,'MaxFunctionEvaluations',20000); % indicate gradient is provided
        [u,FVAL,EXITFLAG,OUTPUT,GRAD,HESSIAN] = fminunc(@(x) m_and_f(x,FE_node_coord,FE_node_nrs,free_ind,ratio),u,options);
        toc

        new_constraints=zeros(nrx,1);
        cnt=1;
        for i=1:nrx
            x1=FE_node_coord(i,1);
            y1=FE_node_coord(i,2);
            z1=FE_node_coord(i,3);


            u1=u(i*3-2);
            v1=u(i*3-1);
            w1=u(i*3);

            if norm([x1+u1-xR;y1+v1-yR;z1+w1-zR])<R
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
        plot(FE_node_coord(:,1)+u(1:3:len_u-2),FE_node_coord(:,2)+u(2:3:len_u-1),'bo')
        for i=1:len_el

            nr1=FE_node_nrs(i,1);
            nr2=FE_node_nrs(i,2);

            x1=FE_node_coord(nr1,1);
            y1=FE_node_coord(nr1,2);
            x2=FE_node_coord(nr2,1);
            y2=FE_node_coord(nr2,2);

            u1=u(nr1*3-2);
            v1=u(nr1*3-1);
            w1=u(nr1*3);
            u2=u(nr2*3-2);
            v2=u(nr2*3-1);
            w2=u(nr2*3);

            L=norm([x2+u2-x1-u1;y2+v2-y1-v1;w2-w1]);


            plot3([x1+u1;x2+u2],[y1+v1;y2+v2],[w1;w2],'b')

        end
        asdf
    end

end
err
