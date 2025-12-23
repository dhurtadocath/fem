clear
close all
clc

% multiprecision
addpath('~/AdvanpixMCT-5.0.0.15222')

mp.Digits(24);
format longG


% SPHERE
xR=1;
yR=0;
zR=3;
R =3.1;
sph = {{xR,yR,zR},R};
nfig = plotsphere(xR,yR,zR,R);

% NET
dl = mp('0.5');
nx=2;
ny=8;
[X,conn]=Net2D(nx,ny,dl,dl);     % returns pos and connectivity
X = [X zeros((nx+1)*(ny+1),1)];
X = displace(X,dl*[-nx,-ny/2,0]);


% % LINE
% nl = 20;
% X = [-1*ones(nl+1,1) linspace(-2,2,nl+1)' 0.5*ones(nl+1,1)];
% conn=[(1:nl)' (2:nl+1)'];

% ratio = 0.01;      % comp_resist/tens_resist
ratio = 0.0;      % comp_resist/tens_resist

% NODES SELECTION
nodes_top = SelectFlatSide(X,'+z');
nodes_base= [1 nx+1 (nx+1)*ny (nx+1)*(ny+1)];
% nodes_base= [1 nl+1];



ndofs=length(X)*3;
dofs=reshape(1:ndofs,3,[])';

dofs_base_x = dofs(nodes_base,1);
dofs_base_y = dofs(nodes_base,2);
dofs_base_z = dofs(nodes_base,3);


% BOUNDARY CONDITIONS
bc1 = {dofs_base_x,"dir",mp('4'), 0.0,1.0};
bc2 = {dofs_base_y,"dir",mp('0.0'), 0.0,1.0};
bc3 = {dofs_base_z,"dir",mp('0.0'), 0.0,1.0};
BCs = {bc1,bc2,bc3};


% CONTACT
k_pen=1000;
mu = 0.5;
slaves = setdiff(nodes_top,nodes_base);
ns = length(slaves);

% MODEL
t0 = mp('0.0');
tf = mp('1.0');
nincr=100;


% INITIALLIZE VARIABLES
u=zeros(ndofs,1,'mp');
% u=zeros(ndofs,1);
actives = [];
sh = -1*ones(ns,2);          % '-1' indicates no contact
shnew=-1*ones(ns,2);          % '-1' indicates no contact
s = zeros(ns,2);
s_pre = s;
ds = s;
t=t0;
dt=(tf-t0)/nincr;
figure(nfig)
hold on
p=plotTruss(u,X,conn,true);
drawnow

% Storage variables
trials = [];
iters_tot = [];
iters_last= [];
Rxs = [];

tic
for incr=1:nincr

    Redo=1;
    iter_out=0;
    trials = [trials,1];
    iters_tot = [iters_tot,0];


    while Redo==1

        iter_out=iter_out+1;

        incr

        [du,df,di]=ApplyBCs(t,t+dt, BCs,ndofs);
        free_ind=setdiff(dofs,di);
        u=u+du;
        % f=f+df;
        ui = zeros(3*length(X),1,'mp');
        ui(1:3:end) = 4*(t+dt);

        idxs_hooked = find(all(sh ~= -1, 2));

        s_opt = s(idxs_hooked,:)+ds(idxs_hooked,:);

        if ~isempty(s_opt)
            lb = zeros(size(s_opt));
            ub =  ones(size(s_opt));
            A = [];
            b = [];
            Aeq = [];
            beq = [];

            nlc = @(sv) nonlcon(sv,u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh);
            options = optimoptions('fmincon','Algorithm','sqp'...
                                  ,'Display','iter-detailed'...
                                  ,'ConstraintTolerance',1e-7...
                                  ,'StepTolerance',1e-8...
                                  ,'OptimalityTolerance',1e-5...
                                  ,'SpecifyObjectiveGradient',true...  
                                  ,'SpecifyConstraintGradient',true);

            [s_opt,~,~,OUT] = fmincon(@(sv) mgt2(sv,u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh),s_opt,A,b,Aeq,beq,lb,ub,nlc,options);
            s(idxs_hooked,:)=s_opt;


            iters = OUT.iterations;
            iters_tot(end)=iters_tot(end)+iters;
        else 
            iters=1;
            iters_tot(end)=iters_tot(end)+1;
        end

        uN = NEWTON(@(uv) mfk_with_constr(uv,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,s_opt,sh),u,free_ind);
        if isnan(uN)
            u = BFGS(@(uv) mf_with_constr(uv,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,s_opt,sh),u,free_ind);
        else
            u=uN;
        end

        err=norm(u-ui)



        [actives, sh, Redo] = checkContact(u, sph, actives, sh, s, X, dofs, slaves, conn, ns, mu,ratio);

        if Redo     % only if nodes entered/exited
            u = u_pre;
            trials(end)=trials(end)+1;
            s = s_pre;
            ds = ds_pre;

        else
            u_pre=u;
            t=t+dt;
            iters_last=[iters_last,iters];
            xs = X(slaves,:) + u(dofs(slaves,:));
            unused = setdiff(1:ns,idxs_hooked);
            s(unused,:)=getProjs(xs(unused,:),sph);
            ds_pre = ds;
            ds = s-s_pre;
            s_pre = s;
        end



    end

    [~,f] = mfk(u,X,conn,free_ind,ratio);
    rx = sum(f(dofs_base_x));
    ry = sum(f(dofs_base_y));
    rz = sum(f(dofs_base_z));
    Rxs=[Rxs;rx,ry,rz];



    delete(p);
    p=plotTruss(u,X,conn);
    drawnow
    saveas(nfig,strcat(num2str(incr), '.png'))

end

toc

figure(23)
hold on
plot(1:100,Rxs(:,1))
plot(1:100,Rxs(:,2))
plot(1:100,Rxs(:,3))
