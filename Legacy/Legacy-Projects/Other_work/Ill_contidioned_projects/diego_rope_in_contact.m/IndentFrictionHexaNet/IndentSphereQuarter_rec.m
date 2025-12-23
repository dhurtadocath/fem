clear
close all
clc

% multiprecision
addpath('~/AdvanpixMCT-5.0.0.15222')

mp.Digits(34);
format longG

Resume = true;


% u_file = matfile('displacements.mat', 'Writable', true);
% f_file = matfile('forces.mat', 'Writable', true);


outdata_path = 'OutData.mat';

% Check if the file already exists
if exist(outdata_path, 'file')
    data = matfile(outdata_path, 'Writable', true);
else
    % If the file doesn't exist, create it by saving an empty variable
    temp = [];
    save(outdata_path, 'temp');
    outfile = matfile(outdata_path, 'Writable', true);
end




if Resume
	load('saved_vars');
    if Redo
    	latest_incr = incr-1;
    else
        latest_incr = incr;
    end
else
	% SPHERE
	xR=mp('0');
	yR=mp('0');
	zR=mp('3');
	R =mp('2.99');
	sph = {{xR,yR,zR},R};


	% NET
	dh = mp('0.5');                 % hexagon's height
	nx=3;                           % number of main rows (there will also be nx-1 secondary rows)
	ny=5;                           % hexagons per row
	% nx=5;                           % number of main rows (there will also be nx-1 secondary rows)
	% ny=9;                           % hexagons per row
	[X,conn]=quarterHexaNet(nx,ny,dh);     % returns pos and connectivity

	X = [X zeros(length(X),1)];


	ratio = 0.0;      % comp_resist/tens_resist

	% NODES SELECTION
	all_nodes = SelectFlatSide(X,'+z');
	nodes_base= [SelectFlatSide(X,'+x'),SelectFlatSide(X,'+y'),];
	nodes_symmX = setdiff(SelectFlatSide(X,'-x'),nodes_base);
	nodes_symmY = setdiff(SelectFlatSide(X,'-y'),nodes_base);   % Symmentry BCs



	ndofs=length(X)*3;
	dofs=reshape(1:ndofs,3,[])';

	dofs_base_x = dofs(nodes_base,1);
	dofs_base_y = dofs(nodes_base,2);
	dofs_base_z = dofs(nodes_base,3);


	% BOUNDARY CONDITIONS
	bc1 = {dofs_base_x,"dir",mp('0.0'), 0.0,1.0};
	bc2 = {dofs_base_y,"dir",mp('0.0'), 0.0,1.0};
	bc3 = {dofs_base_z,"dir",mp('0.0'), 0.0,1.0};

	bc_symmX = {dofs(nodes_symmX,1),"dir",mp('0.0'), 0.0,1.0};
	bc_symmY = {dofs(nodes_symmY,2),"dir",mp('0.0'), 0.0,1.0};

	BCs = {bc1,bc2,bc3,bc_symmX,bc_symmY};


	% CONTACT
	k_pen=1000;
	mu = 0.5;
	% slaves = setdiff(all_nodes,[nodes_symmX,nodes_symmY,nodes_base]);
	slaves = setdiff(all_nodes,nodes_base); % If i find a way to relate dofs for the added nodes, then they can be out of the slave group (nodes_symmY)
	ns = length(slaves);

	% MODEL
	t0 = mp('0.0');
	tf = mp('1.0');
	nincr=100;
	latest_incr = 0;

	% INITIALLIZE VARIABLES
	u=zeros(ndofs,1,'mp');
	u_pre=u;
	actives = [];
	sh = -1*ones(ns,2);          % '-1' indicates no contact
	shnew=-1*ones(ns,2);          % '-1' indicates no contact

	xs = X(slaves,:) + u(dofs(slaves,:));
	s = getProjs(xs,sph);
	s_pre = s;
	ds = zeros(ns,2);
	ds_pre=ds;
	t=t0;
	dt=(tf-t0)/nincr;

    outfile.u_tot = zeros(nincr,ndofs);
    outfile.f_tot = zeros(nincr,ndofs);
    outfile.act_slaves = zeros(nincr,ns);

	% Storage variables
	trials = [];
	iters_tot = [];
	iters_last= [];
	Rxs = [];

end

figure()
% hold on
view(37.5,30.0)
axis equal
psph = plotsphere(sph);
nfig=get(gcf,'Number');

ptrs=plotTruss_XY_symm(u,X,conn,true);
drawnow




tic
global BFGScache;
BFGScache = memoize(@BFGS);

for incr=(latest_incr+1):nincr

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
        sph{1}{3} = zR+(-1.0)*(t+dt)/tf;


        idxs_hooked = find(all(sh ~= -1, 2));
        s_opt = s(idxs_hooked,:)+ds(idxs_hooked,:);

        if ~isempty(idxs_hooked)
            nh = length(idxs_hooked);
    
            idxs_at_X_axis = find(X(slaves(idxs_hooked),2)==0);
            idxs_at_Y_axis = find(X(slaves(idxs_hooked),1)==0);
            nhx = length(idxs_at_X_axis);
            nhy = length(idxs_at_Y_axis);
    
            lb = zeros(size(s_opt));
            ub =  ones(size(s_opt));
            A = [];
            b = [];
            Aeq = zeros(nhx+nhy,2*nh);
            beq = zeros(nhx+nhy,1);
            for i = 1:nhy
                Aeq(i,idxs_at_Y_axis(i)) = 1;
                beq(i) = 0.5;
            end
            for i = 1:nhx
                Aeq(nhy+i,nh+idxs_at_X_axis(i)) = 1;
                beq(nhy+i)=0.5;
            end

            nlc = @(sv) nonlcon(sv,u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh);
            options = optimoptions('fmincon','Algorithm','sqp'...
                                  ,'UseParallel',true...
                                  ,'Display','iter-detailed'...
                                  ,'ConstraintTolerance',1e-4...
                                  ,'StepTolerance',1e-10...
                                  ,'OptimalityTolerance',1e-5...
                                  ,'SpecifyObjectiveGradient',false...
                                  ,'CheckGradients',false...
                                  ,'FiniteDifferenceType','central'...
                                  ,'SpecifyConstraintGradient',false);

            [s_opt,~,~,OUT] = fmincon(@(sv) mgt2(sv,u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,sh),s_opt,A,b,Aeq,beq,lb,ub,nlc,options);
            s(idxs_hooked,:)=s_opt;


            iters = OUT.iterations;
            iters_tot(end)=iters_tot(end)+iters;
        else 
            iters=1;
            iters_tot(end)=iters_tot(end)+1;
            if isempty(actives)
                ui = zeros(3*length(X),1,'mp');
                ui(3:3:end) = 0.0*(t+dt);
                err=norm(u-ui)
            end

        end

        uN = NEWTON(@(uv) mfk_with_constr(uv,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,s_opt,sh),u,free_ind);
        if isnan(uN)
            u = BFGScache(@(uv) mf_with_constr(uv,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,s_opt,sh),u,free_ind);
        else
            u=uN;
        end

        delete(ptrs);
        delete(psph);
        ptrs = plotTruss(u,X,conn);
        psph = plotsphere(sph);
        drawnow

        [actives, sh, Redo] = checkContact(u, sph, actives, sh, s, X, dofs, slaves, conn, ns, mu,ratio);

        if Redo     % only if nodes entered/exited
            % u = u_pre;
            trials(end)=trials(end)+1;
            s = s_pre;
            ds = ds_pre;

        else
            % u_pre=u;
            t=t+dt;
            iters_last=[iters_last,iters];
            xs = X(slaves,:) + u(dofs(slaves,:));
            unused = setdiff(1:ns,idxs_hooked);
            s(unused,:)=getProjs(xs(unused,:),sph);
            ds_pre = ds;
            ds = s-s_pre;
            s_pre = s;
        end

        save('saved_vars');

    end

    % compute force 'f' for obtained displacement 'u'
    [~,f] = mfk(u,X,conn,free_ind,ratio);


    % Update storage files with displacements and forces
    outfile.u_tot(incr,:) = u;
    outfile.f_tot(incr,:) = f;
    outfile.act_slaves(incr,:) = ismember(slaves,actives);

    % create and save image
    clf(nfig,'reset')
    view(37.5,30.0)
    axis equal
    ptrs = plotTruss_XY_symm(u,X,conn,false);
    psph = plotsphere(sph);
    zlim([-1.0 3.0])
    saveas(nfig,strcat(num2str(incr), '.png'))

end
