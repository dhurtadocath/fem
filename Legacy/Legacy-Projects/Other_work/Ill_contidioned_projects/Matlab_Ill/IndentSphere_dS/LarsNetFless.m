clear
close all
clc

% % multiprecision
% addpath('~/AdvanpixMCT-5.0.0.15222')
% mp.Digits(34);
format longG

addpath('fnctns/')

Resume = false;

% outdata_path = [folderName '/OutData.mat'];
% 
% % Check if the file already exists
% if exist(outdata_path, 'file')
%     data = matfile(outdata_path, 'Writable', true);
% else
%     % If the file doesn't exist, create it by saving an empty variable
%     temp = [];
%     save(outdata_path, 'temp','-v7.3');
%     outfile = matfile(outdata_path, 'Writable', true);
% end




if Resume
    folderName = getfolderName();
    addpath(folderName)

	load('saved_vars');
    if Redo
    	latest_incr = incr-1;
    else
        latest_incr = incr;
    end
else
    folderName = createFolder();
    addpath(folderName)

	% SPHERE
    Cx=0.5;
	Cy=0.5;
	Cz=0.75;
	R =0.749;
    indentation = 0.2;
	sph = {{Cx,Cy,Cz},R};


	% NET
    loaded_x   = load("diego_randomnetworks/FE_node_coord100.mat","FE_node_coord");
    loaded_conn= load("diego_randomnetworks/FE_con100.mat","FE_con");
	X = loaded_x.FE_node_coord;
    X = [X zeros(length(X),1)];
    conn = loaded_conn.FE_con;


	ratio = 0.0;      % comp_resist/tens_resist

	% NODES SELECTION
	all_nodes = SelectFlatSide(X,'+z');
	nodes_base= [SelectFlatSide(X,'-x'),SelectFlatSide(X,'+x'),];


	ndofs=length(X)*3;
	dofs=reshape(1:ndofs,3,[])';

	dofs_base_x = dofs(nodes_base,1);
	dofs_base_y = dofs(nodes_base,2);
	dofs_base_z = dofs(nodes_base,3);


	% BOUNDARY CONDITIONS
	bc1 = {dofs_base_x,"dir",0.0, 0.0,1.0};
	bc2 = {dofs_base_y,"dir",0.0, 0.0,1.0};
	bc3 = {dofs_base_z,"dir",0.0, 0.0,1.0};

	BCs = {bc1,bc2,bc3};


	% CONTACT
	k_pen=1000;
	mu = 0.5;
	% slaves = setdiff(all_nodes,[nodes_symmX,nodes_symmY,nodes_base]);
	slaves = setdiff(all_nodes,nodes_base); % If i find a way to relate dofs for the added nodes, then they can be out of the slave group (nodes_symmY)
	ns = length(slaves);

	% MODEL
	t0 = 0.0;
	tf = 1.0;
	nincr=100;
	latest_incr = 0;

	% INITIALLIZE VARIABLES
	u=zeros(ndofs,1);
	u_pre=u;
	actives = [];

    % trying new variables inserted (04-03-2024)
    is_free = zeros(size(X),'logical');
    is_active = zeros(1,ns,'logical');
	
    
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

ptrs=plotTruss(u,X,conn,true);
drawnow

%% Compile Mex file, if necessary
% codegen -config:mex fullBFGStoCompileToMex.m -args {u,X,conn,is_free,is_active,mu,ratio,k_pen,Cx,Cy,Cz,R,slaves,dofs,zeros(ns,2,'double'),sh} -report
%%

% global BFGScache;
% BFGScache = memoize(@BFGS);

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
        % sph{1}{3} = zR+(-1.0)*(t+dt)/tf;
        sph{1}{3} = Cz+(-indentation)*(t+dt)/tf;


        idxs_hooked = find(all(sh ~= -1, 2));
        s_opt = s(idxs_hooked,:)+ds(idxs_hooked,:);

        iters=1;
        iters_tot(end)=iters_tot(end)+1;
        if isempty(actives)
            ui = zeros(3*length(X),1);
            ui(3:3:end) = 0.0*(t+dt);
            err=norm(u-ui)
        end


        uN = NEWTON(@(uv) mfk_with_constr(uv,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,s_opt,sh),u,free_ind);
        if isnan(uN)
            
            is_active = ismember(slaves,slaves(actives));     %before: 'actives'
            is_free = ismember(dofs,free_ind);
            Cx = sph{1}{1};
            Cy = sph{1}{2};
            cz = sph{1}{3};
            R = sph{2};
            s_full = zeros(length(slaves),2);
            s_full(idxs_hooked,:)=s_opt;
            tic
            u = fullBFGStoCompileToMex_mex(u,X,conn,is_free,is_active,mu,ratio,k_pen,Cx,Cy,cz,R,slaves,dofs,s_full,-1*ones(size(s_full),'like',s_full));
            % u = new_fullBFGStoCompileToMex(u,X,conn,is_free,is_active,mu,ratio,k_pen,Cx,Cy,cz,R,slaves,dofs,s_full,-1*ones(size(s_full),'like',s_full));
            toc

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

        save([folderName 'saved_vars']);

    end

    % compute force 'f' for obtained displacement 'u'
    [~,f] = mfk(u,X,conn,free_ind,ratio);


    % Saving data in csv files
    Appendrow([folderName 'displacements.csv'],double(u'))
    Appendrow([folderName 'forces.csv'],f')
    Appendrow([folderName 'actives.csv'],ismember(slaves,slaves(actives)))


    % create and save image
    clf(nfig,'reset')
    view(37.5,30.0)
    axis equal
    ptrs = plotTruss(u,X,conn,false);
    psph = plotsphere(sph);
    zlim([-1.0 3.0])
    saveas(nfig,strcat(folderName, num2str(incr), '.png'))

end
