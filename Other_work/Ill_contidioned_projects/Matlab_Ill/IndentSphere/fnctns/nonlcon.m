function [cc,asd,u,dccds,sdf]=nonlcon(s,u0,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,sh)
% function [asd,cl,sdf,dclds]=nonlcon(s,u0,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,sh)
% function [cc,cl,dccds,dclds]=nonlcon(s,u0,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,sh)

    % u =   BFGS(@(uv) mf_with_constr(uv,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,s,sh),u0,fr);
    % u = NEWTON(@(uv) mfk_with_constr(uv,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,s,sh),u0,fr);  

    idxs_hooked = find(all(sh ~= -1, 2))';
    nh = length(idxs_hooked);
    % ns = height(s);
    Cxy = cell2mat(sph{1})';
    R  = sph{2};

    % dofs_hooked = dofs(idxs_hooked,:);
    ih_count = 0;
    for idxh=idxs_hooked
        node = slaves(idxh);
        dofi = dofs(node,:);
        ih_count = ih_count+1;
        si = s(ih_count,:);
        nor = get_normals(si);
        xc = Cxy+R*nor;
        u_imposed = xc-X(node,:)';
        u0(dofi) = u_imposed;
        fr=setdiff(fr,dofi);
    end

    uN = NEWTON(@(uv) mfk_with_constr(uv,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,s,sh),u0,fr);
    if isnan(uN)
        % global BFGScache;
        % u = BFGScache(@(uv) mf_with_constr(uv,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,s,sh),u0,fr);

        is_active = ismember(slaves,slaves(idx_act));     %before: 'actives'
        is_free = ismember(dofs,fr);
        Cx = sph{1}{1};
        Cy = sph{1}{2};
        Cz = sph{1}{3};
        R = sph{2};
        s_full = zeros(length(slaves),2);
        s_full(idxs_hooked,:)=s;

        % tic
        % u = customBFGSMex(u0,X,conn,is_free,is_active,mu,ratio,k_pen,Cx,Cy,Cz,R,slaves,dofs,s_full,sh);
        % toc
        % tic
        % u_old = fullBFGStoCompileToMex(u0,X,conn,is_free,is_active,mu,ratio,k_pen,Cx,Cy,Cz,R,slaves,dofs,s_full,sh);
        % toc
        % tic
        % u = new_fullBFGStoCompileToMex(u0,X,conn,is_free,is_active,mu,ratio,k_pen,Cx,Cy,Cz,R,slaves,dofs,s_full,sh);
        % toc
        u = fullBFGStoCompileToMex_mex(u0,X,conn,is_free,is_active,mu,ratio,k_pen,Cx,Cy,Cz,R,slaves,dofs,s_full,sh);
        0;
    else
        u=uN;
    end




    cc = zeros(nh,1);       % coulomb's condition
    cl = zeros(nh,1);       % force alignment condition
    cnt_act = 1;
    if nargout>3
        dccds = zeros(nh,2*nh);
        dclds = zeros(nh,2*nh);
        [~,fint,kint] = mfk(u,X,conn,fr,ratio);
        duds = get_duds(u,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,s,sh);

        % if nh>1
        %     0;
        % end

    else
        [~,fint] = mfk(u,X,conn,fr,ratio);
    end

    for ids=idxs_hooked
        if nargout>3
            [nor,dnds] = get_normals(s(cnt_act,:));
        else
            nor = get_normals(s(cnt_act,:));
        end
        nor_h = get_normals(sh(ids,:));
        xc = Cxy+R*nor;
        xch= Cxy+R*nor_h;
   
        dofi = dofs(slaves(ids),:);
        f = fint(dofi);
        Fn = (f'*nor);
        fn = Fn*nor;
        ft = f - fn;
        Ft = norm(ft);

        if (xch-xc)==0.0
            cl(cnt_act) =  0.0;
            Tau = 0;        % Check this in the future
        else
            dsh = xc-xch;
            Taup = dsh-(dsh'*nor)*nor;     % projection vector pointing TOWARDS the hook
            Taup = -Taup;
            Tau = Taup/norm(Taup);     % normalized vector on the 'nor'-plane pointing towards the hook
            cl(cnt_act) =  Ft-ft'*Tau;
        end
            % cc(cnt_act) = Ft-mu*abs(Fn);    % it must be <0
            % cc(cnt_act) = (Ft-mu*abs(Fn))^3;    % it must be <0
            % cc(cnt_act) = Ft/abs(Fn)-mu;    % it must be <0
            cc(cnt_act) = 1-mu*abs(Fn)/Ft;    % it must be <0

            % if Fn==0
            %     cc(cnt_act) = 0;    % it must be <0
            % else
            %     cc(cnt_act) = Ft/(mu*abs(Fn))-1;    % it must be <0
            % end

        % derivatives (if required)
        if nargout>3 && norm(f)>0
            % if nh==3
            %   0;
            % end

            % coulombs constraint
            % dccdfn = -mu*fn'/abs(Fn);
            % dccdft = ft'/Ft;
            % dfndf = nor*nor';
            % dfndn = (f'*nor)*eye(3)+f*nor';
            % dftdf = eye(3);
            % dftdfn = -eye(3);
            % dfdu = kint(dofi,:);

            % dccds(cnt_act,2*cnt_act-1:2*cnt_act) = dccds(cnt_act,2*cnt_act-1:2*cnt_act) ...
            %     + (dccdfn+dccdft*dftdfn)*dfndn*dnds;
            
            % dccds(cnt_act,[cnt_act,nh+cnt_act]) = dccds(cnt_act,[cnt_act,nh+cnt_act]) ...
            %     + (dccdfn+dccdft*dftdfn)*dfndn*dnds;
            % dccds(cnt_act,:) = dccds(cnt_act,:) ...
            %     + ((dccdfn+dccdft*dftdfn)*dfndf+dccdft*dftdf)*dfdu*duds;

            dccdFt = 1;
            dFtdft = ft'/norm(ft);
            dftdf  = eye(3);
            dftdfn = -eye(3);
            dfndFn = nor;
            dfndn  = Fn*eye(3);
            dFndf  = nor';
            dFndn  = f';
            dccdabsFn = -mu;
            dabsFndFn = sign(Fn);
            dfdu = kint(dofi,:);




            dccds(cnt_act,[cnt_act,nh+cnt_act]) = dccds(cnt_act,[cnt_act,nh+cnt_act]) ...
                + (dccdFt*dFtdft*dftdfn*(dfndn+dfndFn*dFndn)+dccdabsFn*dabsFndFn*dFndn)*dnds;
            dccds(cnt_act,:) = dccds(cnt_act,:) ...
                + (dccdFt*dFtdft*(dftdf+dftdfn*dfndFn*dFndf)+dccdabsFn*dabsFndFn*dFndf)*dfdu*duds;




            dfndf = nor*nor';
            dfndn = (f'*nor)*eye(3)+f*nor';


            % line constraint
            dcldft = ft/Ft - Tau;
            dcldtau = -ft;
            DftDf = (dftdf+dftdfn*dfndf);   % total derivative
            DftDn = dftdfn*dfndn;           % total derivative
            dtaudtaup = (eye(3)-Tau*Tau')/norm(Taup);
            dtaupdxc = -DftDf;
            dxcds = R*dnds;
            dtaupdn = -(dsh'*nor)*eye(3)-dsh*nor';
            % dclds(cnt_act,2*cnt_act-1:2*cnt_act) = dclds(cnt_act,2*cnt_act-1:2*cnt_act)...
            %     + (dcldft'*DftDn+dcldtau'*dtaudtaup*dtaupdn)*dnds...
            %     + dcldtau'*dtaudtaup*dtaupdxc*dxcds;
            dclds(cnt_act,[cnt_act,nh+cnt_act]) = dclds(cnt_act,[cnt_act,nh+cnt_act])...
                + (dcldft'*DftDn+dcldtau'*dtaudtaup*dtaupdn)*dnds...
                + dcldtau'*dtaudtaup*dtaupdxc*dxcds;
            dclds(cnt_act,:) = dclds(cnt_act,:)...
                + dcldft'*DftDf*dfdu*duds;

            if sum(sum(isnan(dccds)))~=0
                0;
            end

        end
        cnt_act = cnt_act + 1;        % counter must continue even if later happens that gn>0
    end

    % 
    % % Comment this (only for checking purposes)
    % fprintf('\n');
    % fprintf('c_line   : ');
    % for i = 1:numel(cl)
    %     fprintf(' %0.2e', cl(i));
    % end
    % fprintf('\t\t');
    % fprintf('c_Coulomb: ');
    % for i = 1:numel(cc)
    %     fprintf(' %0.2e', cc(i));
    % end
    % 
    % if nargout>2
    %     fprintf('\n');
    %     fprintf('dc_line   : \n');
    %     for i = 1:numel(dclds(:,1))
    %         fprintf(' %0.2e', dclds(i,:));
    %         fprintf('\n');
    %     end
    %     fprintf('dc_Coulomb: \n');
    %     for i = 1:numel(dccds(:,1))
    %         fprintf(' %0.2e', dccds(i,:));
    %         fprintf('\n');
    %     end
    %     fprintf('\n');
    % end
    % 
        
    if nargout>3
        dccds = dccds';
        dclds = dclds';
    end
asd=0;
sdf=zeros(2*nh,1);
end

function [n,dnds] = get_normals(s)
    theta = pi*(1-s(2));
    phi = pi*s(1);
    n = -[cos(phi)*sin(theta); cos(theta); sin(theta)*sin(phi)];
    if nargout>1
        dndtheta = -[cos(phi)*cos(theta); -sin(theta);cos(theta)*sin(phi)];
        dndphi   = -[-sin(phi)*sin(theta);       0.0 ;sin(theta)*cos(phi)];
        dnds = pi*[dndphi,-dndtheta];
    end
end

function duds = get_duds(u,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,s,sh)
    R  = sph{2};
    ns = height(s);
    ndofs = length(u);
    [~,~,KK] = mfk_with_constr(u,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,s,sh);
    dfds = zeros(ndofs,ns,2);
    duds = zeros(ndofs,2*ns);
    cnt_act=1;
    for ids=idx_act
        if ~all(sh(ids,:)==[-1,-1])
            si = s(cnt_act,:);
            [~,dnds]=get_normals(si);
            dxcds = R*dnds;
            a = dofs(slaves(ids),:)';
            dfds(a,cnt_act,:) = dfds(a,cnt_act,:) - k_pen*permute(dxcds,[1 3 2]) ;
            cnt_act = cnt_act + 1;
        end
    end

    % if ns>1
    %         0;
    % end

    
    dfds = reshape(dfds,ndofs,2*ns);    % reshaping dfds
        
    % duds(fr,:)=linsolve(KK(fr,fr),-dfds(fr,:));
    condK = condest(KK);

    if condK>1e20
        duds(fr,:)=lsqminnorm(double(KK(fr,fr)),-dfds(fr,:),1e-20);
        % duds(fr,:)=pinv(KK(fr,fr))*(-dfds(fr,:));
    else
        duds(fr,:)=KK(fr,fr)\(-dfds(fr,:));
    end
end