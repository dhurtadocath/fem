function [cc,cl,dccds,dclds]=nonlcon(s,u0,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,sh)

    % u =   BFGS(@(uv) mfk_with_constr(uv,X,conn,free_ind,idx_act,mu,ratio,k_pen,sph,slaves,dofs,s,sh),u0,free_ind);
    u = NEWTON(@(uv) mfk_with_constr(uv,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,s,sh),u0,fr);    
    ns = height(s);
    Cxy = cell2mat(sph{1})';
    R  = sph{2};
    cc = zeros(ns,1);       % coulomb's condition
    cl = zeros(ns,1);       % force alignment condition
    cnt_act = 1;
    if nargout>2
        dccds = zeros(ns,2*ns);
        dclds = zeros(ns,2*ns);
        [~,fint,kint] = mfk(u,X,conn,fr,ratio);
        duds = get_duds(u,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,s,sh);
    else
        [~,fint] = mfk(u,X,conn,fr,ratio);
    end

    for ids=idx_act
        if ~all(sh(ids,:)==[-1,-1])
            if nargout>2
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

            if (xc-xch)==0.0
                cl(cnt_act) =  0.0;
            else
                dsh = xc-xch;
                Taup = dsh-(dsh'*nor)*nor;     % projection vector pointing TOWARDS the hook
                Taup = -Taup;
                Tau = Taup/norm(Taup);     % normalized vector on the 'nor'-plane pointing towards the hook
                cl(cnt_act) =  Ft-ft'*Tau;
            end
            cc(cnt_act) = Ft-mu*Fn;    % it must be <0

            % derivatives (if required)
            if nargout>2
                % coulombs constraint
                dccdfn = -mu*fn'/Fn;
                dccdft = ft'/Ft;
                dfndf = nor*nor';
                dfndn = (f'*nor)*eye(3)+f*nor';
                dftdf = eye(3);
                dftdfn = -eye(3);
                dfdu = kint(dofi,:);

                dccds(cnt_act,2*cnt_act-1:2*cnt_act) = dccds(cnt_act,2*cnt_act-1:2*cnt_act) ...
                    + (dccdfn+dccdft*dftdfn)*dfndn*dnds;
                dccds(cnt_act,:) = dccds(cnt_act,:) ...
                    + ((dccdfn+dccdft*dftdfn)*dfndf+dccdft*dftdf)*dfdu*duds;

                % line constraint
                dcldft = ft/Ft - Tau;
                dcldtau = -ft;
                DftDf = (dftdf+dftdfn*dfndf);   % total derivative
                DftDn = dftdfn*dfndn;           % total derivative
                dtaudtaup = (eye(3)-Tau*Tau')/norm(Taup);
                dtaupdxc = -DftDf;
                dxcds = R*dnds;
                dtaupdn = -(dsh'*nor)*eye(3)-dsh*nor';
                dclds(cnt_act,2*cnt_act-1:2*cnt_act) = dclds(cnt_act,2*cnt_act-1:2*cnt_act)...
                    + (dcldft'*DftDn+dcldtau'*dtaudtaup*dtaupdn)*dnds...
                    + dcldtau'*dtaudtaup*dtaupdxc*dxcds;
                dclds(cnt_act,:) = dclds(cnt_act,:)...
                    + dcldft'*DftDf*dfdu*duds;

            end
            cnt_act = cnt_act + 1;        % counter must continue even if later happens that gn>0
        end
    end

    if nargout>2
        dccds = dccds';
        dclds = dclds';
    end

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
    dfds = reshape(dfds,ndofs,2*ns);    % reshaping dfds
    duds(fr,:)=linsolve(KK(fr,fr),-dfds(fr,:));

end