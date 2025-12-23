function [act, shnew, Redo] = checkContact(u, sph, idx_act, sh, s, s_pre, X, dofs, slaves, conn, ns, mu,ratio)
    % Now I want to check if gn<0 for the not actives and if Fn>1e-10 for 
    % the active ones.

    Cxy = cell2mat(sph{1});
    Cr = sph{2};

    Fn_eps = 1e-4;
    gn_eps = 1e-5;


    if nargin<12
        ratio=1;
    end
    
    xs = X(slaves,:) + u(dofs(slaves,:));
    act = [];
    shnew = -1*ones(ns,2);          % '-1' indicates no contact
    Redo = false;
    [~,fint] = mfk(u,X,conn,dofs,ratio);
    exited=[];

    for idx = 1:ns

        % NOT ACTIVE
        if ~ismember(idx,idx_act)
            xsi = xs(idx,:);
            gn = norm(xsi - Cxy) - Cr;
            % check for 'significant' penetration
            if gn<-gn_eps
                fprintf('node %d just entered (gn=%0.4e). REDO...\n', slaves(idx),gn);
                Redo = true;
                % in this case I want to keep sh(i) = (-1,-1)
                % sh(idx,:) = s(idx,:); ...so I keep this commented
            else
                continue   % to skip adding this node to 'act'
            end


        % ACTIVE
        else
            si = s(idx,:);
            theta = pi*(1-si(2));
            phi = pi*si(1);
            nor = -[cos(phi)*sin(theta); cos(theta); sin(theta)*sin(phi)];
            fnode = fint(dofs(slaves(idx),:));
            Fn = -fnode'*nor;   % with the '-', it will give Fn>0 for the pulling case


            if Fn<-Fn_eps               % non-negligible normal contact force
                if all(sh(idx,:)==[-1,-1])      % = was not frictional
                    fprintf('node %d exceeded Fn_lim (Fn=%0.4e) REDO...\n', slaves(idx),Fn);
                    Redo = true;
                    % sh(idx,:)=si;
                    sh(idx,:)=s_pre(idx,:);
                else
                    f_proj = fnode - (fnode' * nor) * nor;
                    plasticity = norm(f_proj)/abs(mu * (fnode' * nor));
                    if plasticity <= 0.99   % elastic case
                        fprintf('node %d elastic. (ratio %0.4e )\n', slaves(idx),plasticity);
                    else
                        fprintf('node %d plastic\n', slaves(idx));
                    end
                    shnew(idx,:)=si;
                end
    
            elseif fnode'*nor>Fn_eps     % non-negligible normal pulling force
                fprintf('node %d exited(Fn=%0.4e). REDO...\n', slaves(idx),Fn);
                Redo = true;
                exited=[exited,idx];
                continue    % to skip adding this node to 'act'


            else                        % negligible normal force. --> frictionless
                if ~all(sh(idx,:)==[-1,-1])
                    fprintf('node %d now frictionless(Fn=%0.4e). REDO...\n', slaves(idx),Fn);
                    sh(idx,:)=[-1,-1];
                    Redo = true;
                else
                    fprintf('node %d frictionless.\n', slaves(idx));
                end
            end

        end
        act = [act, idx];

    end
    if Redo
        shnew=sh;
        shnew(exited,:)=-1*ones(length(exited),2);
    end
end
