function [act, shnew, Redo] = checkContact(u, sph, idx_act, sh, s, X, dofs, slaves, conn, ns, mu,ratio)
    % Now I want to check if gn<0 for the not actives and if Fn>1e-10 for 
    % the active ones.

    Cxy = cell2mat(sph{1});
    Cr = sph{2};

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

        if ~ismember(idx,idx_act)
            xsi = xs(idx,:);
            gn = norm(xsi - Cxy) - Cr;
            if gn<-1e-6
                fprintf('node %d just entered. Increment will Redo...\n', slaves(idx));
                Redo = true;
                sh(idx,:) = s(idx,:);
                % sh(idx,:) = 0.5*(si+getProjs(xsi,sph)) ;
            else
                continue
            end
        else

            si = s(idx,:);
            theta = pi*(1-si(2));
            phi = pi*si(1);
            nor = -[cos(phi)*sin(theta); cos(theta); sin(theta)*sin(phi)];
            fnode = fint(dofs(slaves(idx),:));

            if fnode'*nor>1e-10
                fprintf('node %d exited. Increment will Redo...\n', slaves(idx));
                Redo = true;
                exited=[exited,idx];
                continue
            else
                shnew(idx,:) = si;

                f_proj = fnode - (fnode' * nor) * nor;
                plasticity = norm(f_proj)/abs(mu * (fnode' * nor));
                if plasticity <= 0.99   % elastic case
                    fprintf('node %d elastic. (ratio %.4f )\n', slaves(idx),plasticity);
                else
                    fprintf('node %d plastic\n', slaves(idx));
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
