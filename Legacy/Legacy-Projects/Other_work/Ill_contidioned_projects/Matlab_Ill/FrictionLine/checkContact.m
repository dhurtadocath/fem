function [act, shnew, Redo] = checkContact(u, sph, idx_act, sh, s, X, dofs, slaves, conn, ns, mu,ratio)
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

    for idx = 1:length(xs)
        xsi = xs(idx,:);
        gn = norm(xsi - Cxy) - Cr;

        if gn >= 0   % slave is out
            if ismember(idx, idx_act)   % ... but was in
                fprintf('node %d exited. Increment will Redo...\n', slaves(idx));
                Redo = true;
                exited=[exited,idx];
                continue
            end
        else        % slave is in
            % si = getProjs(xsi,sph);
            si = s(idx,:);
            theta = pi*(1-si(2));
            phi = pi*si(1);
            nor = -[cos(phi)*sin(theta); cos(theta); sin(theta)*sin(phi)];

            if ismember(idx, idx_act)   % was active
                if all(sh(idx,:)==[-1,-1])      % first slide
                    fprintf('node %d first slide\n', slaves(idx));
                    shnew(idx,:) = si;
                    color = 'g';
                else
                    fnode = fint(dofs(slaves(idx),:));
                    f_proj = fnode - (fnode' * nor) * nor;

                    % if norm(f_proj) <= abs(mu * (fnode' * nor))   % elastic case
                    plasticity = norm(f_proj)/abs(mu * (fnode' * nor));
                    if plasticity <= 0.99   % elastic case
                        shnew(idx,:) = sh(idx,:);
                        fprintf('node %d elastic. (ratio %.4f )\n', slaves(idx),plasticity);
                        color = 'b';
                    else
                        shnew(idx,:) = si;
                        fprintf('node %d plastic\n', slaves(idx));
                        color = 'r';

                    end
                    quiver3(xsi(1),xsi(2),xsi(3),fnode(1),fnode(2),fnode(3),color)
                end
                plot3(xsi(1),xsi(2),xsi(3),'o','Color',color)
            else
                fprintf('node %d just entered. Increment will Redo...\n', slaves(idx));
                Redo = true;
            end

            act = [act, idx];
        end
    end
    if Redo
        shnew=sh;
        shnew(exited,:)=-1*ones(length(exited),2);
    end
end
