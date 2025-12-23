function [mOBJ,dmds] = mgt2(s,u0,X,conn,free_ind,idx_act,mu,ratio,k_pen,sph,slaves,dofs,sh)
    Cxy = cell2mat(sph{1})';
    R  = sph{2};
    mOBJ=0;
    cnt_act = 1;
    dmds =zeros(size(s));
    % fprintf('currently trying following s:');
    % for i = 1:numel(s)
    %     fprintf(' %.4f', s(i));
    % end
    % fprintf('\n');
    
    for ids=idx_act
        if ~all(sh(ids,:)==[-1,-1])
            si = s(cnt_act,:);
            % theta = pi*(1-si(2));
            % phi = pi*si(1);
            % nor = -[cos(phi)*sin(theta); cos(theta); sin(theta)*sin(phi)];
            [nor,dnds] = get_normals(si,1);
            shi = sh(ids,:);
            theta_h = pi*(1-shi(2));
            phi_h = pi*shi(1);
            nor_h = -[cos(phi_h)*sin(theta_h); cos(theta_h); sin(theta_h)*sin(phi_h)];
            xc = Cxy+R*nor;
            xch= Cxy+R*nor_h;
    
            mOBJ=mOBJ+(xc-xch)'*(xc-xch);

            if nargout>1
                dxcds = R*dnds;
                dmds(cnt_act,:)=dmds(cnt_act,:)+2*(xc-xch)'*dxcds;
            end
            
            cnt_act = cnt_act + 1;

        end
    end

    % fprintf('. m=%.4f', mOBJ);
    % 
    % fprintf('\n');


end

function [n,dnds] = get_normals(s,ord)
    if nargin<2
        ord=0;
    end
    theta = pi*(1-s(2));
    phi = pi*s(1);
    n = -[cos(phi)*sin(theta); cos(theta); sin(theta)*sin(phi)];
    if ord==1
        dndtheta = -[cos(phi)*cos(theta); -sin(theta);cos(theta)*sin(phi)];
        dndphi   = -[-sin(phi)*sin(theta);       0.0 ;sin(theta)*cos(phi)];
        dnds = pi*[dndphi,-dndtheta];
    else
        dnds=0;
    end
end
