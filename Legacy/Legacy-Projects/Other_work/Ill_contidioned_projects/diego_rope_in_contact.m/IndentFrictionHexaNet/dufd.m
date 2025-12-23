function duds=dufd(s,u0,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,sh, eps)

    idxs_hooked = find(all(sh ~= -1, 2))';
    nh = length(idxs_hooked);
    ndofs = length(u0);

    duds = zeros(ndofs,2*nh);
    
    uN = NEWTON(@(uv) mfk_with_constr(uv,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,s,sh),u0,fr);
    if isnan(uN)
        u0 = BFGS(@(uv) mf_with_constr(uv,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,s,sh),u0,fr);
    else
        u0=uN;
    end

    for i=1:2*nh
        s_i = s;
        s_i(i) = s(i)+eps;

        uN = NEWTON(@(uv) mfk_with_constr(uv,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,s_i,sh),u0,fr);
        if isnan(uN)
            u = BFGS(@(uv) mf_with_constr(uv,X,conn,fr,idx_act,mu,ratio,k_pen,sph,slaves,dofs,s_i,sh),u0,fr);
        else
            u=uN;
        end
        duds(:,i) = (u-u0)/eps;
    end
    
end
