function [u,m_new,iter] = BFGS_mf_constr(u,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,s_opt,sh)

    % Work this out to have fixed-size arrays and transform to MEX
    0;
    actives = ismember(slaves,slaves(actives));
    free_ind = ismember(dofs,free_ind);

    [u,m_new,iter] = BFGS(@(uv) mf_with_constr(uv,X,conn,free_ind,actives,mu,ratio,k_pen,sph,slaves,dofs,s_opt,sh),u,free_ind);

end

