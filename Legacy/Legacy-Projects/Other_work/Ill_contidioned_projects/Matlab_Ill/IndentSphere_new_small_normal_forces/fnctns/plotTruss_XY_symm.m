function [p]=plotTruss_XY_symm(u,FE_node_coord,FE_node_nrs,dispNodes,p,STRAINS,time)
    if nargin<5
        p = hggroup();
    end


    if nargin>5
        X_I = FE_node_coord;
        p = plotTruss(u,X_I,FE_node_nrs,dispNodes,p,STRAINS,time,0);
    
        X_II = X_I;
        X_II(:,1) = -1*X_I(:,1);
        u_II = u;
        u_II(1:3:end) = -1*u(1:3:end);
    
        p = plotTruss(u_II,X_II,FE_node_nrs,dispNodes,p,STRAINS,time,0);
    
        X_III = X_II;
        X_III(:,2) = -1*X_II(:,2);
        u_III = u_II;
        u_III(2:3:end) = -1*u_II(2:3:end);
        p = plotTruss(u_III,X_III,FE_node_nrs,dispNodes,p,STRAINS,time,0);
    
        X_IV = X_I;
        X_IV(:,2) = -1*X_I(:,2);
        u_IV = u;
        u_IV(2:3:end) = -1*u(2:3:end);
        p = plotTruss(u_IV,X_IV,FE_node_nrs,dispNodes,p,STRAINS,time,1);
    else
        X_I = FE_node_coord;
        p = plotTruss(u,X_I,FE_node_nrs,dispNodes,p);
    
        X_II = X_I;
        X_II(:,1) = -1*X_I(:,1);
        u_II = u;
        u_II(1:3:end) = -1*u(1:3:end);
    
        p = plotTruss(u_II,X_II,FE_node_nrs,dispNodes,p);
    
        X_III = X_II;
        X_III(:,2) = -1*X_II(:,2);
        u_III = u_II;
        u_III(2:3:end) = -1*u_II(2:3:end);
        p = plotTruss(u_III,X_III,FE_node_nrs,dispNodes,p);
    
        X_IV = X_I;
        X_IV(:,2) = -1*X_I(:,2);
        u_IV = u;
        u_IV(2:3:end) = -1*u(2:3:end);
        p = plotTruss(u_IV,X_IV,FE_node_nrs,dispNodes,p);
    end

end