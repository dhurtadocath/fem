function [p]=plotTruss_XY_symm(u,FE_node_coord,FE_node_nrs,dispNodes,p)
    if nargin<5
        p = hggroup();
    end

    X_I = FE_node_coord;
    p = plotTruss(u,X_I,FE_node_nrs,dispNodes,p);

    X_II = X_I;
    X_II(:,1) = -1*X_I(:,1);
    p = plotTruss(u,X_II,FE_node_nrs,false,p);

    X_III = X_II;
    X_III(:,2) = -1*X_I(:,2);
    p = plotTruss(u,X_III,FE_node_nrs,false,p);

    X_IV = X_I;
    X_IV(:,2) = -1*X_I(:,2);
    p = plotTruss(u,X_IV,FE_node_nrs,false,p);

end