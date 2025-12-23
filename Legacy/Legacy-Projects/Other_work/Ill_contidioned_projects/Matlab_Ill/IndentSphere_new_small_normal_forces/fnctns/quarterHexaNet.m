function [X_new, C_new] = quarterHexaNet(nx, ny, h)

    [X, C] = HexaNet(nx, ny, h);
    a = h*sqrt(3)/3;
    DX = a*(3*nx-1);
    DY = ny*h;
    X = displace(X,[-DX/2,-DY/2]);     % centering the HexaNet

    ind_x_pos = find(X(:,1) >= 0 & X(:,2) >= 0);
    ind_c_inn = find(ismember(C(:,1), ind_x_pos) & ismember(C(:,2), ind_x_pos));
    
    Ny = (ny+1)/2;
    X_new = [[zeros(1,Ny); double(h/2:h:h*(Ny-1/2))]'; X(ind_x_pos,:)];

    C_new = zeros(Ny+length(ind_c_inn),2);

    for i=1:Ny
        C_new(i,:)=[i,Ny+i];
    end

    for i = 1 : length(ind_c_inn)
        C_new(Ny+i,:) = [find(ind_x_pos==C(ind_c_inn(i),1))+Ny   find(ind_x_pos==C(ind_c_inn(i),2))+Ny];
    end

    Youngs = ones(length(C_new),1);
    corr_duplicates = X_new(C_new(:,1),2)==0 & X_new(C_new(:,2),2)==0;
    Youngs = Youngs - 0.5*corr_duplicates;

    C_new = [C_new,Youngs];

end