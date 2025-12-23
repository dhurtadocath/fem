function [X, C] = HexaNet(nx, ny, h)
    C = [];
    X = zeros(2*nx*(2*ny+1), 2);
    % X((nx + 1) * (ny + 1), :) = [nx * dx, ny * dy];
    
    Ny = 2*ny+1;
    node_count=0;

    a = h*sqrt(3)/3;
    for i = 1:nx
        %              <
        % Panel type:A <
        %              <
        Ca = [];
        for j = 1:ny
            X(node_count+j,:)=[(i-1)*3*a (2*j-1)*h/2];
            X(node_count+ny+j,:)=[(i-1)*3*a+a/2 (j-1)*h];
            if j==ny
                X(node_count+ny+j+1,:)= [(i-1)*3*a+a/2 (j)*h];
            end
            Ca = [Ca; node_count+j,node_count+ny+j;node_count+j,node_count+ny+j+1];
        end
        
        %                         -
        %   Intermediate lines    -
        %                         -
        node_count = node_count+Ny;
        for j = 1:ny+1
            Ca = [Ca; node_count-ny-1+j,node_count+j];
        end

        %              >
        % Panel type:B >
        %              >
        Cb = [];
        for j = 1:ny
            X(node_count+j,:)=[(i-1)*3*a+3*a/2 (j-1)*h];
            if j==ny
                X(node_count+ny+1,:)= [(i-1)*3*a+3*a/2 (j)*h];
            end
            X(node_count+ny+1+j,:)=[(i-1)*3*a+2*a (2*j-1)*h/2];
            Cb = [Cb; node_count+j,node_count+ny+j+1;node_count+j+1,node_count+ny+j+1];
        end
        node_count = node_count+Ny;
        C = [C;Ca;Cb];

        if i~=nx
            %                         -
            %   Intermediate lines    -
            %                         -
            for j = 1:ny
                C = [C; node_count-ny+j,node_count+j];
            end
        end

    end

end