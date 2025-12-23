function [X, C] = HexaNet(nx, ny, h)
    C = [];
    X = zeros(2*nx*(2*ny+1), 2);
    % X((nx + 1) * (ny + 1), :) = [nx * dx, ny * dy];
    
    Ny = (ny+1)/2;
    node_count=0;

    a = h*sqrt(3)/3;



    %%%%%%%%%%%%%%%%%%      =>
    % Panel type:B_ext      _>
    %%%%%%%%%%%%%%%%%%      _>
    Cbe = [];
    for j = 1:Ny
        X(j,:)     = [0     h*(j-1/2)];
        X(Ny+j,:)  = [a/2   h*(j-1/2)];
        X(2*Ny+j,:)= [a     h*(j -1 )];
        Cbe = [Cbe; j,Ny+j; Ny+j,2*Ny+j];
        if j~=Ny
           Cbe = [Cbe; Ny+j,2*Ny+j+1];
        end
    end
    node_count = node_count+3*Ny;
    C = [C;Cbe];

    %%%%%%%%%%%%%%%%%%%%%%    -
    %   Intermediate lines    -
    %%%%%%%%%%%%%%%%%%%%%%    -
    for j = 1:Ny
        Ca = [Ca; 2*Ny+j,3*Ny+j];
    end

  Nx = (nx-1)/2;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:Nx
        %%%%%%%%%%%%%% <
        % Panel type:A <
        %%%%%%%%%%%%%% <
        Ca = [];
        for j = 1:ny
            X(node_count+j,:)=[a*(3*i-1) h*(j-1)];
            X(node_count+Ny+j,:)=[a*(3*i-1/2) h*(j-1)];
            if j~=Ny:
                Ca = [Ca; node_count+j,node_count+Ny+j;node_count+Ny+j,node_count+j+1];
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%    -
        %   Intermediate lines    -
        %%%%%%%%%%%%%%%%%%%%%%    -
        node_count = node_count+Ny;
        for j = 1:ny+1
            Ca = [Ca; node_count-ny-1+j,node_count+j];
        end

        %%%%%%%%%%%%%% >
        % Panel type:B >
        %%%%%%%%%%%%%% >
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
            %%%%%%%%%%%%%%%%%%%%%%    -
            %   Intermediate lines    -
            %%%%%%%%%%%%%%%%%%%%%%    -
            for j = 1:ny
                C = [C; node_count-ny+j,node_count+j];
            end
        end

    end

end