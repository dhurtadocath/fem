function [X, C] = Net2D(nx, ny, dx, dy)
    C = [];
    X = zeros((nx + 1) * (ny + 1), 2,'mp');
    X((nx + 1) * (ny + 1), :) = [nx * dx, ny * dy];
    
    for i = 1:ny
        for j = 1:nx
            a = (nx + 1) * (i - 1) + j;
            b = (nx + 1) * (i - 1) + j + 1;
            c = (nx + 1) * i + j;
            d = (nx + 1) * i + j + 1;

            if i == 1
                C = [C; [a, b]];
            end
            if j == 1
                C = [C; [a, c]];
            end
            C = [C; [b, d]; [c, d]];

            X(a, :) = [(j - 1) * dx, (i - 1) * dy];
            if j == nx
                X(b, :) = [j * dx, (i - 1) * dy];
            end
            if i == ny
                X(c, :) = [(j - 1) * dx, i * dy];
            end
        end
    end
end