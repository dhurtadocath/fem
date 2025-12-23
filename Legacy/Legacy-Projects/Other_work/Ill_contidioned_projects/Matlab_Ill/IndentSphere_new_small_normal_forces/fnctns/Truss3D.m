function [X, C] = Truss3D(nx, ny, nz, dx, dy, dz)
    % C = zeros(6*nx*ny*nz + 3*(nx*ny + ny*nz + nx*nz) + nx + ny + nz,1);
    C=[];
    X = zeros((nx+1)*(ny+1)*(nz+1), 3);
    X(1, :) = [0.0, 0.0, 0.0];
    for k = 1:nz
        for j = 1:ny
            for i = 1:nx
                a = (nx+1)*(ny+1)*(k-1) + (nx+1)*(j-1) + (i);
                b = (nx+1)*(ny+1)*(k-1) + (nx+1)*(j-1) + (i) + 1;
                c = (nx+1)*(ny+1)*(k-1) + (nx+1)*(j) + (i) + 1;
                d = (nx+1)*(ny+1)*(k-1) + (nx+1)*(j) + (i);
                e = (nx+1)*(ny+1)*(k) + (nx+1)*(j-1) + (i);
                f = (nx+1)*(ny+1)*(k) + (nx+1)*(j-1) + (i) + 1;
                g = (nx+1)*(ny+1)*(k) + (nx+1)*(j) + (i) + 1;
                h = (nx+1)*(ny+1)*(k) + (nx+1)*(j) + (i);

                tidx = mod(i+j+k+1, 2); % diag type index   0: a,c,f,h      1: b,d,e,g
                if i == 1
                    C = [C; [e, h]; [d, h]];
                    if tidx == 0; C = [C; [a, h]]; else C = [C; [d, e]]; end
                    X(h, :) = [0.0, j*dy, k*dz];
                end
                if j == 1
                    C = [C; [e, f]; [b, f]];
                    if tidx == 0; C = [C; [a, f]]; else C = [C; [b, e]]; end
                    X(f, :) = [i*dx, 0.0, k*dz];
                end
                if k == 1
                    C = [C; [b, c]; [c, d]];
                    if tidx == 0; C = [C; [a, c]]; else C = [C; [b, d]]; end
                    X(c, :) = [i*dx, j*dy, 0.0];
                end

                if i+j == 2
                    C = [C; [a, e]];
                    X(e, :) = [0.0, 0.0, k*dz];
                end
                if i+k == 2
                    C = [C; [a, d]];
                    X(d, :) = [0.0, j*dy, 0.0];
                end
                if j+k == 2
                    C = [C; [a, b]];
                    X(b, :) = [i*dx, 0.0, 0.0];
                end

                diags = [[c, f]; [c, h]; [f, h]]; if tidx == 1; diags = [[b, g]; [d, g]; [e, g]]; end
                C = [C; [g, h]; [f, g]; [c, g]; diags];
                X(g, :) = [(i*dx), j*dy, k*dz];
            end
        end
    end
end