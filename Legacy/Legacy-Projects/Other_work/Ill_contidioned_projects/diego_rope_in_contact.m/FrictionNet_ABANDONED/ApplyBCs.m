function [du, df, diri] = ApplyBCs(tpre, tnow, BCs, ndofs)
    % BCs = {{ dof_list , "dir/neu", val, t0(opt), tf(opt) },{...},...}
    diri = [];
    du = zeros(ndofs, 1,'mp');
    df = zeros(ndofs, 1,'mp');
    
    for i = 1:length(BCs)
        BC = BCs{i};
        if length(BC) == 3
            BC{end+1} = 0;
            BC{end+1} = 1;
        end
        dof_list = BC{1};
        typ = BC{2};
        val = BC{3};
        t0 = BC{4};
        tf = BC{5};
        
        if t0 <= tnow && tnow <= tf || t0 <= tpre && tpre <= tf
            % load factor
            if tpre < t0         % just entered in time interval
                LF = (tnow - t0) / (tf - t0);
            elseif tnow > tf     % just exited time interval
                LF = (tf - tpre) / (tf - t0);
            else                 % going through time interval (normal case)
                LF = (tnow - tpre) / (tf - t0);
            end

            if strcmp(typ, 'dir')     % Dirichlet
                diri = [diri, dof_list];
                if LF ~= 0
                    for j = 1:length(dof_list)
                        dof = dof_list(j);
                        du(dof) = du(dof) + val * LF;
                    end
                end
            elseif LF ~= 0             % Neumann
                for j = 1:length(dof_list)
                    dof = dof_list(j);
                    df(dof) = df(dof) + val * LF;
                end
            end
        end
    end
end