function selectedNodes = SelectFlatSide(X, side, tol)
    if nargin < 3
        tol = 1e-6;
    end
    
    minX = min(X(:, 1));
    maxX = max(X(:, 1));
    minY = min(X(:, 2));
    maxY = max(X(:, 2));
    minZ = min(X(:, 3));
    maxZ = max(X(:, 3));
    eps = tol;
    
    if contains(side, 'x')
        if contains(side, '-')
            selectedNodes = SelectNodesByBox(X, [minX-eps, minY-eps, minZ-eps], [minX+eps, maxY+eps, maxZ+eps]);
        else
            selectedNodes = SelectNodesByBox(X, [maxX-eps, minY-eps, minZ-eps], [maxX+eps, maxY+eps, maxZ+eps]);
        end
    elseif contains(side, 'y')
        if contains(side, '-')
            selectedNodes = SelectNodesByBox(X, [minX-eps, minY-eps, minZ-eps], [maxX+eps, minY+eps, maxZ+eps]);
        else
            selectedNodes = SelectNodesByBox(X, [minX-eps, maxY-eps, minZ-eps], [maxX+eps, maxY+eps, maxZ+eps]);
        end
    else
        if contains(side, '-')
            selectedNodes = SelectNodesByBox(X, [minX-eps, minY-eps, minZ-eps], [maxX+eps, maxY+eps, minZ+eps]);
        else
            selectedNodes = SelectNodesByBox(X, [minX-eps, minY-eps, maxZ-eps], [maxX+eps, maxY+eps, maxZ+eps]);
        end
    end
end