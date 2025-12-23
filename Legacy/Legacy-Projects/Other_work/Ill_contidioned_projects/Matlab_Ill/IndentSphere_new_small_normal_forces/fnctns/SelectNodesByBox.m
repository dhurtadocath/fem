function selectedNodes = SelectNodesByBox(X, xyz_a, xyz_b)
    nodes = [];
    xa = xyz_a(1);
    ya = xyz_a(2);
    za = xyz_a(3);
    xb = xyz_b(1);
    yb = xyz_b(2);
    zb = xyz_b(3);
    
    for node_id = 1:size(X, 1)
        x = X(node_id, 1);
        y = X(node_id, 2);
        z = X(node_id, 3);
        
        if xa <= x && x <= xb && ya <= y && y <= yb && za <= z && z <= zb
            nodes = [nodes, node_id];
        end
    end
    
    selectedNodes = nodes;
end