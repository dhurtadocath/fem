function X = displace(X, disp)
    disp = double(disp); % Convert to double if disp is not already
    n = size(X, 1);
    for i = 1:n
        X(i, :) = X(i, :) + disp;
    end
end