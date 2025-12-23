function [p]=plotsphere(sph,p)

    xR = sph{1}{1};
    yR = sph{1}{2};
    zR = sph{1}{3};
    R = sph{2};

    if nargin<2
        p = hggroup();
    end

    [X,Y,Z]=sphere(200);

    positiveZIndices = Z > 0;  % Indices where Z is positive
    X(positiveZIndices) = NaN;  % Set X values of positive Z to NaN
    Y(positiveZIndices) = NaN;  % Set Y values of positive Z to NaN
    Z(positiveZIndices) = NaN;  % Set Z values of positive Z to NaN
    
    X=X*R+xR;
    Y=Y*R+yR;
    Z=Z*R+zR;
    
    h = surf(double(X),double(Y),double(Z), 'EdgeColor', 'none', 'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.5,'Parent',p);  % Gray color
    set(h, 'FaceLighting', 'none');  % Turn off lighting effects
    % camproj('perspective');  % Use perspective projection for a smoother rotation
    
    % nfig=get(gcf,'Number');
    
end