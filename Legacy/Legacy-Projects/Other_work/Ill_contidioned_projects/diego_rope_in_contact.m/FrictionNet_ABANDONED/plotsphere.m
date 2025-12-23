function nfig=plotsphere(xR,yR,zR,R,fig)
    if nargin>4
        figure(fig)
    else
        figure()
    end
        hold on
    view(37.5,30.0)
    axis equal
    
    [X,Y,Z]=sphere(200);

    positiveZIndices = Z > 0;  % Indices where Z is positive
    X(positiveZIndices) = NaN;  % Set X values of positive Z to NaN
    Y(positiveZIndices) = NaN;  % Set Y values of positive Z to NaN
    Z(positiveZIndices) = NaN;  % Set Z values of positive Z to NaN
    
    X=X*R+xR;
    Y=Y*R+yR;
    Z=Z*R+zR;
    
    h = surf(double(X),double(Y),double(Z), 'EdgeColor', 'none', 'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.5);  % Gray color
    set(h, 'FaceLighting', 'none');  % Turn off lighting effects
    % camproj('perspective');  % Use perspective projection for a smoother rotation
    
    nfig=get(gcf,'Number');
    
end