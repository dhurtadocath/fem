classdef LBFGSHessianApproximation < optim.internal.fminunc.AbstractHessianApproximation
    % LBFGS Hessian approximation class
    %
    % FOR INTERNAL USE ONLY -- This feature is intentionally undocumented.
    % Its behavior may change, or it may be removed in a future release.

    % Copyright 2021 The MathWorks, Inc.

    properties (GetAccess = public, SetAccess = protected)

        % Number of most recent iterations to store
        Capacity % (1, 1) double

        % Delta X
        S % (:, :) double % Size: nVars x this.Capacity

        % Delta gradient
        Y % (:, :) double % Size: nVars x this.Capacity

        % Rho: P(k) = 1/(S(:, k)'*Y(:, k)) 
        P % (1, :) double % Size: 1 x this.Capacity

        % Alpha
        A % (1, :) double % Size: 1 x this.Capacity

        % Number of iterations stored so far
        NumIterationsStored = 0; % (1, 1) double
    end

    properties (Constant)

        % Default number of most recent iterations to store.
        % fminusub references this value before constructing
        % a Hessian approximation object
        DefaultCapacity = 10; % (1, 1) double
    end

    methods (Access = public)

        function this = LBFGSHessianApproximation(nVars, capacity)

        % Set Capacity and pre-allocate S, Y, P, and A
        this.Capacity = capacity;
        this.S = zeros(nVars, capacity);
        this.Y = zeros(nVars, capacity);
        this.P = zeros(1, capacity);
        this.A = zeros(1, capacity);
        end

        function r = computeSearchDirection(this, grad)

        % Return gradient direction if no history
        if this.NumIterationsStored == 0
            r = -grad;
            return
        end

        % Loop 1
        q = -grad;
        for k = this.NumIterationsStored:-1:1
            this.A(k) = this.P(k) * this.S(:, k)' * q;
            q = q - this.A(k) * this.Y(:, k);
        end

        % Scale initial search direction
        k = this.NumIterationsStored;
        g = (this.S(:, k)' * this.Y(:, k)) / (this.Y(:, k)' * this.Y(:, k));
        r = g * q;

        % Loop 2
        for k = 1:this.NumIterationsStored
            B = this.P(k) * this.Y(:, k)' * r;
            r = r + this.S(:, k) * (this.A(k) - B);
        end
        end
    end

    methods (Access = protected)

        function this = updateHessian(this, deltaX, deltaGrad, deltaXDeltaGrad)

        % If the number of iterations hasn't reached capacity, increment.
        % Else, shift oldest iteration to right-most column (NumIterationsStored)
        if this.NumIterationsStored < this.Capacity
            this.NumIterationsStored = this.NumIterationsStored + 1;
        else
            this.S = circshift(this.S, -1, 2);
            this.Y = circshift(this.Y, -1, 2);
            this.P = circshift(this.P, -1, 2);
        end

        % Set latest iteration
        this.S(:, this.NumIterationsStored) = deltaX;
        this.Y(:, this.NumIterationsStored) = deltaGrad;
        this.P(this.NumIterationsStored) = 1/deltaXDeltaGrad;
        end
    end
end
