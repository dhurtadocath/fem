classdef BFGSHessianApproximation < optim.internal.fminunc.AbstractDenseHessianApproximation
    % BFGS Hessian approximation class
    %
    % FOR INTERNAL USE ONLY -- This feature is intentionally undocumented.
    % Its behavior may change, or it may be removed in a future release.

    % Copyright 2021 The MathWorks, Inc.

    methods (Access = public)

        function this = BFGSHessianApproximation(nVars)

        % Call superclass constructor
        this = this@optim.internal.fminunc.AbstractDenseHessianApproximation(nVars);
        end
    end

    methods (Access = protected)

        function this = updateHessian(this, deltaX, deltaGrad, deltaXDeltaGrad)

        % BFGS update
        HdeltaGrad = this.Value * deltaGrad;
        % NOTE: the outer product (deltaX*deltaX') is not parenthesized on
        % purpose. This is how the computation was originally done. Adding the
        % parentheses changes the results subtly, creating an incompatibility.
        % Since it doesn't improve the solver, this is undesirable.
        this.Value = this.Value + ...
            (1 + deltaGrad'*HdeltaGrad/deltaXDeltaGrad) * ...
            deltaX*deltaX'/deltaXDeltaGrad - (deltaX*HdeltaGrad' + ...
            HdeltaGrad*deltaX')/deltaXDeltaGrad; %#ok
        end
    end
end
