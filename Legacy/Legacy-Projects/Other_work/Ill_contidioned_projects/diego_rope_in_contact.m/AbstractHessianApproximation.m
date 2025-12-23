classdef (Abstract) AbstractHessianApproximation
    % Base class of Hessian approximation classes used for fminunc's quasi-newton algorithm
    %
    % FOR INTERNAL USE ONLY -- This feature is intentionally undocumented.
    % Its behavior may change, or it may be removed in a future release.

    % Copyright 2021 The MathWorks, Inc.

    methods (Access = public)

        function [this, msg] = update(this, deltaX, deltaGrad, iter)

        % Update Hessian approximation. Use this base class method to implement the
        % template pattern. Concrete classes may overrride updatePreProcessing() method
        % and must implement updateHessian() method

        % Pre-processing
        [this, deltaXDeltaGrad, updateOk] = this.updatePreProcessing(deltaX, deltaGrad, iter);

        % Update the Hessian if ok to proceed
        if updateOk
            this = this.updateHessian(deltaX, deltaGrad, deltaXDeltaGrad);
            msg = '';
        else
            % msg = 'skipped update';
            msg = 'forcing update';
            this = this.updateHessian(deltaX, deltaGrad, deltaXDeltaGrad);
        end
        end
    end

    methods (Access = protected)

        function [this, deltaXDeltaGrad, updateOk] = updatePreProcessing(this, deltaX, deltaGrad, ~)

        % Check update
        deltaXDeltaGrad = deltaX'*deltaGrad;
        updateOk = deltaXDeltaGrad >= sqrt(eps)*max(eps, norm(deltaX)*norm(deltaGrad));
        end
    end

    methods (Abstract, Access = public)

        % Compute search direction
        srchDir = computeSearchDirection(this, grad);
    end

    methods (Abstract, Access = protected)

        % Subclass specific Hessian approximation update
        H = updateHessian(this, deltaX, deltaGrad, deltaXDeltaGrad);
    end
end
