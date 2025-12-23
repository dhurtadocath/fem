classdef DefaultOptionValues
    % Default option values that may need to be referenced across various
    % files/components
    %
    % FOR INTERNAL USE ONLY -- This feature is intentionally undocumented.
    % Its behavior may change, or it may be removed in a future release.

    % Copyright 2022 The MathWorks, Inc.

    properties (Constant)

        ProblemdefOptions = struct(...
            'FromSolve', false, ...
            'FunOnWorkers', true, ...
            'ObjectiveMultiplier', 1, ...
            'ObjectiveOffset', 0);
    end
end
