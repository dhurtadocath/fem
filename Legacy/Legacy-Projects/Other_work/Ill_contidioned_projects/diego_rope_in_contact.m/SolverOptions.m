classdef (Abstract) SolverOptions < matlab.mixin.CustomDisplay
    %

    %SolverOptions Base class for Optimization Toolbox options
    %   SolverOptions is an abstract class representing a set of options for an
    %   Optimization Toolbox solver. You cannot create instances of this class
    %   directly. You must create an instance of a derived class such
    %   optim.options.Fmincon.
    %
    %   For instances which are scalar, the object display is customized. We
    %   utilize the matlab.mixin.CustomDisplay class to help us create the
    %   display. To use this functionality, SolverOptions inherits from
    %   matlab.mixin.CustomDisplay.
    %
    %   See also MATLAB.MIXIN.CUSTOMDISPLAY

    %   Copyright 2012-2022 The MathWorks, Inc.

    properties (Hidden, Abstract, Access = protected)
        %OPTIONSSTORE Contains the option values and meta-data for the class
        %   OptionsStore is a structure containing a structure of option values for
        %   the solver. It also contains other meta-data that provides information
        %   about the state of the options, e.g. whether a given option has been
        %   set by the user.
        %
        %   This property must be defined and initialized by subclasses.
        OptionsStore
    end

    properties (Hidden, Abstract, GetAccess = public)
        %PROPERTYMETAINFO Constant, globally visible metadata about this class.
        %
        % This data is used to spec the options in this class for internal
        % clients such as: tab-complete, and the options validation
        %
        % This property must be defined and initialized by subclasses.
        PropertyMetaInfo
    end

    properties (Hidden, SetAccess = private, GetAccess = private)
        %PROBLEMDEFOPTIONS Internal options for problem-based approach
        ProblemdefOptions = optim.internal.constants.DefaultOptionValues.ProblemdefOptions;
    end

    properties(Hidden, SetAccess = protected, GetAccess = public)
        %FROMOPTIMPROBLEM Identifies if the options object was created by
        %the optimoptions method of an OptimizationProblem.
        %
        %   This property is set to true only by the optimoptions method of
        %   OptimizationProblem. It is used to determine how to update the
        %   options object when the problem simplifies after extracting the
        %   coefficients for solve.
        FromOptimProblem = false;
    end

    properties(Hidden, SetAccess = protected, GetAccess = public)
        %VERSION Keep track of the version for the objects
        %   Version is an integer to indicate the version number of the object.
        Version
    end

    methods (Hidden, Access = protected)

        function obj = setProperty(obj, name, value)
            %SETPROPERTY Set a value for a solver option
            %
            %   OBJ = SETPROPERTY(OBJ, NAME, VALUE) sets the value of the named
            %   property to the specified value or throws an error.
            %
            %   OBJ = SETPROPERTY(OBJ, NAME, VALUE, POSSVALUES) allows callers to
            %   specify a list of possible values when the option is a cell string.

            % Validate
            [value,validvalue,errid,errmsg] = obj.PropertyMetaInfo.(name).validate(name,value);

            if validvalue
                obj = setPropertyNoChecks(obj, name, value);
            else
                ME = MException(errid,errmsg);
                throwAsCaller(ME);
            end
        end

        function obj = setAliasProperty(obj, name, alias, value)
            %SETPROPERTY Set a value for a solver option with a fixed alias
            %
            %   OBJ = SETPROPERTY(OBJ, NAME, VALUE) sets the value of the named
            %   property to the specified value or throws an error.
            %
            %   OBJ = SETPROPERTY(OBJ, NAME, VALUE, POSSVALUES) allows callers to
            %   specify a list of possible values when the option is a cell string.

            % Validate
            [value,validvalue,errid,errmsg] = obj.PropertyMetaInfo.(name).validate(name,value);

            if validvalue
                obj = setPropertyNoChecks(obj, alias, value);
            else
                ME = MException(errid,errmsg);
                throwAsCaller(ME);
            end
        end

        function obj = setNewProperty(obj, name, value)
            %SETPROPERTY Set a value for a solver option
            %
            %   OBJ = SETPROPERTY(OBJ, NAME, VALUE) sets the value of the named
            %   property to the specified value or throws an error.
            %
            %   OBJ = SETPROPERTY(OBJ, NAME, VALUE, POSSVALUES) allows callers to
            %   specify a list of possible values when the option is a cell string.

            % Validate
            [value,validvalue,errid,errmsg] = obj.PropertyMetaInfo.(name).validate(name,value);

            if validvalue
                [name, value] = optim.options.OptionAliasStore.mapOptionToStore(...
                    name, value, obj.OptionsStore.Options);
                if ~iscell(name)
                    obj = setPropertyNoChecks(obj, name, value);
                else
                    for k = 1:numel(name)
                        obj = setPropertyNoChecks(obj, name{k}, value{k});
                    end
                end
            else
                ME = MException(errid,errmsg);
                throwAsCaller(ME);
            end
        end

        function obj = setPropertyNoChecks(obj, name, value)
            %SETPROPERTYNOCHECKS Set a value for a solver option without
            %                    error checking
            %
            %   OBJ = SETPROPERTYNOCHECKS(OBJ, NAME, VALUE) sets the value
            %   of the named property to the specified value with no error
            %   checking.
            %
            %   NOTE: This method is designed for internal values of a
            %   public option only, e.g. setting the Display option to
            %   'testing'. For publicly supported values, use the
            %   setProperty method.

            obj.OptionsStore.Options.(name) = value;
            obj.OptionsStore.SetByUser.(name) = true;

        end

        function obj = upgradeFrom13a(obj, s)
            %UPGRADEFROM13A Perform common tasks to load 13a options
            %
            %   OBJ = UPGRADEFROM13A(OBJ, S) copies the OptionsStore and
            %   <Solver>Version fields from the structure into the object.
            %
            %   NOTE: Objects saved in R2013a will come in as structures.
            %   This is because we removed properties from
            %   optim.options.SolverOptions in 13b but did not need to
            %   write a loadobj at that time, as objects loaded correctly.
            %   Once we have a loadobj, the object loading throws an
            %   exception.

            % We need to reset OptionsStore
            obj.OptionsStore = s.OptionsStore;

            % Set the version number back to allow loadobj to incrementally
            % upgrade the object correctly
            obj.Version = s.Version;

            % Note that s.SolverName is correctly set by the
            % constructor. The other three fields in s,
            % (StrongStartTag, StrongEndTag & EnableLinks) were removed
            % in R2013b.
        end

    end

    methods (Hidden)

        function obj = SolverOptions(varargin)
            %SOLVEROPTIONS Construct a new SolverOptions object
            %
            %   OPTS = OPTIM.OPTIONS.SOLVEROPTIONS creates a set of options with each
            %   option set to its default value.
            %
            %   OPTS = OPTIM.OPTIONS.SOLVEROPTIONS(PARAM, VAL, ...) creates a set of
            %   options with the named parameters altered with the specified values.
            %
            %   OPTS = OPTIM.OPTIONS.SOLVEROPTIONS(OLDOPTS, PARAM, VAL, ...) creates a
            %   copy of OLDOPTS with the named parameters altered with the specified
            %   value

            if nargin > 0

                % Deal with the case where the first input to the
                % constructor is a SolveOptions object.
                if isa(varargin{1},'optim.options.SolverOptions')
                    if strcmp(class(varargin{1}),class(obj))
                        obj = varargin{1};
                    else
                        % Get the properties from options object passed
                        % into the constructor.
                        thisProps = getOptionNames(varargin{1});
                        % Set the common properties. Note that we
                        % investigated first finding the properties that
                        % are common to both objects and just looping over
                        % those. We found that in most cases this was no
                        % quicker than just looping over the properties of
                        % the object passed in.
                        for i = 1:length(thisProps)
                            try %#ok

                                % An option can now be set by setting the
                                % actual underlying option or its alias.
                                % For example, an object being passed in
                                % may have a MaxIterations property. This
                                % will also have a MaxIter property. If the
                                % property has already been set (either by
                                % the alias or the underying property) we
                                % move to the next iteration of the for
                                % loop.
                                if obj.OptionsStore.SetByUser.(optim.options.OptionAliasStore.getAlias(thisProps{i}, obj.SolverName, obj.OptionsStore.Options))
                                    continue
                                end

                                % Store the value of option. Note that this
                                % is the default value because we haven't
                                % set this property yet.
                                defaultValue = obj.(thisProps{i});

                                % Try to set one of the properties of the
                                % old object in the new one.
                                obj.(thisProps{i}) = varargin{1}.(thisProps{i});

                                % If here, the property is common to both
                                % objects. We need to revert the SetByUser
                                % flag if the property has not been set and
                                % the default values of the property are
                                % equal.
                                %
                                % The following code is equivalent to this
                                % pseudo code:
                                % if isPropSetByUser
                                %      % If here, the property was set in
                                %      % the old object. In this case, as
                                %      % we have set the property in the
                                %      % new object there is nothing more
                                %      % to do here.
                                %  else
                                %      % If here, the property was not set
                                %      % in the old object. We now check
                                %      % whether the default value of the
                                %      % property in each object is
                                %      % identical.
                                %      if isDefaultEqual
                                %         % If the defaults are equal, then
                                %         % this property has not changed
                                %         % value. As such we do not want
                                %         % it to be marked as "SetByUser".
                                %      OptionsStore.SetByUser = false
                                %      else
                                %         % If the defaults are not equal
                                %         % we want to use the value from
                                %         % the old object. In this case,
                                %         % the property should be set and
                                %         % it should appear that it has
                                %         % been set. There is nothing more
                                %         % to do in this case.
                                %      end
                                % end
                                isPropSetByUserInOldObject = ...
                                    varargin{1}.OptionsStore.SetByUser.(thisProps{i});
                                if ~isPropSetByUserInOldObject && ...
                                        ( ischar(defaultValue) && strcmp(defaultValue, varargin{1}.(thisProps{i})) || ...
                                        isequal(defaultValue, varargin{1}.(thisProps{i})) )
                                    obj.OptionsStore.SetByUser.(thisProps{i}) = false;
                                end
                            end
                        end
                        % Copy over problemdef options
                        obj = setProblemdefOptions(obj, getProblemdefOptions(varargin{1}));
                    end
                    firstInputObj = true;
                else
                    firstInputObj = false;
                end

                % Extract the options that we want to set.
                if firstInputObj
                    pvPairs = varargin(2:end);
                else
                    pvPairs = varargin;
                end

                % If we have any string inputs, we need to convert to char.
                % We need to also account for any inputs that may contain
                % cell arrays of strings.
                isStringType = cellfun(@isstring, pvPairs);
                isScalarType = cellfun(@isscalar, pvPairs);
                isCellArray = cellfun(@iscell, pvPairs);

                % convert these to row vectors since we will be using them
                % with the find function.  The find function returns a 0x1
                % double if a column of all zeros is input -- causing a
                % single iteration in a for loop if we're not careful.
                isStringType = isStringType(:).';
                isScalarType = isScalarType(:).';
                isCellArray = isCellArray(:).';

                if any(isStringType)
                    % convert scalar strings to character array
                    stringScalarIdx = find(isStringType & isScalarType);
                    for i = stringScalarIdx
                        pvPairs{i} = char(pvPairs{i});
                    end

                    % convert string vectors to cell array of char arrays
                    stringVectorIdx = find(isStringType & ~isScalarType);
                    for i = stringVectorIdx
                        strArray = pvPairs{i};
                        pvPairs{i} = cellstr(strArray(:)');
                    end
                end

                if any(isCellArray)
                   % loop through all cell array inputs and convert any
                   % string datatypes to character arrays
                   cellArrayIdx = find(isCellArray);
                    for i = cellArrayIdx
                        myCell = pvPairs{i};
                        isStringTypeInCell = cellfun(@isstring, myCell);
                        isScalarTypeInCell = cellfun(@isscalar, myCell);

                        stringAndScalar = isStringTypeInCell & isScalarTypeInCell;
                        stringAndArray = isStringTypeInCell & ~isScalarTypeInCell;

                        % convert to row when using inside of find or for
                        % loop can trigger when traversing all zero column
                        % vector
                        stringAndScalar = stringAndScalar(:).';
                        stringAndArray = stringAndArray(:).';

                        % convert scalar strings to character arrays
                        myCell(stringAndScalar) = cellstr(myCell(stringAndScalar));

                        % convert non-scalar strings to cell arrays of
                        % character vectors
                        strArrayIdx = find(stringAndArray);
                        for j = strArrayIdx
                            myCell{j} = cellstr(myCell{j});
                        end

                        pvPairs{i} = myCell;
                    end
                end

                % Loop through each param-value pair and just try to set
                % the option. When the option has been fully specified with
                % the correct case, this is fast. The catch clause deals
                % with partial matches or errors.
                haveCreatedInputParser = false;
                for i = 1:2:length(pvPairs)
                    try
                        obj.(pvPairs{i}) = pvPairs{i+1};
                    catch ME %#ok

                        % Create the input parser if we haven't already. We
                        % do it here to avoid creating it if possible, as
                        % it is slow to set up.
                        if ~haveCreatedInputParser
                            ip = inputParser;
                            % Structures are currently not supported as
                            % an input to optimoptions. Setting the
                            % StructExpand property of the input parser to
                            % false, forces the parser to treat the
                            % structure as a single input and not a set of
                            % param-value pairs.
                            ip.StructExpand =  false;
                            % Get list of option names
                            allOptionNames = getOptionNames(obj);
                            for j = 1:length(allOptionNames)
                                % Just specify an empty default as we already have the
                                % defaults in the options object.
                                ip.addParameter(allOptionNames{j}, []);
                            end
                            haveCreatedInputParser = true;
                        end

                        % Get the p-v pair to parse.
                        thisPair = pvPairs(i:min(i+1, length(pvPairs)));
                        ip.parse(thisPair{:});

                        % Determine the option that was specified in p-v pairs.
                        % These options will now be matched even if only partially
                        % specified (by 13a). Now set the specified value in the
                        % options object.
                        optionSet = setdiff(allOptionNames, ip.UsingDefaults);
                        obj.(optionSet{1}) = ip.Results.(optionSet{1});
                    end
                end
            end
        end

    end

    % Helper methods required for display
    methods (Hidden, Abstract, Access = protected)
        needExtraHeader(obj)
        needExtraFooter(obj)
        getDisplayOptions(obj)
    end

    % Display methods
    methods (Hidden, Access = protected)

        function header = addExtraHeader(obj, header)
            %ADDEXTRAHEADER Append text to the header
            %
            %   HEADER = ADDEXTRAHEADER(OBJ, HEADER) appends the "Set by user" section
            %   of the display to the header if no options are set by user.
            %
            %   There is currently no way of specifying a string to display when no
            %   options are set by the user. In this case, we have to just specify one
            %   property group and fold the Set by user group into the header.

            if needExtraHeader(obj)
                allDefaultStr = getString(message('MATLAB:optimfun:options:SolverOptions:AllDefaultStr'));
                header = sprintf('%s\n   %s\n     %s\n', ...
                    header, ...
                    getString(message('MATLAB:optimfun:options:SolverOptions:SetByUserHeader', ...
                    optim.options.SolverOptions.generateStrongStartTag, ...
                    optim.options.SolverOptions.generateStrongEndTag)), ...
                    allDefaultStr);
            end

        end

        function header = getHeader(obj)
            %GETHEADER Return the header for the display
            %
            %   HEADER = GETHEADER(OBJ) returns the header for the display. This method
            %   must be implemented as this class inherits from
            %   matlab.mixin.CustomDisplay.

            if isscalar(obj)
                if optim.options.SolverOptions.enableLinks
                    solverLink = sprintf('<a href="matlab: helpPopup %s" style="font-weight:bold">%s</a>', obj.SolverName, obj.SolverName);
                else
                    solverLink = obj.SolverName;
                end
                header = sprintf('  %s\n', ...
                    getString(message('MATLAB:optimfun:options:SolverOptions:HeaderStr', solverLink)));

                % Add extra header
                header = addExtraHeader(obj, header);
            else
                header = getHeader@matlab.mixin.CustomDisplay(obj);
            end
        end

        function footer = addExtraFooter(obj, footer)
            %ADDEXTRAFOOTER Add text to the footer
            %
            %   HEADER = ADDEXTRAFOOTER(OBJ, HEADER) prepends the "Default" section
            %   of the display to the footer if all options are set by user.
            %
            %   There is currently no way of specifying a string to display when no
            %   options are set by the user. In this case, we have to just specify one
            %   property group and fold the Set by user group into the header.

            if needExtraFooter(obj)
                allSetByUserStr = getString(message('MATLAB:optimfun:options:SolverOptions:AllSetByUserStr'));
                footer = sprintf('%s   %s\n     %s\n', ...
                    footer, ...
                    getString(message('MATLAB:optimfun:options:SolverOptions:DefaultHeader', ...
                    optim.options.SolverOptions.generateStrongStartTag, ....
                    optim.options.SolverOptions.generateStrongEndTag)), ...
                    allSetByUserStr);
            end

        end

        function footer = getFooter(obj)
            %GETFOOTER Return the footer for the display
            %
            %   FOOTER = GETFOOTER(OBJ) returns the footer for the display. This method
            %   must be implemented as this class inherits from
            %   matlab.mixin.CustomDisplay.

            if isscalar(obj)
                % Call superclass getFooter
                footer = getFooter@matlab.mixin.CustomDisplay(obj);

                % Add extra footer
                footer = addExtraFooter(obj, footer);
            else
                footer = getFooter@matlab.mixin.CustomDisplay(obj);
            end
        end

        function groups = getPropertyGroups(obj)
            %GETPROPERTYGROUPS Return the property groups for the display
            %
            %   GROUPS = GETPROPERTYGROUPS(OBJ) returns the property groups for the
            %   display. If all the options are at the default value, one group is
            %   returned containing all the options with the title "Default". If all
            %   the options have been set by the user, one group is returned with the
            %   title "Set by User". Otherwise two property groups are returned, one
            %   containing the options set by the user and another containing the
            %   remaining properties at their default values.
            %
            %   This method must be implemented as this class inherits from
            %   matlab.mixin.CustomDisplay.

            if isscalar(obj)

                % Get names of all properties to be displayed
                allOptions = getDisplayOptions(obj);

                % Sort the options to make them display in alphabetical
                % order.
                [~, idx] = sort(lower(allOptions));
                allOptions = allOptions(idx);

                % Create the group cell arrays
                if needExtraHeader(obj)
                    % Create the mixin property groups
                    defaultHeader = getString(message('MATLAB:optimfun:options:SolverOptions:DefaultHeader', ...
                        optim.options.SolverOptions.generateStrongStartTag, ...
                        optim.options.SolverOptions.generateStrongEndTag));
                    groups = matlab.mixin.util.PropertyGroup(allOptions, defaultHeader);
                elseif needExtraFooter(obj)
                    % Create the mixin property groups
                    setByUserHeader = getString(message('MATLAB:optimfun:options:SolverOptions:SetByUserHeader', ...
                        optim.options.SolverOptions.generateStrongStartTag, ...
                        optim.options.SolverOptions.generateStrongEndTag));
                    groups = matlab.mixin.util.PropertyGroup(allOptions, setByUserHeader);
                else
                    % Split the options into a group that have been set by
                    % user and a group taking the default value
                    idxSetByUser = false(1, length(allOptions));
                    for i = 1:length(allOptions)
                        idxSetByUser(i) = obj.OptionsStore.SetByUser.(optim.options.OptionAliasStore.getAlias(allOptions{i}, obj.SolverName, obj.OptionsStore.Options));
                    end
                    setByUserGroup = allOptions(idxSetByUser);
                    defaultGroup = allOptions(~idxSetByUser);

                    % Create the mixin property groups
                    setByUserHeader = getString(message('MATLAB:optimfun:options:SolverOptions:SetByUserHeader', ...
                        optim.options.SolverOptions.generateStrongStartTag, ...
                        optim.options.SolverOptions.generateStrongEndTag));
                    groups = matlab.mixin.util.PropertyGroup(setByUserGroup, setByUserHeader);
                    defaultHeader = getString(message('MATLAB:optimfun:options:SolverOptions:DefaultHeader', ...
                        optim.options.SolverOptions.generateStrongStartTag, ...
                        optim.options.SolverOptions.generateStrongEndTag));
                    groups(2) = matlab.mixin.util.PropertyGroup(defaultGroup, defaultHeader);
                end
            else
                groups = getPropertyGroups@matlab.mixin.CustomDisplay(obj);
            end

        end

    end

    methods (Hidden)

        function OptionsStruct = extractOptionsStructure(obj)
            %EXTRACTOPTIONSSTRUCTURE Extract options structure from
            %OptionsStore
            %
            %   OPTIONSSTRUCT = EXTRACTOPTIONSSTRUCTURE(OBJ) extracts a
            %   plain structure containing the options from
            %   obj.OptionsStore, the ProblemdefOptions, and any additional
            %   custom options needed by the caller. The solver calls
            %   convertForSolver, which in turn calls this method to obtain
            %   a plain options structure.

            % Call helper method to generate the structure with options for
            % the user of this method (with custom fields if needed).
            OptionsStruct = extractCustomOptionsStructure(obj);

            % Append the ProblemdefOptions field
            OptionsStruct.ProblemdefOptions = obj.ProblemdefOptions;

        end

        function obj = setProblemdefOptions(obj, ProblemdefOptions)

            %SETPROBLEMDEFOPTIONS Set problemdef options
            %
            %   OBJ = SETPROBLEMDEFOPTIONS(OBJ, PROBLEMDEFOPTIONS) sets the
            %   specified problemdef options in OBJ

            obj.ProblemdefOptions = ProblemdefOptions;

        end

        function ProblemdefOptions = getProblemdefOptions(obj)

            %GETPROBLEMDEFOPTIONS Get problemdef options
            %
            %   PROBLEMDEFOPTIONS = GETPROBLEMDEFOPTIONS(OBJ) gets the
            %   problemdef options in OBJ

            ProblemdefOptions = obj.ProblemdefOptions;

        end

    end   % Hidden methods

    % Required hidden helper methods
    methods (Abstract, Hidden)
        % Helper method to generate the structure with options needed by
        % the user of this method (Subclasses can add custom fields if
        % needed).
        extractCustomOptionsStructure(obj)
    end

    % Required public interface methods
    methods (Abstract)
        obj = resetoptions(obj, name)
    end

    % Protected validator for resetoptions
    methods (Access = protected)

        function name = validateResetOptions(obj, name)

            % Attempt to convert name into a cellstr, which checks that the
            % name is either a char, cellstr, cell of char/strings or a
            % string. cellstr throws an error for any other type.
            try
                % Handle non-cell inputs. Remove whitespace.
                name = strip(cellstr(name));
            catch
                % Throw a customized error so the user doesn't think it
                % has come from some internal function call.
                throwAsCaller(MException(message('MATLAB:optimfun:options:SolverOptions:InvalidOptionType')));
            end

            % Validate the list of options to be reset. After this
            % for-loop, the cell array, name, will contain exact matches of
            % the option names in obj.
            validOptionNames = getOptionNames(obj);
            for i = 1:numel(name)
                try
                    name{i} = validatestring(name{i}, validOptionNames);
                catch ME
                    switch ME.identifier
                        case 'MATLAB:unrecognizedStringChoice'
                            [linkToSolverCshStart, linkToSolverCshEnd] = ...
                                optim.options.createLinkToSolverCsh(obj.SolverName);
                            throwAsCaller(MException(message('MATLAB:optimfun:options:SolverOptions:UnmatchedOption', ...
                                name{i}, upper(obj.SolverName), upper(obj.SolverName), ...
                                linkToSolverCshStart, linkToSolverCshEnd)));
                        case 'MATLAB:ambiguousStringChoice'
                            throwAsCaller(MException(message('MATLAB:optimfun:options:SolverOptions:AmbiguousOption', name{i})));
                        otherwise
                            throw(ME);
                    end
                end
            end

        end

    end

    % Required method for tab completion of property assignments. Note that
    % this uses different infrastructure to the JSON-based tab completion
    % of function signatures. As such, we need to support the set method
    % here.
    methods (Hidden)
        function validCompletions = set(opts, optionName)
            try
                propTypeStruct = opts.PropertyMetaInfo.(optionName);
                % The tab completion infrastructure for property
                % assignments only supports enumerations for now.
                switch propTypeStruct(1).TabType
                    case 'enum'
                        validCompletions = propTypeStruct.Values;
                    case 'fcnenum'
                        validCompletions = propTypeStruct(1).Values;
                    otherwise
                        validCompletions = {};
                end
            catch
                validCompletions = {};
            end
        end
    end

    % Hidden helper methods
    methods (Hidden)
        function setByUserOptions = getOptionsSetByUser(obj)
            %GETOPTIONSSETBYUSER Return the options set by the user
            %
            %   SETBYUSEROPTIONS = GETOPTIONSSETBYUSER(OBJ) returns a cell string of
            %   option names that have been set by the user.

            setByUserOptions = getSetByUserOptionNames(obj);

            % The fields in the SetByUser structures use the old names -
            % map to the new names
            for j = 1:length(setByUserOptions)
                thisOption = optim.options.OptionAliasStore.getNameFromAlias(...
                    setByUserOptions{j});
                setByUserOptions{j} = thisOption{1};
            end
        end

        function optionsStruct = getSetByUserOptionsStruct(obj)
            %GETOPTIONSSETBYUSER Return the options set by the user in a
            %struct, with the remaining options empty.
            %
            %   SETBYUSEROPTIONS = GETOPTIONSSETBYUSER(OBJ) returns the
            %   struct provided with that have been set by the user.

            % Get the names of the options set by the user, as well as the
            % names of all options
            [setByUserOptions,allOptions] = getSetByUserOptionNames(obj);

            % Make "copy" struct with all fields empty
            empties = cell(size(allOptions));
            [empties{:}] = deal([]);
            optionsStruct = cell2struct(empties,allOptions,1);

            % Overwrite the empties for those options set by user
            for i = 1: length(setByUserOptions)
                optionsStruct.(setByUserOptions{i}) = obj.OptionsStore.Options.(setByUserOptions{i});
            end
        end

        function [setByUserOptions,allOptions] = getSetByUserOptionNames(obj)
            %GETOPTIONSSETBYUSER Return the options set by the user
            %
            %   SETBYUSEROPTIONS = GETOPTIONSSETBYUSER(OBJ) returns a cell string of
            %   option names that have been set by the user.

            allOptions = fieldnames(obj.OptionsStore.SetByUser);
            setByUser = struct2cell(obj.OptionsStore.SetByUser);
            setByUserOptions = allOptions([setByUser{:}]);
        end

        function obj = convertForSolver(obj, Solver)
            %CONVERTFORSOLVER Convert optimoptions for the named solver
            %
            %   OPTIONS = CONVERTFORSOLVER(OBJ, SOLVER) converts the supplied options
            %   to a set of options for the named solver.

            % It is possible for a user to pass in a vector of options to the
            % solver. Silently use the first element in this array.
            if ~isscalar(obj)
                obj = obj(1);
            end

            % Ensure Solver string is a proper name
            Solver = char(Solver);
            Solver(1) = upper(Solver(1));

            % Create class name string. @todo: When the name of all options
            % classes end in "Options" replace code with line in "if"
            % clause
            objStr = "optim.options." + Solver;
            if any(strcmp(Solver, {'Ga', 'Patternsearch', 'Simulannealbnd', 'Gamultiobj', 'Coneprog'}))
                objStr = objStr + "Options";
            end

            % Issue warning if user has passed an options object of the
            % wrong type.
            if ~isa(obj, objStr)
                 % Create warning string with link
                convertOptsText = addLink( ...
                    getString(message('MATLAB:optimfun:options:SolverOptions:ConvertOptions', ...
                    upper(obj.SolverName))),'optim','convert_options',false);

                warning(message('MATLAB:optimfun:options:SolverOptions:WrongSolverOptions', ...
                                upper(obj.SolverName), upper(Solver), ...
                                upper(Solver), upper(obj.SolverName), ...
                                convertOptsText) );
                % Call the factory function to convert the solver object
                obj = optim.options.createSolverOptions(Solver, obj);
            end

        end

        function OptionsStruct = mapOptionsForSolver(~, OptionsStruct)
            %MAPOPTIONSFORSOLVER Map options for use by the solver
            %
            %   OptionsStruct = MAPOPTIONSFORSOLVER(obj, OptionsStruct)
            %   maps the specified structure so it can be used in the
            %   solver functions and in OPTIMTOOL.
            %
            %   This method does not alter the supplied structure.
            %   Subclasses can optionally supply mappings.

        end

        function [obj, OptimsetStruct] = mapOptimsetToOptions(obj, OptimsetStruct)
            %MAPOPTIMSETTOOPTIONS Map optimset structure to optimoptions
            %
            %   OBJ = MAPOPTIMSETTOOPTIONS(OBJ, OPTIMSETSTRUCT) maps specified optimset
            %   options, OptimsetStruct, to the equivalent options in the specified
            %   optimization object, obj.
            %
            %   [OBJ, OPTIONSSTRUCT] = MAPOPTIMSETTOOPTIONS(OBJ, OPTIMSETSTRUCT)
            %   additionally returns an options structure modified with any conversions
            %   that were performed on the options object.
            %
            %   This method does not alter the supplied options object or optimset
            %   structure. Subclasses can optionally supply mappings.

        end

        function OptionsStore = getOptionsStore(obj)
            %GETOPTIONSSTORE Return the OptionsStore
            %
            %   OPTIONSSTORE = GETOPTIONSSTORE(OBJ) returns the OptionsStore.

            OptionsStore = obj.OptionsStore;
        end

        function isSet = isSetByUser(obj, optionName)
            %ISSETBYUSER Return whether an option is set set by the user
            %
            %   ISSET = ISSETBYUSER(OBJ, OPTIONNAME) returns whether the specified
            %   option has been set by the user.

            isSet = obj.OptionsStore.SetByUser.(optionName);
        end

        function obj = replaceSpecialStrings(obj)
            %replaceSpecialStrings Replace special string values
            %
            %   obj = replaceSpecialStrings(obj) replaces special string
            %   option values with their equivalent numerical value. We
            %   currently only use this method to convert FinDiffRelStep.
            %   However, in the future we would like to move the special
            %   string replacement code from the solver files to the
            %   options classes.
            %
            %   This method does not replace any special strings.
            %   Subclasses can optionally replace special strings.

        end

        function [OptionNames, idxHiddenOptions] = getOptionNames(obj)
            %GETOPTIONNAMES Get the options a user can set/get
            %
            %   OPTIONNAMES = GETOPTIONNAMES(OBJ) returns a list of the
            %   options that a user can set/get. This list is the union of
            %   the public properties and those that are hidden, dependent
            %   and have public set/get access.

            % All public properties are options that can be set/get by the
            % user.
            OptionNames = properties(obj);

            % Find any hidden public options. These options will not be
            % returned from the call to properties. These options can still
            % be set/get by users.
            mc = metaclass(obj);
            idxDep = arrayfun(@i_checkForHiddenPublic, mc.PropertyList);
            OptionsOnDeprecationPath = cell(1, sum(idxDep));
            [OptionsOnDeprecationPath{:}] = deal(mc.PropertyList(idxDep).Name);

            % Return indices of hidden options
            idxHiddenOptions = false(numel(OptionNames), 1);
            idxHiddenOptions = [idxHiddenOptions; true(numel(OptionsOnDeprecationPath), 1)];

            % Return public properties and hidden public names
            OptionNames = [OptionNames; OptionsOnDeprecationPath'];

            function isHiddenPublic = i_checkForHiddenPublic(thisProperty)

                isDep = thisProperty.Dependent;
                isHidden = thisProperty.Hidden;
                isSetPublic = strcmp(thisProperty.SetAccess, 'public');
                isGetPublic = strcmp(thisProperty.GetAccess, 'public');

                isHiddenPublic = isDep && isHidden && isSetPublic && isGetPublic;

            end
        end

        function obj = copyForOutputAndPlotFcn(obj, otherStruct)
            % Find options common to both the input structure and the
            % underlying options store
            optionsToCopy = intersect(fieldnames(obj.OptionsStore.Options), ...
                                      fieldnames(otherStruct));

            % Copy the options individually
            for k = 1:numel(optionsToCopy)
               obj.OptionsStore.Options.(optionsToCopy{k}) = otherStruct.(optionsToCopy{k});
            end

            if isfield(otherStruct,'TolFun')
                obj.OptionsStore.Options.TolFunValue = otherStruct.TolFun;
            end
        end
    end

    methods (Access=protected, Static)

        function StrongStartTag = generateStrongStartTag
        %GENERATESTRONGSTARTTAG Start tag for using bold face
        %
        %   STRONGSTARTTAG = GENERATESTRONGSTARTTAG returns a string
        %   holding bold face mark up ('<strong>') or no mark up ('')
        %   depending on whether we can use bold face mark up at the
        %   command prompt.

            if optim.options.SolverOptions.enableLinks
                StrongStartTag = '<strong>';
            else
                StrongStartTag = '';
            end

        end

        function StrongEndTag = generateStrongEndTag
        %GENERATESTRONGENDTAG End tag for using bold face
        %
        %   STRONGENDTAG = GENERATESTRONGENDTAG returns a string holding
        %   the end of the bold face mark up ('</strong>') or no mark up
        %   ('') depending on whether we can use bold face mark up at the
        %   command prompt.
            if optim.options.SolverOptions.enableLinks
                StrongEndTag = '</strong>';
            else
                StrongEndTag = '';
            end
        end

        function AreLinksEnabled = enableLinks
        %ENABLELINKS Indicate whether hyperlinks are enabled
        %
        %   ARELINKSENABLED = ENABLELINKS returns whether the customized
        %   display contains hyperlinks. EnableLinks holds a boolean flag
        %   indicating whether hyperlinks can be enabled or not.

        AreLinksEnabled = feature('hotlinks') && ~isdeployed;

        end


    end

    methods (Hidden, Access = ?optim.internal.problemdef.ProblemImpl)
        function obj = setFromOptimProblem(obj)
            %SETFROMOPTIMPROBLEM sets the FromOptimProblem option to true
            %
            %   SETFROMOPTIMPROBLEM is the method used by the optimoptions
            %   method of ProblemImpl to indicate that the options object
            %   was created by the problem-based approach and not by
            %   specifying a solver.
            obj.FromOptimProblem = true;
        end

    end

end
