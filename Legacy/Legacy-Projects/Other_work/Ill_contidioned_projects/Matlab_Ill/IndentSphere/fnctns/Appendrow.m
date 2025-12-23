function [outputArg1,outputArg2] = Appendrow(filename,row)

    if isfile(filename)
        % existingData = readmatrix(filename);
        existingData = csvread(filename);
    else
        existingData = []; % Initialize as needed
    end

    updatedData = [existingData; row];
    % writematrix(updatedData, filename);
    csvwrite(filename,updatedData);
end

