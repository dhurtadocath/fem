clear
close all
clc


%% Add important paths
addpath('../fnctns/')

%% Load all the data in the current folder
vars = {'X', 'sph', 'conn' 'nodes_base' 'slaves'};
% load('saved_vars', vars{:});         % change so that it opens only the necessary vars
load('saved_vars');         % change so that it opens only the necessary vars
sph{1}{3} = sph{1}{3}+0.2;

ACTIVES = readmatrix('actives.csv');
DISPLACEMENTS = readmatrix('displacements.csv');
FORCES = readmatrix('forces.csv');
n_times = size(ACTIVES,1);
DOFS = {dofs_base_x,dofs_base_y,dofs_base_z};

STRAINS = get_deformations(DISPLACEMENTS,X,conn);


load('vars_fless.mat');
Fless = [Fx,Fy,Fz];

plot4d(ACTIVES,DISPLACEMENTS,FORCES,DOFS,STRAINS,slaves,sph,X,conn,n_times,Fless);



%% Functions
function plot4d(ACTIVES,DISPLACEMENTS,FORCES,DOFS,STRAINS,slaves,sph,X,conn,n_times,Fless)
    % Create a figure
    f = figure('Name', 'Interactive Figure with Scrollbar');

    dofs_base_x = DOFS{1};
    dofs_base_y = DOFS{2};
    dofs_base_z = DOFS{3};
    
    subplot1Position = [0.05, 0.25, 0.45, 0.7]; % [left, bottom, width, height]
    subplot2Position = [0.55, 0.3, 0.35, 0.6]; % Adjust as necessary

    % Storing the plot handles for the 2D plots
    ax2 = subplot(1,2,2,'Parent',f,'Position',subplot2Position);
    hold on;

    Fx = Fless(:,1);
    Fy = Fless(:,2);
    Fz = Fless(:,3);
    plotHandles = zeros(1, 3); % Array to store plot handles
    plotHandles(1) = plot(1:n_times, sum(abs(FORCES(:,dofs_base_x)), 2), 'b','LineWidth',1.5);
    plotHandles(2) = plot(1:n_times, sum(abs(FORCES(:,dofs_base_y)), 2), 'g','LineWidth',1.5);
    plotHandles(3) = plot(1:n_times, sum(abs(FORCES(:,dofs_base_z)), 2), 'r','LineWidth',1.5);

    plotHandles(4) = plot(1:n_times, Fx, '-.b');
    plotHandles(5) = plot(1:n_times, Fy, '-.g');
    plotHandles(6) = plot(1:n_times, Fz, '-.r');
    legend('\Sigma|Fx| (\mu=0.5)', ...
        '\Sigma|Fy| (\mu=0.5)', ...
        '\Sigma|Fz| (\mu=0.5)', ...
        '\Sigma|Fx| (\mu=0.0)', ...
        '\Sigma|Fy| (\mu=0.0)', ...
        '\Sigma|Fz| (\mu=0.0)', ...
        'Location','northwest');
    
    % Store handles for current time points (initially empty)
    currentTimePointHandles = gobjects(1, 3); % MATLAB Graphics Objects array

    ax = subplot(1,2,1,'Parent',f,'Position',subplot1Position);


    % Define the initial parameter value
    time = 1;

    % Create axes for plotting
    hold(ax, 'on'); % Hold on to the current axes
    axis equal
    view(37.5,30.0)


    scrollbarPosition = [0.1, 0.1, 0.8, 0.03]; % Lower the scrollbar
    labelYPosition = 0.08; % Y position for labels

    % Create a scrollbar
    scrollbar = uicontrol('Parent', f, 'Style', 'slider', ...
                          'Units', 'normalized', ...
                          'Position', scrollbarPosition, ...
                          'min', 1, 'max', n_times, 'Value', time);


    scrollbar.SliderStep = [1/100, 1/10];

    % Add text labels for min, max, and current values
    uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.05, labelYPosition, 0.05, 0.05], 'String', num2str(1));
    uicontrol('Style', 'text', 'Units', 'normalized', 'Position', [0.9, labelYPosition, 0.05, 0.05], 'String', num2str(n_times));
    currentValueLabel = uicontrol('Style', 'text', 'Units', 'normalized', 'Position',  [0.45, labelYPosition + 0.05, 0.1, 0.05], 'String', num2str(time));


    % Set callback for the scrollbar
    % scrollbar.Callback = @(src, event) redraw_figure(src, ax, ax2, plotHandles,currentTimePointHandles, DISPLACEMENTS,ACTIVES,slaves,sph,X,conn,n_times,currentValueLabel);
    scrollbar.Callback = @(src, event) callbackFunction(src);

    % Initial drawing
    currentTimePointHandles = redraw_figure(scrollbar, ax, ax2, plotHandles, currentTimePointHandles, DISPLACEMENTS, ACTIVES, slaves, sph, X, conn, n_times, currentValueLabel,STRAINS);


    function callbackFunction(src)
        % Update currentTimePointHandles with each call
        currentTimePointHandles = redraw_figure(src, ax, ax2, plotHandles, currentTimePointHandles, DISPLACEMENTS, ACTIVES, slaves, sph, X, conn, n_times, currentValueLabel,STRAINS);
    end


end

function currentTimePointHandles = redraw_figure(scrollbar, ax, ax2, plotHandles,currentTimePointHandles,DISPLACEMENTS,ACTIVES,slaves,sph,X,conn,n_times,currentValueLabel,STRAINS)

    time = round(get(scrollbar, 'Value'));

    cla(ax);

    % Sphere's position
    sph{1}{3} = sph{1}{3}+(-0.2)*(time)/n_times;
    set(currentValueLabel, 'String', num2str(time));

    delete(currentTimePointHandles(currentTimePointHandles~=0));

    % Update the 2D plot with current time points
    colors = ['b','g','r'];
    for i = 1:length(plotHandles(1:3))
        % Add a thick point on each curve
        xData = get(plotHandles(i), 'XData');
        yData = get(plotHandles(i), 'YData');
        currentTimePointHandles(i) = plot(ax2, xData(time), yData(time), 'ko', 'MarkerSize', 5, 'MarkerFaceColor', colors(i));
    end

    % legend(ax2, plotHandles, {'\SigmaFx','\SigmaFy','\SigmaFz'});
    legend(ax2, plotHandles, {'\Sigma|Fx| (\mu=0.5)', ...
        '\Sigma|Fy| (\mu=0.5)', ...
        '\Sigma|Fz| (\mu=0.5)', ...
        '\Sigma|Fx| (\mu=0.0)', ...
        '\Sigma|Fy| (\mu=0.0)', ...
        '\Sigma|Fz| (\mu=0.0)'});
% legend('\Sigma|Fx|','\Sigma|Fy|','\Sigma|Fz|', ...
%         '\Sigma|Fx| (frictionless)', ...
%         '\Sigma|Fy| (frictionless)', ...
%         '\Sigma|Fz| (frictionless)', ...
%         'Location','northwest');

    actives = slaves(find(ACTIVES(time,:)));
    if isempty(actives)
        actives=[];
    end
    ax = plotTruss_XY_symm(DISPLACEMENTS(time,:)',X,conn,actives,ax,STRAINS,time);
    ax = plotsphere(sph,ax);
    % xlim([-0.25 1.25])
    % ylim([-0.25 1.25])
    zlim([-2 4])

    % Update the figure
    drawnow;
    return;
end