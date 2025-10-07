% Load libraries
clear;
clc;

% Open file selection dialog
[filename, pathname] = uigetfile('*.csv', 'Select the AirQuality.csv file');
if isequal(filename, 0)
    disp('User selected Cancel');
else
    % Load data with preserved variable names
    data = readtable(fullfile(pathname, filename), 'VariableNamingRule', 'preserve');
    variables = data.Properties.VariableNames;

    % Parameters
    alpha = 0.05;

    % Run the PC algorithm
    G = pc_algorithm(data, alpha);

    % Plotting the result
    figure;
    G = graph(G);
    plot(G, 'Layout', 'circle', 'NodeColor', 'lightgreen', 'MarkerSize', 10, 'NodeFontSize', 12, 'NodeFontWeight', 'bold');
    title('Simulation Result');
    saveas(gcf, 'result.png');
end

% Define conditional independence test function
function isIndependent = conditional_independence_test(data, X, Y, S, alpha)
    if isempty(S)
        corrMatrix = corrcoef(data{:, {X, Y}});
        if size(corrMatrix, 1) < 2
            isIndependent = true;
            return;
        end
        corr = corrMatrix(1, 2);
    else
        X_res = regress(data{:, X}, data{:, S});
        Y_res = regress(data{:, Y}, data{:, S});
        corrMatrix = corrcoef(X_res, Y_res);
        if size(corrMatrix, 1) < 2
            isIndependent = true;
            return;
        end
        corr = corrMatrix(1, 2);
    end
    isIndependent = abs(corr) < alpha;  % Update to check the absolute value of correlation
end

% Build the skeleton of the causal graph
function [graph, sep_set] = build_skeleton(data, alpha)
    variables = data.Properties.VariableNames;
    n = numel(variables);
    graph = ones(n); % Full graph
    sep_set = cell(n);

    for l = 0:n-1
        for i = 1:n
            for j = i+1:n
                if graph(i, j)
                    for S = nchoosek(variables(setdiff(1:n, [i j])), l)
                        S = S{:};
                        if conditional_independence_test(data, variables{i}, variables{j}, S, alpha)
                            graph(i, j) = 0;
                            graph(j, i) = 0;
                            sep_set{i, j} = S;
                            break;
                        end
                    end
                end
            end
        end
    end
end

% Orient edges
function directed_graph = orient_edges(graph, sep_set)
    n = size(graph, 1);
    directed_graph = graph;

    for i = 1:n
        for j = i+1:n
            if directed_graph(i, j) && directed_graph(j, i)
                for k = 1:n
                    if k ~= i && k ~= j && directed_graph(i, k) && directed_graph(k, j)
                        if isempty(sep_set{i, j}) || ~ismember(k, sep_set{i, j})
                            directed_graph(j, i) = 0;
                        end
                    end
                end
            end
        end
    end
end

% PC Algorithm
function directed_graph = pc_algorithm(data, alpha)
    [skeleton, sep_set] = build_skeleton(data, alpha);
    directed_graph = orient_edges(skeleton, sep_set);
end
