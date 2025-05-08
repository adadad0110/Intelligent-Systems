function ea1()
    clc; clear; close all;

    %% Configuration
    cfg.fitnessMode      = 'linear';    % 'power_law' or 'linear'
    cfg.reproductionMode = 'RWS';       % 'RWS' or 'TS'
    cfg.crossoverMode    = 'convex';    % 'average' or 'convex'
    cfg.populationSize   = 10;
    cfg.maxGenerations   = 200;
    cfg.valueRange       = [-10, 10];

    %% Run EA
    [bestX, fitnessHistory, genCount] = runEA(cfg);

    %% Plot Results
    figure;
    plot(1:genCount, fitnessHistory, '-');
    xlabel('Generation');
    ylabel('Max Fitness');
    grid on;
    if ~isempty(bestX)
        title(sprintf('EA (%s, %s, %s) GEN:%d, x=%.4f', ...
            cfg.fitnessMode, cfg.reproductionMode, cfg.crossoverMode, genCount, bestX), ...
            'Interpreter','none');
    else
        title(sprintf('EA (%s, %s, %s) Reached Max Generations (%d)', ...
            cfg.fitnessMode, cfg.reproductionMode, cfg.crossoverMode, cfg.maxGenerations), ...
            'Interpreter','none');
    end
end

%% Loop
function [bestX, maxFitHistory, generation] = runEA(cfg)
    pop = (cfg.valueRange(2)-cfg.valueRange(1)).*rand(1, cfg.populationSize) + cfg.valueRange(1);
    maxFitHistory = zeros(1, cfg.maxGenerations);
    bestX = [];

    xGrid = -10:0.1:10;
    yGrid = objective(xGrid);
    trueMax = max(yGrid);
    termFit = adaptFitness(trueMax, cfg);

    for generation = 1:cfg.maxGenerations
        rawFit   = objective(pop);
        fitness  = adaptFitness(rawFit, cfg);
        maxFitHistory(generation) = max(fitness);

        if maxFitHistory(generation) >= termFit
            idx   = find(fitness == max(fitness), 1);
            bestX = pop(idx);
            break;
        end

        parents = selectParentsReal(pop, fitness, cfg);
        offspring = doCrossoverReal(parents, cfg);
        pop = doMutationReal(offspring, cfg);
    end

    maxFitHistory = maxFitHistory(1:generation);
end

%% Objective Function
function y = objective(x)
    y = -15*(sin(2*x)).^2 - (x-2).^2 + 160;
end

%% Fitness Adaptation
function fAdj = adaptFitness(fRaw, cfg)
    switch cfg.fitnessMode
        case 'power_law'
            fAdj = fRaw .^ 5;
        case 'linear'
            fAdj = 1e5 * fRaw + 5;
        otherwise
            error('Unknown fitness mode.');
    end
end

%% Selection (RWS or Tournament)
function parents = selectParentsReal(pop, fitness, cfg)
    N = cfg.populationSize;
    parents = zeros(size(pop));
    switch cfg.reproductionMode
        case 'RWS'
            cumFit = cumsum(fitness);
            total  = cumFit(end);
            r      = rand(1, N) * total;
            for i = 1:N
                idx = find(cumFit >= r(i), 1);
                parents(i) = pop(idx);
            end
        case 'TS'
            for i = 1:N
                cand = randperm(N, 2);
                [~, b] = max(fitness(cand));
                parents(i) = pop(cand(b));
            end
        otherwise
            error('Unknown reproduction mode.');
    end
end

%% Crossover (Average or Convex)
function offspring = doCrossoverReal(parents, cfg)
    N = cfg.populationSize;
    offspring = parents;
    switch cfg.crossoverMode
        case 'average'
            for i = 1:(N-2)
                idx = randperm(N, 2);
                val = mean(parents(idx));
                val = min(max(val, cfg.valueRange(1)), cfg.valueRange(2));
                offspring(i) = val;
            end
        case 'convex'
            for i = 1:(N-2)
                idx = randperm(N, 2);
                r   = rand;
                val = r*parents(idx(1)) + (1-r)*parents(idx(2));
                val = min(max(val, cfg.valueRange(1)), cfg.valueRange(2));
                offspring(i) = val;
            end
        otherwise
            error('Unknown crossover mode.');
    end
end

%% Mutation (Uniform Real Mutation)
function mutants = doMutationReal(pop, cfg)
    N = cfg.populationSize;
    mutants = pop;
    if rand > 0.5
        d = (cfg.valueRange(2)-cfg.valueRange(1)) * 0.1 * (2*rand(1,N)-1);
        mutants = pop + d;
        mutants = min(max(mutants, cfg.valueRange(1)), cfg.valueRange(2));
    end
end
