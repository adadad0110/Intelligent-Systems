function rga1()
    clc; clear; close all;

    %% Configuration
    cfg.fitnessMode      = 'linear';   % 'power_law' or 'linear'
    cfg.reproductionMode = 'TS';          % 'RWS' or 'TS'
    cfg.populationSize   = 20;            
    cfg.maxGenerations   = 300;           
    cfg.valueRange       = [-10, 10];    
    cfg.crossoverPairs   = 4;             
    cfg.crossoverScale   = 0.1;           
    cfg.mutationProb     = 0.3;           
    cfg.mutationScaleMax = 0.5;           
    cfg.tolerance        = 1e-3;         
    %% Run GA
    [bestX, fitnessHist, genCount] = runRGA(cfg);

    %% Plot
    figure;
    plot(1:genCount, fitnessHist, '-');
    xlabel('Generation');
    ylabel('Max Fitness');
     if ~isempty(bestX)
        title( sprintf('RGA (%s) (%s) GEN:%d, x=%.4f', cfg.fitnessMode, cfg.reproductionMode, genCount, bestX),'Interpreter','none');
    else
        title( sprintf('RGA (%s) (%s) Reached Max Generations (%d)', cfg.fitnessMode, cfg.reproductionMode, cfg.maxGenerations),'Interpreter','none');
    end
end
%% Loop
function [bestX, maxFitHist, generation] = runRGA(cfg)
    pop = (cfg.valueRange(2)-cfg.valueRange(1)) .* rand(1, cfg.populationSize) + cfg.valueRange(1);
    maxFitHist = zeros(1, cfg.maxGenerations);
    bestX = [];

    xGrid   = linspace(cfg.valueRange(1), cfg.valueRange(2), 1e4);
    rawVals = arrayfun(@objective, xGrid);
    [termVal, ~] = max(rawVals);
    termFitRaw = termVal;
    termFit     = adaptFitness(termVal, cfg) * (1 - cfg.tolerance);

    for generation = 1:cfg.maxGenerations
        rawFit = objective(pop);
        fit    = adaptFitness(rawFit, cfg);
        maxFitHist(generation) = max(fit);

        if maxFitHist(generation) >= termFit
            bestIdx = find(fit == max(fit), 1);
            bestX   = pop(bestIdx);
            break;
        end

        parents  = selectParents(pop, fit, cfg);
        children = doCrossover(parents, cfg);
        mutated  = doMutation(children, cfg);

        [~, prevBest] = max(fit);
        mutated(1)    = pop(prevBest);
        pop = mutated;
    end

    maxFitHist = maxFitHist(1:generation);
end

%% Objective Function
function y = objective(x)
    y = -15*(sin(2*x)).^2 - (x-2).^2 + 160;
end

%% FitnessMode
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

%% reproduction
function parents = selectParents(pop, fit, cfg)
    N = numel(pop);
    parents = zeros(size(pop));
    switch cfg.reproductionMode
        case 'RWS'
            cumFit = cumsum(fit);
            total  = cumFit(end);
            rands  = rand(1, N) * total;
            for i = 1:N
                idx = find(cumFit >= rands(i), 1);
                parents(i) = pop(idx);
            end
        case 'TS'
            for i = 1:N
                cand = randperm(N, 2);
                [~, b] = max(fit(cand));
                parents(i) = pop(cand(b));
            end
        otherwise
            error('Unknown reproduction mode.');
    end
end

%% Crossover
function offspring = doCrossover(parents, cfg)
    N = numel(parents);
    offspring = parents;
    pairs = randperm(N, cfg.crossoverPairs*2);
    for i = 1:cfg.crossoverPairs
        i1 = pairs(2*i-1);
        i2 = pairs(2*i);
        sigma = (randi([-10 10]) * cfg.crossoverScale);
        delta = parents(i1) - parents(i2);
        o1 = parents(i1) + sigma * delta;
        o2 = parents(i2) - sigma * delta;

        if abs(o1)<=cfg.valueRange(2) && abs(o2)<=cfg.valueRange(2)
            offspring(i1) = o1;
            offspring(i2) = o2;
        end
    end
end

%% Mutation 
function mutants = doMutation(pop, cfg)
    mutants = pop;
    if rand < cfg.mutationProb
        scale = (randi([0 round(cfg.mutationScaleMax*10)]) * 0.1);
        noise = (randi([-10 10], size(pop)) * 0.1);
        cand  = mutants + scale .* noise;
        inRng = cand>=cfg.valueRange(1) & cand<=cfg.valueRange(2);
        mutants(inRng) = cand(inRng);
    end
end