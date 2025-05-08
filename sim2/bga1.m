function bga1()
    clc; clear; close all; 

    %% Configuration
    cfg.fitnessMode      = 'power_law';    % 'power_law' or 'linear'
    cfg.reproductionMode = 'RWS';       % 'RWS' or 'TS'
    cfg.crossoverMode    = 'mask'; % 'one-point', 'two-point', or 'mask'
    cfg.mutationMode     = 'string';    % 'bit', 'string', or 'uniform'
    cfg.populationSize   = 10;
    cfg.chromosomeLength = 10;
    cfg.maxGenerations   = 1000;
    cfg.decodeWeights    = [8 4 2 1 0.5 0.25 0.125 0.0625 0.03125];
    cfg.valueRange       = [-10, 10];

    %% Run GA
    [bestX, fitnessHistory, genCount] = runBGA(cfg);

    %% plot紀錄
    fitnessHistory = fitnessHistory(1:genCount); 
    fitnessHistory = fitnessHistory(:)';          

    %% Plot Results
    figure;
    plot(1:genCount, fitnessHistory, '-')
    grid on;
    if ~isempty(bestX)
        title( sprintf('BGA (%s) (%s) GEN:%d, x=%.4f', ...
               cfg.fitnessMode, cfg.reproductionMode, genCount, bestX), ...
               'Interpreter','none' );
    else
        title( sprintf('BGA (%s) (%s) Reached Max Generations (%d)', ...
               cfg.fitnessMode, cfg.reproductionMode, cfg.maxGenerations), ...
               'Interpreter','none' );
    end
end
%% Loop
function [bestX, maxFitHistory, generation] = runBGA(cfg)
    % Initialize
    pop = randi([0 1], cfg.populationSize, cfg.chromosomeLength);
    maxFitHistory = zeros(1, cfg.maxGenerations);
    bestX = [];

    trueMaxX    = fminbnd(@(x)-objective(x), cfg.valueRange(1), cfg.valueRange(2));
    step        = cfg.decodeWeights(end);                
    discX       = round(trueMaxX/step) * step;           
    termValDisc = objective(discX);                      
    termFit     = adaptFitness(termValDisc, cfg);        

    for generation = 1:cfg.maxGenerations
        decoded = decodePopulation(pop, cfg);
        fitness = adaptFitness(objective(decoded), cfg);
        maxFitHistory(generation) = max(fitness);
   
        if maxFitHistory(generation) >= termFit
            bestIdx = find(fitness == max(fitness), 1);
            bestX = decoded(bestIdx);
            break;
        end

        parents = selectParents(pop, fitness, cfg);
        offspring = doCrossover(parents, cfg);
        pop = doMutation(offspring, cfg);
    end

    % gen紀錄
    maxFitHistory = maxFitHistory(1:generation);
end

%% Objective Function
function y = objective(x)
    y = -15*(sin(2*x)).^2 - (x-2).^2 + 160;
end

%% Fitness mode
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

%% Decode Binary
function values = decodePopulation(pop, cfg)
    signBits = pop(:,1);
    fracBits = pop(:,2:end);
    decoded  = fracBits * cfg.decodeWeights';
    values   = decoded .* (1 - 2*signBits);
end

%% reproduction
function parents = selectParents(pop, fitness, cfg)
    N = size(pop,1);
    parents = zeros(size(pop));
    switch cfg.reproductionMode
        case 'RWS'
            cumFit = cumsum(fitness);
            total  = cumFit(end);
            r      = rand(1,N) * total;
            for i = 1:N
                idx = find(cumFit >= r(i), 1);
                parents(i,:) = pop(idx,:);
            end
        case 'TS'
            for i = 1:N
                cand = randperm(N,2);
                [~,b] = max(fitness(cand));
                parents(i,:) = pop(cand(b),:);
            end
        otherwise
            error('Unknown reproduction mode.');
    end
end

%% Crossover
function offspring = doCrossover(parents, cfg)
    N = size(parents,1);
    offspring = parents;
    pairs = randperm(N, N-2);
    for i = 1:2:length(pairs)-1
        a = pairs(i); b = pairs(i+1);
        switch cfg.crossoverMode
            case 'one-point'
                pt = randi(cfg.chromosomeLength);
                offspring([a b],:) = swapSegment(parents([a b],:), pt, pt);
            case 'two-point'
                pts = sort(randi(cfg.chromosomeLength,1,2));
                offspring([a b],:) = swapSegment(parents([a b],:), pts(1), pts(2));
            case 'mask'
                mask   = randi([0 1],1, cfg.chromosomeLength);
                tA     = parents(a,:);
                tB     = parents(b,:);
                offspring(a,:) = tA.*mask + tB.*(1-mask);
                offspring(b,:) = tB.*mask + tA.*(1-mask);
            otherwise
                error('Unknown crossover mode.');
        end
    end
end

%% Swap
function pairOff = swapSegment(pair, i1, i2)
    pairOff            = pair;
    pairOff(1,i1:i2) = pair(2,i1:i2);
    pairOff(2,i1:i2) = pair(1,i1:i2);
end

%% Mutation 
function mutants = doMutation(pop, cfg)
    mutants = pop;
    N       = size(pop,1);
    switch cfg.mutationMode
        case 'bit'
            i = randi(N);
            j = randi(cfg.chromosomeLength);
            mutants(i,j) = 1 - mutants(i,j);
        case 'string'
            if randi(10) == 1
                i = randi(N);
                mutants(i,:) = 1 - mutants(i,:);
            end
        case 'uniform'
            if randi(2) == 1
                i   = randi(N);
                pos = randi(cfg.chromosomeLength,1,2);
                mutants(i,pos) = 1 - mutants(i,pos);
            end
        otherwise
            error('Unknown mutation mode.');
    end
end
