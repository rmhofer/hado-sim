function outQn= qnPlayRandomOneOperation(maybeQnIn)
    workingQn= fixQnAndPenalizeButDontComplain(maybeQnIn);
    numAvailOps= 7;
    numelQn= numel(workingQn);
    sizeQn= size(workingQn);
    if isQn(workingQn)
        % change workingQn according to a single randomly chosen operation
        whichOp= round(rand*numAvailOps*.9998 + .5001);
        if whichOp==1
            % round some of the items in the array to some number of decimal places
            placesToRound= round(rand*16*.9998+.5001);
            numValsToMessWith= min(1+poissrnd(numelQn*0.3),numelQn);
            scrambledOrderIndices= randomizeString([1:1:numelQn]);
            indicesToMessWith= scrambledOrderIndices(1,1:numValsToMessWith);
            workingQn(indicesToMessWith)= roundJN(workingQn(indicesToMessWith),placesToRound);
        elseif whichOp==2
            % increase entropy.
            % I think I could make things more robust and scale better if I applied this only to a subset of items
            propWayToMaxEnt= betarnd(0.8,0.8);
            maxEntDist= ones(sizeQn)./numelQn;
            workingQn= (1-propWayToMaxEnt)*workingQn + propWayToMaxEnt*maxEntDist;
        elseif whichOp==3
            % decrease entropy
            propWayToMinEnt= betarnd(0.8,0.8); 
            % I think I could make things more robust and scale better if I applied this only to a subset of items
            whichItemIsMax= find(workingQn==max(max(workingQn)));
            minEntDist= zeros(sizeQn); 
            minEntDist(whichItemIsMax)=1;
            workingQn= (1-propWayToMinEnt)*workingQn + propWayToMinEnt*minEntDist;
        elseif whichOp==4
            % add a *single* noisy value to some values within the array
            numValsToMessWith= min(1+poissrnd(numelQn*0.3),numelQn);
            scrambledOrderIndices= randomizeString([1:1:numelQn]);
            indicesToMessWith= scrambledOrderIndices(1,1:numValsToMessWith);
            %workingQn(indicesToMessWith)= workingQn(indicesToMessWith) + normrnd(0,rand*0.1,[1 1]);
            % I want to move it a random amount because I think i'm moving too much in late stages
            workingQn(indicesToMessWith)= workingQn(indicesToMessWith) + normrnd(0,betarnd(.4,.6),[1 1]);
        elseif whichOp==5
            % add noise to some values within the array
            numValsToMessWith= min(1+poissrnd(numelQn*0.3),numelQn);
            scrambledOrderIndices= randomizeString([1:1:numelQn]);
            indicesToMessWith= scrambledOrderIndices(1,1:numValsToMessWith);
            %workingQn(indicesToMessWith)= workingQn(indicesToMessWith) + normrnd(0,rand*0.1,size(indicesToMessWith));
            % I want to move it a random amount because I think i'm moving too much in late stages
            workingQn(indicesToMessWith)= workingQn(indicesToMessWith) + normrnd(0,betarnd(.4,.6),size(indicesToMessWith));
        elseif whichOp==6
            % get a new completely random question and average it with the existing question
            newRandQn= normalize(rand(sizeQn));  
            if isQn(newRandQn)
                propWayToRand= betarnd(0.8,0.8);
                workingQn= (1-propWayToRand)*workingQn + propWayToRand*newRandQn;
            end
        elseif whichOp==7
            % take a subset of values and set them to be closer to each other
            numValsToEqualize= min(1+poissrnd(numelQn*0.3),numelQn);
            scrambledOrderIndices= randomizeString([1:1:numelQn]);
            indicesToAverage= scrambledOrderIndices(1,1:numValsToEqualize);
            equalizedArr= workingQn;
            equalizedArr(indicesToAverage)= mean(mean(equalizedArr(indicesToAverage)));
            %propWayToEqual= roundJN(rand,1);
            % I think it might be faster if it tends to be high or low proportions
            propWayToEqual= betarnd(0.5,0.5);
            outArr= (1-propWayToEqual)*workingQn + propWayToEqual*equalizedArr;
        else
            disp('there is be a problem assigning random ops in qnPlayRandomOneOperation; check the number of random operations')
            whichOp
            numAvailOps
        end
        % make sure that you return a legit question array.
        % if what you have after this operation is not legit,
        %   then return the input that you initially started with, without further ado  
        if sum(sum(workingQn))<0
            workingQn= workingQn - min(min(workingQn));
        end
        workingQn= fixQnAndPenalizeButDontComplain(workingQn);
        if isQn(workingQn)
            outQn=workingQn;    
        else
            outQn= maybeQnIn;
        end
    else
        disp('bogus input in qnPlayRandomOneOperation.m; here is input and modified Qn array:')
        maybeQnIn
        workingQn
    end
end


function [QnOut,penalty]= fixQnAndPenalizeButDontComplain(maybeQnArr)
    % a question array should be a pdf that sums to 1
    % if not, chop it to make it sum to 1
    % but return a penalty term to correct the optimizations
    
    if isQn(maybeQnArr)
        QnOut= maybeQnArr;
        penalty= 0;
    else
        workArr= maybeQnArr;
        workArr= max(workArr,0);
        workArr= min(workArr,1);
        sumNotPositive= sum(sum(workArr))<=0 ;
        if sumNotPositive
            while (workArr==zeros(size(workArr)))
                % just use something completely random, in this case
                workArr= normalize(betarnd(1,1,size(workArr)));  
            end
        end
        workArr= normalize(workArr);
        QnOut= workArr;
        penalty= sum(sum(abs(maybeQnArr - workArr)));
    end    
end