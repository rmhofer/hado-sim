function [worstQn,worstQnVal]= heuristicChangeQn(startQn,degOrd,numTries,plusOneIfBestMinusOneIfWorst,outputOrNot)
%
    if nargin==4
        outputOrNot= '';
    end
    maxPlacesToConsider= 16;
    workQn= sortQn(normalize(startQn));
    workQnVal= valQnSmRaw(workQn,degOrd); 
    for(i=1:numTries); 
        newQn= qnPlayRandomOneOperation(workQn); 
        if isQn(newQn)
            newQnVal= valQnSmRaw(newQn,degOrd); 
            if (roundJN(newQnVal,maxPlacesToConsider) - roundJN(workQnVal,maxPlacesToConsider))*plusOneIfBestMinusOneIfWorst > 0
                workQn= newQn;
                workQnVal= valQnSmRaw(workQn,degOrd);
            end
        end    
    end
    
    worstQn=sortQn(workQn);
    worstQnVal= workQnVal;
    
    dispDigs= 14;
    if ~strcmp(outputOrNot,'noOutput')
        disp(['order_degree=[' cStr([degOrd(2) degOrd(1)]) '], euQn=' cStr(worstQnVal,',',dispDigs)  ', Qn=[' cStr(worstQn,',',dispDigs) ']'])
    end
        
end

