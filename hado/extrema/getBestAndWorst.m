function getBestAndWorst
    
    oneCaseBestOrWorst(2,'best')
    oneCaseBestOrWorst(2,'worst')
    oneCaseBestOrWorst(3,'best')
    oneCaseBestOrWorst(3,'worst')
end

function oneCaseBestOrWorst(numKatVals,bestOrWorst)
    ordVals= 2.^[-3:1:6];
    degVals= ordVals;
    optPow= 4;     
    disp(' ')
    fname= ['katVals' cStr(numKatVals) '_' bestOrWorst '_numOpt_1e' cStr(optPow) '.csv'];
    disp(fname)
    for(i=1:length(ordVals))
        order= ordVals(i);
        for(j=1:length(degVals))
            degree= degVals(j);
            if strcmp(bestOrWorst,'best')
                bestOrWorstNum=1;
            elseif strcmp(bestOrWorst,'worst')
                bestOrWorstNum=-1;
            else
                disp('failed to specify if best or worst question in oneCaseBestOrWorst in getBestAndWorst.m')
            end
            mostQn=[];
            mostQnVal= -inf;
            for(k=1:10)
                [thisQn,thisQnVal]= heuristicChangeQn(normalize(rand(2,numKatVals)),[degree order],10^(optPow-1),bestOrWorstNum,'noOutput');
                if (thisQnVal*bestOrWorstNum)>mostQnVal
                   mostQn= thisQn;
                   mostQnVal= thisQnVal;
                end
            end
            [thisQn,thisQnVal]= heuristicChangeQn(normalize(mostQn),[degree order],10^(optPow),bestOrWorstNum);
            qnValArr(i,j)= thisQnVal;
        end
        pause(5);
    end
    csvwrite(fname,qnValArr);
    pause(100);
end    
