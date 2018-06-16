function checkExpMat
    
    % coded so subjects always prefer the first test
    casesExp= [
    0.7 0    0.24 0.57 0     % expt 1, cond 1, psych sci exp matters
    0.7 0    0.29 0.57 0     % expt 1, cond 2, psych sci exp matters
    0.7 0.05 0.95 0.57 0     % expt 1, cond 4, psych sci exp matters
    0.5 0    0.50 0.25 0.75  % expt 3, cond 1, psych sci exp matters
    0.7 0    0.15 0.57 0     % expt 3, cond 2, psych sci exp matters
    0.7 0.04 0.37 0.57 0     % expt 3, cond 3, psych sci exp matters
    0.7 0.04 0.60 0.57 0     % see re-re-re-re-rebuttal letter from psych sci exp matter
    0.44 0.56 0.30 0 0.22    % meder & nelson, JDM, asymm search, env 1
    0.36 0.14 0.80 0 0.44    % meder & nelson, JDM, asymm search, env 2
    % the following cases appear to not discriminate among any of the SM info gain measures
    %0.7 0    0.40 0.73 0.22  % expt 1, cond 3, psych sci exp matters
    %0.67 0.95 0  0 0.45      % nelson filimon cottrell, in prep, expt 1
    %0.67 0.95 0  0 0.81      % nelson filimon cottrell, in prep, expt 2
    ] ;  
    
    casesExpStrongPrefsHighConf= [
    0.7 0.05 0.95 0.57 0     % expt 1, cond 4, psych sci exp matters
    0.7 0.04 0.60 0.57 0     % see re-re-re-re-rebuttal letter from psych sci exp matter
    0.8  0.30  1  .64 .64    %I made this up but i think it's a no brainer
    ] ;  
    
    casesExpFlipItForFunnySubjectsIfHighConf= [
    0.7 0.57 0  0    0.24    % expt 1, cond 1, psych sci exp matters
    0.7 0.57 0  0    0.29    % expt 1, cond 2, psych sci exp matters
    0.7 0.57 0 0.04 0.37     % expt 3, cond 3, psych sci exp matters    
    0.7 0.05 0.95 0.57 0     % expt 1, cond 4, psych sci exp matters
    0.7 0.04 0.60 0.57 0     % see re-re-re-re-rebuttal letter from psych sci exp matter    
    ] ;
    
    % individual tests from for instance 2005 useful questions paper
    casesDes= [
    % from the words-and-numbers tasks in the 2005 useful questions paper
    0.5  0.70 0.30    0.99  1.00
    0.5  0.30 0.0001  0.99  1.00
    0.5  0.01 0.99    0.99  1.00
    0.5  0.30 0.0001  0.70  0.30
    0.5  0.01  0.99   0.30 0.0001
    0.5  0.01  0.99   0.70 0.30
    % skov and sherman, etc: extreme vs moderate likelihoods
    .5  .90 .55  .65 .30
    % skov and sherman: cases that don't really tell us anythning
    .5 .26 .74 .48 .52
    .5 .26 .74 .34 .66
    .5 .34 .66 .48 .52
    % cases from the 2010 experience matters paper.  some were clsoe to chance preferences, though.
    % DO WE KNOW WHAT THE words-and-numbers preferences were in experiment 3?
    0.7 0.57 0    0    0.24  % expt 1, cond 1, psych sci exp matters
    0.7 0.57 0    0    0.29  % expt 1, cond 2, psych sci exp matters
    0.7 0.05 0.95 0.57 0     % expt 1, cond 4, psych sci exp matters
    0.5 0    0.50 0.25 0.75  % expt 3, cond 1, psych sci exp matters
    % cases from turtle island experiments 1 through 4 (as numbered in charleys thesis)
    % i think some numbers in the trees within the paper are wrong.  
    % in any case i have to check this carefully.
    0.50 0.10 0.80 0.10 0.30
    0.70 0.41 0.93 0.03 0.30
    0.70 0.43 1.00 0.04 0.37
    0.72 0.03 0.83 0.39 1.00
    ] ;
    
    % which set of scenarios do you want to use?
    %cases= casesExp
    cases= casesDes
    %cases= casesExpStrongPrefsHighConf;
    
    [numCases junk]= size(cases);  
    
    orderVals= [0 2.^[-3:1:8] ];
    degreeVals= fliplr(orderVals);
    score= zeros(length(degreeVals),length(orderVals));
    %maxValCount= 0;
    for(i=1:numCases)
        thisCase= cases(i,:);
        p_h1= thisCase(1);
        D= thisCase(2:3);
        E= thisCase(4:5);
        %if maxVal([p_h1 D])>maxVal([p_h1 E])
        %    maxValCount= maxValCount+1;
        %elseif maxVal([p_h1 D])==maxVal([p_h1 E])
        %    maxValCount= maxValCount+0.5;
        %end
        for (j=1:length(degreeVals))
            thisDegree= degreeVals(j);
            for (k=1:length(orderVals))
                thisOrder= orderVals(k);
                %igScale= infoGainTwoCatSm([p_h1 0 1], [thisDegree,thisOrder]);
                thisIgD= infoGainTwoCatSm([p_h1 D], [thisDegree,thisOrder]);
                ig_D{i}(j,k)= thisIgD;
                thisIgE= infoGainTwoCatSm([p_h1 E], [thisDegree,thisOrder]);
                ig_E{i}(j,k)= thisIgE;
                %ig_diff{i}(j,k)= [ig_D{i}(j,k) - ig_E{i}(j,k)]./igScale;
                ig_bin{i}(j,k)= thisIgD > thisIgE;
                firstQnHigherIg= (thisIgD > thisIgE);
                qnsTieIg= (thisIgD == thisIgE);
                secondQnHigherIg= (thisIgD < thisIgE) ;
                if firstQnHigherIg
                    score(j,k)= score(j,k) + 1; 
                elseif qnsTieIg
                    % if it is a tie, count the score as 0.5 instead of 1
                    % this happens a lot with the zero-order measures.
                    %   if it happens not in the case of a zero-order measure, then say something
                    score(j,k)= score(j,k) + 0.5; 
                    if thisOrder>0
                        disp('tie for information gain (relevance) values in checkExpMat.m')
                        [p_h1 D E]
                        thisDegree
                        thisOrder
                    end
                elseif secondQnHigherIg
                    % if you picked wrong add nothing to the score
                else
                    disp('trouble figuring out which question has higher ig in checkExpMat.m')
                end
            end
        end
        % if you want, you can also visualize each individual problem type
        %plotSomething(ig_bin{i},degreeVals,orderVals,cStr(thisCase))
    end
    %maxValScore= maxValCount / numCases
    toPlotArr= score./numCases;
    %meanMinMaxText= ['mean ' num2str(mean(mean(toPlotArr))) ', min ' num2str(min(min(toPlotArr))) ', max ' num2str(max(max(toPlotArr))) ', maxValScore=' cStr(maxValScore)];
    meanMinMaxText= ['mean ' num2str(mean(mean(toPlotArr))) ', min ' num2str(min(min(toPlotArr))) ', max ' num2str(max(max(toPlotArr))) ];
    plotSomething(toPlotArr,degreeVals,orderVals,['proportion of cases (out of ' num2str(numCases) ') correctly predicted, ' meanMinMaxText ])
    csvwrite('expMatt.csv',normalize(toPlotArr))
end


function plotSomething(arrIn,degreeVals,orderVals,titleText)
    figure
    minVal= min(0,min(min(arrIn)));
    maxVal= max(1,max(max(arrIn)));
    imagesc(arrIn,[minVal maxVal])
    axis square
    title(titleText)
    xlabel('order')
    ylabel('degree','interpreter','LaTex')
    %numTicksDesired= 11;  
    numIncs= length(degreeVals);
    incsBetweenTicks= 4;
    desiredXTickIncs= [1:incsBetweenTicks:numIncs];  
    desiredXTickVals= orderVals(desiredXTickIncs); 
    set(gca, 'TickDir', 'out') 
    set(gca,'XTick',desiredXTickIncs);  
    set(gca,'XTickLabel',desiredXTickVals);  
    desiredYTickIncs= desiredXTickIncs;
    desiredYTickVals= desiredXTickVals;
    set(gca,'YTick',desiredYTickIncs);
    set(gca,'YTickLabel',fliplr(desiredYTickVals));
    colormap bone
    colorbar
end