function ternaryPlay(ordDeg,bestOrWorstQn)
% usage:  ternaryPlay([inf,0],'worst') to see the worst question from the perspective of way nonconcave entropy
% usage:  ternaryPlay([5,1.8],'best') to see the best question from the perspective of Arimoto(order=5)
 
    close gcf;
 
    if nargin==0
        ordDeg=[1 1]
        bestOrWorstQn= 'best'
    end
    order=ordDeg(1);  
    degree=ordDeg(2);

    % plot an equilateral triangle with the vertices labeled
    % my triangle scheme
    % top, (x=0.5, y=k) means P(k1)=1
    % bottom right, (x=1, y=0) means P(k2)=1
    % bottom left, (x=0,   y=0) means P(k3)=1
    k=(1-.5^2)^.5;  
    figure
    hold on
    axis equal
    axis off
    ylim([-.02*k 1.02*k])
    %plotLabelledPoint([1 0 0],'   k_1')
    %plotLabelledPoint([0 1 0],'   k_2')
    %plotLabelledPoint([0 0 1],'   k_3')  
    %line([0,1],[0,0])
    %line([0,0.5],[0,k])
    %line([0.5,1],[k,0])
    
    % plot the basic entropy landscape
    incs=120;
    thisPdfInd=1;
    % the array pdfs includes the pdf values in cols 3, 4, and 5; 
    %   and an entropy value (for sorting) in col 1
    %   and the entropy according to the relevant SM relevance function in col 2
    for(i=0:incs)
        pk1= i/incs;
        for(j=0:incs)
            propOfRemainingToK2= j/incs;
            pk2= (1-pk1)*propOfRemainingToK2;
            pk3= 1-pk1-pk2;
            pdf= [pk1 pk2 pk3];
            prePdfs(thisPdfInd,:)=	 [entropyJN(pdf) entSmScaled(pdf,[degree order]) pdf];
            thisPdfInd= thisPdfInd+1;
        end
    end
    % sort the pdfs according to Shannon entropy, which is the first column.
    % i am hoping this will make it so that the most interesting points are visible, 
    %   and not covered up
    prePdfs=flipud(sortrows(prePdfs));
    % find out the max and min entropy values, so you can scale colors accordingly
    maxObtainedEnt= maxx(prePdfs(:,2));  
    minObtainedEnt= minn(prePdfs(:,2));  
    % now get rid of the entropy sort column, and just keep the actual pdfs
    smEntAndPdfs= prePdfs(:,2:5);
    % plot each point    
    [numPdfs,junk]= size(smEntAndPdfs);
    for(i=1:numPdfs)
        pdf= smEntAndPdfs(i,2:4);
        entThisPdf= smEntAndPdfs(i,1) + 0.001*normrnd(0,0.0000001);
        entThisPdf= max(min(entThisPdf,1),0);
        propOfMaxColorThisEnt= (entThisPdf-minObtainedEnt)./(maxObtainedEnt-minObtainedEnt);  
        thisColor= 0.99*[propOfMaxColorThisEnt^3 propOfMaxColorThisEnt^1.5 propOfMaxColorThisEnt^1.5];
        plotColorPoint(pdf,5,thisColor); 
    end
        
    % find the best question possible given this entropy family;
    %   and visualize it via its priors, and each possible answer
    if strcmp(bestOrWorstQn,'best') 
        posNegParam=1;
    elseif strcmp(bestOrWorstQn,'worst')
        posNegParam=-2;
    else
        disp('problem setting bestOrWorstQn in ternaryPlay.m; the input should be best or worst in single quotes')
        bestOrWorstQn
    end
    Qn=rand(2,3);
    for(i=[2000 8000 40000]); 
        [Qn,QnVal]= heuristicChangeQn(Qn,[degree order],i,posNegParam); 
    end
    priors= sum(Qn);
    e1= normalize(Qn(1,:));
    e2= normalize(Qn(2,:));
    plotLabelledPoint(priors,'   priors')
    plotLabelledPoint(e1,'   e_1')
    plotLabelledPoint(e2,'   e_2')
    
    set(gcf,'position',[440,358,686,420])
    set(gca,'position',[.05 .05 .8 .8])
    titleStr= ['SM(' cStr(order) ',' cStr(degree) '), euQn=' cStr(QnVal,',',3) ', Qn=[' cStr(Qn,', ',3) ']  .'] ;
    disp(titleStr);
    titleHandle= title(titleStr,'FontSize',15);
    titlePos= get(titleHandle,'position');
    titlePos(2)=0.95;
    set(titleHandle,'position',titlePos)
    fname= ['./ternaryPics/' 'ternary_ord' cStr(order) '_deg' cStr(degree) '_' bestOrWorstQn '_50k' '.png'];
    %fname= ['./ternaryPics/entropyOnly_ord' cStr(order) '_deg' cStr(degree) '.png'];
    disp(fname)
    set(gcf,'PaperPositionMode','auto')
    print(fname,'-dpng','-r150');
           
end


function plotColorPoint(pdfArr,markerSize,color)
    xyArrToPlot= probsToTernCoords(pdfArr);  
    plot(xyArrToPlot(1),xyArrToPlot(2),'^','MarkerFaceColor',color,'MarkerSize',markerSize,'MarkerEdgeColor',color)
end


function plotLabelledPoint(pdfArr,textStr)
    xyArrToPlot= probsToTernCoords(pdfArr);  
    plot(xyArrToPlot(1),xyArrToPlot(2),'o','MarkerFaceColor','none','MarkerSize',12,'LineWidth',3,'MarkerEdgeColor',[.5 0 .6])
    if length(textStr)>0
        text(xyArrToPlot(1),xyArrToPlot(2),[textStr ' (' cStr(pdfArr,', ',3) ')'],'FontSize',15,'color',[.5 0 .6])    
    end
end


function arrOutXY= probsToTernCoords(arrInXYZ)
    if ( (~isPdf(arrInXYZ)) || (length(arrInXYZ)~=3) )
        disp('problem with input in probsToTernCoords function in ternaryPlay.m')
        disp('input should be 1 by 3 array with probabilities that sum to 1')
        arrInXYZ
    else
        k=(1-.5^2)^.5;  
        outY= k*arrInXYZ(1);  
        pk2= arrInXYZ(2);  
        pk3= arrInXYZ(3);  
        outX= 0.5*(1+pk2-pk3);  
        arrOutXY=[outX outY];  
    end
end


