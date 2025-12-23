function [p]=plotTruss(u,FE_node_coord,FE_node_nrs,dispNodes,p,STRAINS,tt,last)
    len_el=length(FE_node_nrs);
    if nargin<5
        p = hggroup();
    end

    if nargin>5
        minstr = min(STRAINS,[],'all');
        maxstr = max(STRAINS,[],'all');
        strains = STRAINS(tt,:);
        mycolormap = jet(64);  % Define colormap
    end

    for i=1:len_el
        nr1=FE_node_nrs(i,1);
        nr2=FE_node_nrs(i,2);

        x1=FE_node_coord(nr1,1);
        y1=FE_node_coord(nr1,2);
        z1=FE_node_coord(nr1,3);
        x2=FE_node_coord(nr2,1);
        y2=FE_node_coord(nr2,2);
        z2=FE_node_coord(nr2,3);

        u1=u(nr1*3-2);
        v1=u(nr1*3-1);      
        w1=u(nr1*3);
        u2=u(nr2*3-2);
        v2=u(nr2*3-1);
        w2=u(nr2*3);
        
        if nargin>5
            strain = strains(i);
            normalizedStrain = (strain - minstr) / (maxstr - minstr);
            colorIndex = max(1, round(normalizedStrain * 63) + 1);  % +1 because MATLAB indices start at 1
            color = mycolormap(colorIndex, :);
            plot3([x1+u1,x2+u2],[y1+v1,y2+v2],[z1+w1,z2+w2],'Color',color,'Parent',p)
        else
            plot3([x1+u1,x2+u2],[y1+v1,y2+v2],[z1+w1,z2+w2],'b','Parent',p)
        end 
    end


    if nargin>3
        if length(dispNodes)==1
            if dispNodes==false
                dispNodes = [];
            end
        end

        for i=dispNodes
            x1=FE_node_coord(i,1)+u(i*3-2);
            y1=FE_node_coord(i,2)+u(i*3-1);
            z1=FE_node_coord(i,3)+u(i*3);

            plot3(x1,y1,z1,'ro','Parent',p);
            % text(x1,y1,z1,int2str(i),'Parent',p);
        end
    end

    if nargin>5
        if  nargin<8
            last = true;
        end

        if last
            colormap(jet);
            c = colorbar;   % Add colorbar
            ylabel(c, 'Strain');  % Add a label to the colorbar
            clim([minstr maxstr]);
        end
    end


end