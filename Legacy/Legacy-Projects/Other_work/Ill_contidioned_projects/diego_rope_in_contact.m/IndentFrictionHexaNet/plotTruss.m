function [p]=plotTruss(u,FE_node_coord,FE_node_nrs,dispNodes,p)
    len_el=length(FE_node_nrs);
    if nargin<5
        p = hggroup();
    end
    if nargin<4
        dispNodes=false;
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
        
        hold on
        plot3([x1+u1,x2+u2],[y1+v1,y2+v2],[z1+w1,z2+w2],'b','Parent',p)
    end

    if dispNodes
        for i=1:length(FE_node_coord)
            x1=FE_node_coord(i,1);
            y1=FE_node_coord(i,2);
            z1=FE_node_coord(i,3);

            text(x1,y1,z1,num2str(i),'Parent',p);
        end
    end


end