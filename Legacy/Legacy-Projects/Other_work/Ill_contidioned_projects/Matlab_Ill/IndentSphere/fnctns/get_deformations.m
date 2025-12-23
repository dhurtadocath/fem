function STRAINS = get_deformations(UU,FE_node_coord,FE_node_nrs)
    %GET_DEFORMATIONS Summary of this function goes here
    %   Detailed explanation goes here
    
    [ntimes,ndofs] = size(UU);
    len_el=length(FE_node_nrs);

    STRAINS = zeros(ntimes,len_el);

    for i=1:len_el
        nr1=FE_node_nrs(i,1);
        nr2=FE_node_nrs(i,2);

        x1=FE_node_coord(nr1,1);
        y1=FE_node_coord(nr1,2);
        z1=FE_node_coord(nr1,3);
        x2=FE_node_coord(nr2,1);
        y2=FE_node_coord(nr2,2);
        z2=FE_node_coord(nr2,3);

        for tt=1:ntimes
            u1=UU(tt,nr1*3-2);
            v1=UU(tt,nr1*3-1);      
            w1=UU(tt,nr1*3);
            u2=UU(tt,nr2*3-2);
            v2=UU(tt,nr2*3-1);
            w2=UU(tt,nr2*3);
            
            L0 = sqrt((x1-x2)^2+(y1-y2)^2+(z1-z2)^2);
            L  = sqrt((x1+u1-x2-u2)^2+(y1+v1-y2-v2)^2+(z1+w1-z2-w2)^2);

            STRAINS(tt,i)=(L-L0)/L0;

        end

    end



end

