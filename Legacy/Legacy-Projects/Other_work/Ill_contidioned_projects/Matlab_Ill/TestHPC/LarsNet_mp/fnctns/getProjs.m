function s = getProjs(xs, sph)
    nsi=size(xs,1);
    if isempty(xs)
        s=0;
    elseif nsi < 2   % if only one point...
        Cxy = cell2mat(sph{1});
        dxyi = xs - Cxy;
        si1 = sign(-dxyi(3)) * acos(-dxyi(1) / norm([dxyi(1), dxyi(3)])) / pi;
        si2 = 1 - acos(-dxyi(2) / norm(dxyi)) / pi;
        s = [si1, si2];
    else   % multiple points...
        s = zeros(nsi,2);
        for idx = 1:nsi
            s(idx, :) = getProjs(xs(idx, :), sph);
        end
    end
end