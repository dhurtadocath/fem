clear
close all
clc

addpath('~/AdvanpixMCT-5.0.0.15222')
mp.Digits(24);
format longG

ratio = 0;

nrx=21;

tot_incr=100;

Utot=mp(10);

k_pen=1;


% xR=mp(5);
% yR=mp(0);
% zR=mp(4);
% R=mp(6);
xR=5;
yR=0;
zR=4;
R =6;
sph = [xR,yR,zR,R];

figure(1)
hold on
view(0.0,0.0)
axis equal

[X,Y,Z]=sphere;

positiveZIndices = Z > 0;  % Indices where Z is positive
X(positiveZIndices) = NaN;  % Set X values of positive Z to NaN
Y(positiveZIndices) = NaN;  % Set Y values of positive Z to NaN
Z(positiveZIndices) = NaN;  % Set Z values of positive Z to NaN

X=X*R+xR;
Y=Y*R+yR;
Z=Z*R+zR;

%surf(double(X),double(Y),double(Z),'FaceColor',[0.5 0.5 0.5],'FaceAlpha',0.5,'EdgeColor','none');
h = surf(double(X),double(Y),double(Z), 'EdgeColor', 'none', 'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.5);  % Gray color
set(h, 'FaceLighting', 'none');  % Turn off lighting effects
camproj('perspective');  % Use perspective projection for a smoother rotation




FE_node_coord=mp([zeros(nrx,1) linspace(-5,5,nrx)' zeros(nrx,1)]);

FE_node_nrs=[(1:nrx-1)' (2:nrx)'];
len_el=nrx-1;
len_u=nrx*3;

% u=mp(zeros(nrx*3,1));
u=zeros(nrx*3,1);
% load u u
% u=mp(u);

free_ind=(4:1:(nrx-1)*3)';
fixed_ind=[1;2;3; (nrx*3-2:1:nrx*3)'];


figure(1)
hold on
p=plotTruss(u,FE_node_coord,FE_node_nrs);
drawnow


new_constraints=zeros(2,1);
new_constraints(1:2)=[];
tic


DIGS = 24:34;

ndigs = length(DIGS);
UERR = zeros(ndigs);
ITER = zeros(ndigs);
cnt_out=1;
for ndig=DIGS


    u=zeros(nrx*3,1);


    mp.Digits(ndig);
    FE_node_coord = mp(FE_node_coord);  % here I update the (possibly only) input that states the number of digits
    for incr=1:1
    
        u0=u;
    
        iter_out=0;
    
        iter_out=iter_out+1;

        old_constraints=new_constraints;

        u=u0;

        incr

        u(1)=mp(incr/tot_incr*Utot);
        u(nrx*3-2)=mp(incr/tot_incr*Utot);

        tic
        [u,mnew,iter] = BFGS(@(x) mf_with_constr(x,FE_node_coord,FE_node_nrs,free_ind,old_constraints,ratio,k_pen,sph),u,free_ind);

        ui = zeros(len_u,1);
        ui(1:3:end)=0.1*(incr);
        u_err=norm(ui-u)
        toc      
        
        new_constraints=zeros(nrx,1);
        cnt=1;
        for i=1:nrx
            
            x1=FE_node_coord(i,1);
            y1=FE_node_coord(i,2);
            z1=FE_node_coord(i,3);

            u1=u(i*3-2);
            v1=u(i*3-1);
            w1=u(i*3);

            if norm([x1+u1-xR;y1+v1-yR;z1+w1-zR])<R
                new_constraints(cnt)=i;
                cnt=cnt+1;
            end

        end
    
        delete(p);
        p=plotTruss(u,FE_node_coord,FE_node_nrs);
        drawnow
    
    end
    UERR(cnt_out)=u_err;
    ITER(cnt_out)=iter;
    cnt_out=cnt_out+1;

end

