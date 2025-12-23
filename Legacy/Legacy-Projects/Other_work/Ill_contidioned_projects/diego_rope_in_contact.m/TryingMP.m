clear all
% close all
clc

addpath('C:\Users\xps\Documents\Multiprecision Computing Toolbox\')
mp.Digits(34);
format longG

%% INPUTS 
nrx=5;
nry=5;

tot_incr=100;

Utot=10;

k_pen=1;

alpha_init=1;
c_par=1e-4;
c_par2=0.9;
r_par=0.5;

tol=1e-25;
tol2=1e-20;

%% vars
xR=(nrx-1)/2;
yR=(nry-1)+4;
zR=4;
R=5;

global len_el
global len_u
global FE_node_coord
global FE_node_nrs

len_el=(nrx-1)*nry+(nry-1)*nrx;
len_u=nrx*nry*3;

%% Nodes coordinates
x=(0:1:nrx-1)';
FE_node_coord=zeros(nrx*nry,2);
for i=1:nry

    y=(i-1);

    FE_node_coord((i-1)*nrx+1:i*nrx,:)=[x y*ones(nrx,1)];
end

%% Connectivity
FE_node_nrs=zeros(len_el,2);
cnt=1;
for j=1:nry

    if j~=nry
        for i=1:nrx-1

            nr1=i+(j-1)*nrx;
            nr2=nr1+1;
            nr3=nr1+nrx;

            FE_node_nrs(cnt:cnt+1,:)=[nr1 nr2;nr1 nr3];
            cnt=cnt+2;
        end
    else
        for i=1:nrx-1

            nr1=i+(j-1)*nrx;
            nr2=nr1+1;

            FE_node_nrs(cnt,:)=[nr1 nr2];
            cnt=cnt+1;
        end

    end
end
for j=1:nry-1

    nr1=j*nrx;
    nr2=nr1+nrx;
    FE_node_nrs(cnt,:)=[nr1 nr2];
    cnt=cnt+1;
end


%% Minimization
u0=zeros(len_u,1,'mp');

options = optimoptions('fminunc','SpecifyObjectiveGradient',true); % indicate gradient is provided
x = fminunc(@(x) Energy(x),u0,options)



%% Functions
function [m_new,f] = Energy(u)
    global len_el
    global len_u
    global FE_node_coord
    global FE_node_nrs

    u

    m_new=mp('0');
    f=zeros(len_u,1,'mp');
    for i=1:len_el

        nr1=FE_node_nrs(i,1);
        nr2=FE_node_nrs(i,2);

        x1=FE_node_coord(nr1,1);
        y1=FE_node_coord(nr1,2);
        x2=FE_node_coord(nr2,1);
        y2=FE_node_coord(nr2,2);

        u1=u(nr1*3-2);
        v1=u(nr1*3-1);
        w1=u(nr1*3);
        u2=u(nr2*3-2);
        v2=u(nr2*3-1);
        w2=u(nr2*3);

        L=norm([x2+u2-x1-u1;y2+v2-y1-v1;w2-w1]);

        if L>1
            E=0.5*(L-1)^2;

            dEdu1=-(x2+u2-x1-u1)/L*(L-1);
            dEdv1=-(y2+v2-y1-v1)/L*(L-1);
            dEdw1=-(w2-w1)/L*(L-1);

            a=[nr1*3-2;nr1*3-1;nr1*3;nr2*3-2;nr2*3-1;nr2*3];
            f(a)=f(a)+[dEdu1;dEdv1;dEdw1;-dEdu1;-dEdv1;-dEdw1];

            m_new=m_new+E;
        end
    end
end

