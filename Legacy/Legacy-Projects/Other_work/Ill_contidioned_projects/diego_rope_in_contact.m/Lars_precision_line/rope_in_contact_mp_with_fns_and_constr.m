clear all
% close all
clc

digits(100)

addpath('~/AdvanpixMCT-5.0.0.15222')
mp.Digits(15);
format longG

ratio = 0;

nrx=21;

tot_incr=100;

Utot=mp(10);

k_pen=mp(1);

alpha_init=mp(1);
c_par=mp(1e-4);
c_par2=mp(0.9);
r_par=mp(0.5);

tol=mp(0);
tol2=mp(1e-15);


xR=mp(5);
yR=mp(0);
zR=mp(4);
R=mp(6);
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

FE_node_nrs=mp([(1:nrx-1)' (2:nrx)']);
len_el=nrx-1;
len_u=nrx*3;

u=mp(zeros(nrx*3,1));
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
for incr=1:tot_incr

    u0=u;

    YN=1;
    iter_out=0;
    while YN==1

        iter_out=iter_out+1;

        old_constraints=new_constraints;

        u=u0;

        incr

        u(1)=mp(incr/tot_incr*Utot);
        u(nrx*3-2)=mp(incr/tot_incr*Utot);



        [m_new,f]=m_and_f(u,FE_node_coord,FE_node_nrs,free_ind,ratio);
        [mc,fc]=mf_constr(u,FE_node_coord,old_constraints,k_pen,sph);
        m_new=m_new+mc;
        f=f+fc;
        f_2=f(free_ind);
        f_res=norm(f(fixed_ind));

        tic
        K_new_inv=inv(eye(len_u-6));
        f_new=zeros(len_u-6,1);
        f_old=ones(len_u-6,1);
        m_old=1e100;
        iter=0;
        while norm(f_2-f_new)>0 && norm(f_2)>0
%         while (m_old-m_new)>tol*abs(m_old)

            iter=iter+1;

            m_old=m_new;
            f_old=f_new;

            f_new=f_2;

            K_old_inv=K_new_inv;
            delta_f=f_new-f_old;

            if iter==1
                h_new=-K_old_inv*f_new;
            else
                K_new_inv=K_old_inv+(delta_u'*delta_f+delta_f'*K_old_inv*delta_f)*delta_u*delta_u'/(delta_u'*delta_f)^2-(K_old_inv*delta_f*delta_u'+delta_u*delta_f'*K_old_inv)/(delta_u'*delta_f);
                h_new=-K_new_inv*f_new;
            end

            alpha3=alpha_init;
            ux=u;
            ux(free_ind)=u(free_ind)+alpha3*h_new;
            [m_3,f]=m_and_f(ux,FE_node_coord,FE_node_nrs,free_ind,ratio);
            [mc,fc]=mf_constr(ux,FE_node_coord,old_constraints,k_pen,sph);
            m_3=m_3+mc;
            f=f+fc;
            f_3=f(free_ind);

            signal1=0;
            if m_3<=m_new+c_par*alpha3*h_new'*f_new && abs(h_new'*f_3)<=c_par2*abs(h_new'*f_new)
                signal1=1;
            end

            iter2a=0;
            while m_3<m_new+c_par*alpha3*h_new'*f_new && h_new'*f_3<-c_par2*h_new'*f_new && signal1==0

                alpha3=alpha3/r_par;

                ux=u;
                ux(free_ind)=u(free_ind)+alpha3*h_new;
                [m_3,f]=m_and_f(ux,FE_node_coord,FE_node_nrs,free_ind,ratio);
                [mc,fc]=mf_constr(ux,FE_node_coord,old_constraints,k_pen,sph);
                m_3=m_3+mc;
                f=f+fc;
                f_3=f(free_ind);


                iter2a=iter2a+1;

                if m_3<=m_new+c_par*alpha3*h_new'*f_new && abs(h_new'*f_3)<=c_par2*abs(h_new'*f_new)

                    signal1=1;

                end

            end

            if signal1==0
                alpha1=0;
                m_1=m_new;
                f_1=f_new;

                alpha2=alpha3/2;
                ux=u;
                ux(free_ind)=u(free_ind)+alpha2*h_new;
                [m_2,f]=m_and_f(ux,FE_node_coord,FE_node_nrs,free_ind,ratio);
                [mc,fc]=mf_constr(ux,FE_node_coord,old_constraints,k_pen,sph);
                m_2=m_2+mc;
                f=f+fc;
                f_2=f(free_ind);

                signal2=0;
                iter2=0;
                while signal2==0

                    iter2=iter2+1;

                    if alpha3-alpha1<tol2

                        signal2=1;
                        m_2=m_new;
                        f_2=f_new;

                    elseif m_2>m_new+c_par*alpha2*h_new'*f_new

                        alpha3=alpha2;
                        m_3=m_2;
                        f_3=f_2;

                        alpha2=0.5*(alpha1+alpha2);

                        ux=u;
                        ux(free_ind)=u(free_ind)+alpha2*h_new;
                        [m_2,f]=m_and_f(ux,FE_node_coord,FE_node_nrs,free_ind,ratio);
                        [mc,fc]=mf_constr(ux,FE_node_coord,old_constraints,k_pen,sph);
                        m_2=m_2+mc;
                        f=f+fc;
                        f_2=f(free_ind);

                    else
                        if h_new'*f_2<c_par2*h_new'*f_new

                            alpha1=alpha2;
                            m_1=m_2;
                            f_1=f_2;

                            alpha2=0.5*(alpha2+alpha3);

                            ux=u;
                            ux(free_ind)=u(free_ind)+alpha2*h_new;
                            [m_2,f]=m_and_f(ux,FE_node_coord,FE_node_nrs,free_ind,ratio);
                            [mc,fc]=mf_constr(ux,FE_node_coord,old_constraints,k_pen,sph);
                            m_2=m_2+mc;
                            f=f+fc;
                            f_2=f(free_ind);

                        elseif h_new'*f_2>-c_par2*h_new'*f_new

                            alpha3=alpha2;
                            m_3=m_2;
                            f_3=f_2;

                            alpha2=0.5*(alpha1+alpha2);

                            ux=u;
                            ux(free_ind)=u(free_ind)+alpha2*h_new;
                            [m_2,f]=m_and_f(ux,FE_node_coord,FE_node_nrs,free_ind,ratio);
                            [mc,fc]=mf_constr(ux,FE_node_coord,old_constraints,k_pen,sph);
                            m_2=m_2+mc;
                            f=f+fc;
                            f_2=f(free_ind);

                        else

                            signal2=1;

                        end
                    end
                end
            end
            delta_u=ux(free_ind)-u(free_ind);

            u=ux;

            if signal1==1
                m_new=m_3;
                f_2=f_3;
            else
                m_new=m_2;
            end

        end

        norm(f_2-f_new)
        norm(f_2)

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

        old_constraints
        new_constraints
        
        YN=0;
        if length(old_constraints)~=length(new_constraints)
            YN=1;
        else
            if isequal(old_constraints,new_constraints)
            else
                YN=1;
            end
        end

    end
    toc

    delete(p);
    p=plotTruss(u,FE_node_coord,FE_node_nrs);
    drawnow

end

