function [u,m_new,iter] = BFGS(FUN,u0,free_ind)

    alpha_init=1;
    c_par=1e-4;
    c_par2=0.9;
    r_par=0.5;
    
    tol=0;
    tol2=1e-15;

    nfr = length(free_ind);
    [m_new,f]=FUN(u0);
    u=u0;
    f_2=f(free_ind);
    K_new_inv=inv(eye(nfr));
    f_new=zeros(nfr,1);
    iter=0;

    while norm(f_2-f_new)>tol && norm(f_2)>0

        iter=iter+1;
    
        f_old=f_new;
    
        f_new=f_2;
    
        K_old_inv=K_new_inv;
        delta_f=f_new-f_old;
    
        if iter==1
            h_new=-K_old_inv*f_new;
        else
            K_new_inv=K_old_inv+(delta_u'*delta_f+delta_f'*K_old_inv*delta_f)*(delta_u*delta_u')/(delta_u'*delta_f)^2-(K_old_inv*delta_f*delta_u'+delta_u*delta_f'*K_old_inv)/(delta_u'*delta_f);
            h_new=-K_new_inv*f_new;
        end
    
        alpha3=alpha_init;
        ux=u;
        ux(free_ind)=u(free_ind)+alpha3*h_new;
        [m_3,f]=FUN(ux);
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
            [m_3,f]=FUN(ux);
            f_3=f(free_ind);
    
            iter2a=iter2a+1;
    
            if m_3<=m_new+c_par*alpha3*h_new'*f_new && abs(h_new'*f_3)<=c_par2*abs(h_new'*f_new)
    
                signal1=1;
    
            end
    
        end
    
        if signal1==0
            alpha1=0;
   
            alpha2=alpha3/2;
            ux=u;
            ux(free_ind)=u(free_ind)+alpha2*h_new;
            [m_2,f]=FUN(ux);
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
                    [m_2,f]=FUN(ux);
                    f_2=f(free_ind);
    
                else
                    if h_new'*f_2<c_par2*h_new'*f_new
    
                        alpha1=alpha2;
                        alpha2=0.5*(alpha2+alpha3);
    
                        ux=u;
                        ux(free_ind)=u(free_ind)+alpha2*h_new;
                        [m_2,f]=FUN(ux);
                        f_2=f(free_ind);
    
                    elseif h_new'*f_2>-c_par2*h_new'*f_new
    
                        alpha3=alpha2;
                        m_3=m_2;
                        f_3=f_2;
    
                        alpha2=0.5*(alpha1+alpha2);
    
                        ux=u;
                        ux(free_ind)=u(free_ind)+alpha2*h_new;
                        [m_2,f]=FUN(ux);
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
end