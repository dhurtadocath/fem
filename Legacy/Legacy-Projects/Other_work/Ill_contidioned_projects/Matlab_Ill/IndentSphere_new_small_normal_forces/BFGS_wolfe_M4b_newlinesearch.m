clear all
close all
clc

delta_alpha=1e-1;

c_par=1e-4;
c_par2=0.9;
tol=1e-13;
tol2=1e-15;

n_var=80;

u=zeros(n_var,1);
G=zeros(n_var,1);
G(62)=1;
G(79)=1;

m_new=-G'*u;
for j=1:10
    m_new=m_new+(sqrt(u(2*j-1)^2+(1+u(2*j))^2)-1)^2;
end
for j=1:30
    m_new=m_new+(sqrt((1+u(2*j+20)-u(2*j))^2+(u(2*j+19)-u(2*j-1))^2)-1)^2;
end
for j=1:36
    fl=floor((j-1)/9);
    m_new=m_new+(sqrt((1+u(2*j+1+2*fl)-u(2*j-1+2*fl))^2+(u(2*j+2+2*fl)-u(2*j+2*fl))^2)-1)^2;
end
f_2=-G;
for j=1:10
    f_2([2*j-1;2*j])=f_2([2*j-1;2*j])+[(2*u(2*j-1)*(((u(2*j) + 1)^2 + u(2*j-1)^2)^(1/2) - 1))/((u(2*j) + 1)^2 + u(2*j-1)^2)^(1/2);
        ((((u(2*j) + 1)^2 + u(2*j-1)^2)^(1/2) - 1)*(2*u(2*j) + 2))/((u(2*j) + 1)^2 + u(2*j-1)^2)^(1/2)];
end
for j=1:30
    f1=((2*u(2*j-1) - 2*u(2*j+19))*(((u(2*j+20) - u(2*j) + 1)^2 + (u(2*j-1) - u(2*j+19))^2)^(1/2) - 1))/((u(2*j+20) - u(2*j) + 1)^2 + (u(2*j-1) - u(2*j+19))^2)^(1/2);
    f2=-((((u(2*j+20) - u(2*j) + 1)^2 + (u(2*j-1) - u(2*j+19))^2)^(1/2) - 1)*(2*u(2*j+20) - 2*u(2*j) + 2))/((u(2*j+20) - u(2*j) + 1)^2 + (u(2*j-1) - u(2*j+19))^2)^(1/2);
    f_2([j*2-1;j*2;j*2+19;j*2+20])=f_2([j*2-1;j*2;j*2+19;j*2+20])+[f1;f2;-f1;-f2];
end
for j=1:36
    fl=floor((j-1)/9);
    f1=-((((u(2*j+2*fl) - u(2*j+2+2*fl))^2 + (u(2*j+1+2*fl) - u(2*j-1+2*fl) + 1)^2)^(1/2) - 1)*(2*u(2*j+1+2*fl) - 2*u(2*j-1+2*fl) + 2))/((u(2*j+2*fl) - u(2*j+2+2*fl))^2 + (u(2*j+1+2*fl) - u(2*j-1+2*fl) + 1)^2)^(1/2);
    f2=((2*u(2*j+2*fl) - 2*u(2*j+2+2*fl))*(((u(2*j+2*fl) - u(2*j+2+2*fl))^2 + (u(2*j+1+2*fl) - u(2*j-1+2*fl) + 1)^2)^(1/2) - 1))/((u(2*j+2*fl) - u(2*j+2+2*fl))^2 + (u(2*j+1+2*fl) - u(2*j-1+2*fl) + 1)^2)^(1/2);
    f_2([2*j-1+2*fl;2*j+2*fl;2*j+1+2*fl;2*j+2+2*fl])=f_2([2*j-1+2*fl;2*j+2*fl;2*j+1+2*fl;2*j+2+2*fl])+[f1;f2;-f1;-f2];
end

tic
K_new_inv=inv(eye(n_var));
f_new=zeros(n_var,1);
m_old=1e100;
cnt=0;
while (m_old-m_new)>tol*abs(m_old)
    
    cnt=cnt+1;
    
    m_old=m_new;
    f_old=f_new;
    
    f_new=f_2;
    
    K_old_inv=K_new_inv;
    delta_f=f_new-f_old;
    
    if cnt==1
        h_new=-K_old_inv*f_new;
    else
        K_new_inv=K_old_inv+(delta_u'*delta_f+delta_f'*K_old_inv*delta_f)*delta_u*delta_u'/(delta_u'*delta_f)^2-(K_old_inv*delta_f*delta_u'+delta_u*delta_f'*K_old_inv)/(delta_u'*delta_f);
        h_new=-K_new_inv*f_new;
    end
    
    alpha1=0;
    m_1=m_new;
    alpha2=delta_alpha;
    ux=u+alpha2*h_new;
    m_2=-G'*ux;
    for j=1:10
        m_2=m_2+(sqrt(ux(2*j-1)^2+(1+ux(2*j))^2)-1)^2;
    end
    for j=1:30
        m_2=m_2+(sqrt((1+ux(2*j+20)-ux(2*j))^2+(ux(2*j+19)-ux(2*j-1))^2)-1)^2;
    end
    for j=1:36
        fl=floor((j-1)/9);
        m_2=m_2+(sqrt((1+ux(2*j+1+2*fl)-ux(2*j-1+2*fl))^2+(ux(2*j+2+2*fl)-ux(2*j+2*fl))^2)-1)^2;
    end

    alpha3=2*delta_alpha;
    ux=u+alpha3*h_new;
    m_3=-G'*ux;
    for j=1:10
        m_3=m_3+(sqrt(ux(2*j-1)^2+(1+ux(2*j))^2)-1)^2;
    end
    for j=1:30
        m_3=m_3+(sqrt((1+ux(2*j+20)-ux(2*j))^2+(ux(2*j+19)-ux(2*j-1))^2)-1)^2;
    end
    for j=1:36
        fl=floor((j-1)/9);
        m_3=m_3+(sqrt((1+ux(2*j+1+2*fl)-ux(2*j-1+2*fl))^2+(ux(2*j+2+2*fl)-ux(2*j+2*fl))^2)-1)^2;
    end

%     if cnt==79
%         ALPHA=0:0.01:1;
%         
%         res=zeros(length(ALPHA),1);
%         for i=1:length(ALPHA)
%             alpha2=ALPHA(i);
%             ux=u+alpha2*h_new;
%             m_2=-G'*ux;
%             for j=1:10
%                 m_2=m_2+(sqrt(ux(2*j-1)^2+(1+ux(2*j))^2)-1)^2;
%             end
%             for j=1:30
%                 m_2=m_2+(sqrt((1+ux(2*j+20)-ux(2*j))^2+(ux(2*j+19)-ux(2*j-1))^2)-1)^2;
%             end
%             for j=1:36
%                 fl=floor((j-1)/9);
%                 m_2=m_2+(sqrt((1+ux(2*j+1+2*fl)-ux(2*j-1+2*fl))^2+(ux(2*j+2+2*fl)-ux(2*j+2*fl))^2)-1)^2;
%             end
%             res(i)=m_2;
%         end
%         plot(ALPHA',res,'bo')
% asdf
%     end
    
    if m_3>m_1 && m_2>m_1
        ux=u;
        m_new=m_1;
        f_2=f_new;
    else
        cnt2=0;
        m_x=m_new;
        f_x=f_new;
        alphax=0.1;
        while alphax>0 && (m_x>m_new+c_par*alphax*h_new'*f_new || h_new'*f_x<c_par2*h_new'*f_new)
            cnt2=cnt2+1;

            pars=-[alpha1^2 alpha1 1;alpha2^2 alpha2 1;alpha3^2 alpha3 1]\[m_1;m_2;m_3];

            alphax=-pars(2)/2/pars(1);

            ux=u+alphax*h_new;
            m_x=-G'*ux;
            for j=1:10
                m_x=m_x+(sqrt(ux(2*j-1)^2+(1+ux(2*j))^2)-1)^2;
            end
            for j=1:30
                m_x=m_x+(sqrt((1+ux(2*j+20)-ux(2*j))^2+(ux(2*j+19)-ux(2*j-1))^2)-1)^2;
            end
            for j=1:36
                fl=floor((j-1)/9);
                m_x=m_x+(sqrt((1+ux(2*j+1+2*fl)-ux(2*j-1+2*fl))^2+(ux(2*j+2+2*fl)-ux(2*j+2*fl))^2)-1)^2;
            end
            f_x=-G;
            for j=1:10
                f_x([2*j-1;2*j])=f_x([2*j-1;2*j])+[(2*ux(2*j-1)*(((ux(2*j) + 1)^2 + ux(2*j-1)^2)^(1/2) - 1))/((ux(2*j) + 1)^2 + ux(2*j-1)^2)^(1/2);
                    ((((ux(2*j) + 1)^2 + ux(2*j-1)^2)^(1/2) - 1)*(2*ux(2*j) + 2))/((ux(2*j) + 1)^2 + ux(2*j-1)^2)^(1/2)];
            end
            for j=1:30
                f1=((2*ux(2*j-1) - 2*ux(2*j+19))*(((ux(2*j+20) - ux(2*j) + 1)^2 + (ux(2*j-1) - ux(2*j+19))^2)^(1/2) - 1))/((ux(2*j+20) - ux(2*j) + 1)^2 + (ux(2*j-1) - ux(2*j+19))^2)^(1/2);
                f2=-((((ux(2*j+20) - ux(2*j) + 1)^2 + (ux(2*j-1) - ux(2*j+19))^2)^(1/2) - 1)*(2*ux(2*j+20) - 2*ux(2*j) + 2))/((ux(2*j+20) - ux(2*j) + 1)^2 + (ux(2*j-1) - ux(2*j+19))^2)^(1/2);
                f_x([j*2-1;j*2;j*2+19;j*2+20])=f_x([j*2-1;j*2;j*2+19;j*2+20])+[f1;f2;-f1;-f2];
            end
            for j=1:36
                fl=floor((j-1)/9);
                f1=-((((ux(2*j+2*fl) - ux(2*j+2+2*fl))^2 + (ux(2*j+1+2*fl) - ux(2*j-1+2*fl) + 1)^2)^(1/2) - 1)*(2*ux(2*j+1+2*fl) - 2*ux(2*j-1+2*fl) + 2))/((ux(2*j+2*fl) - ux(2*j+2+2*fl))^2 + (ux(2*j+1+2*fl) - ux(2*j-1+2*fl) + 1)^2)^(1/2);
                f2=((2*ux(2*j+2*fl) - 2*ux(2*j+2+2*fl))*(((ux(2*j+2*fl) - ux(2*j+2+2*fl))^2 + (ux(2*j+1+2*fl) - ux(2*j-1+2*fl) + 1)^2)^(1/2) - 1))/((ux(2*j+2*fl) - ux(2*j+2+2*fl))^2 + (ux(2*j+1+2*fl) - ux(2*j-1+2*fl) + 1)^2)^(1/2);
                f_x([2*j-1+2*fl;2*j+2*fl;2*j+1+2*fl;2*j+2+2*fl])=f_x([2*j-1+2*fl;2*j+2*fl;2*j+1+2*fl;2*j+2+2*fl])+[f1;f2;-f1;-f2];
            end
            

            if alphax>alpha3

                alpha1=alpha2;
                alpha2=alpha3;
                alpha3=alphax;

                m_1=m_2;
                m_2=m_3;
                m_3=m_x;

            elseif alphax<alpha1

                alpha3=alpha2;
                alpha2=alpha1;
                alpha1=alphax;

                m_3=m_2;
                m_2=m_1;
                m_1=m_x;

            elseif alphax<alpha3 && alphax>alpha2

                if alphax-alpha1<alpha3-alpha2

                    alpha3=alphax;
                    m_3=m_x;

                else

                    alpha1=alpha2;
                    alpha2=alphax;

                    m_1=m_2;
                    m_2=m_x;

                end

            elseif alphax<alpha2 && alphax>alpha1

                if alpha3-alphax<alpha2-alpha1

                    alpha1=alphax;
                    m_1=m_x;

                else

                    alpha3=alpha2;
                    alpha2=alphax;

                    m_3=m_2;
                    m_2=m_x;

                end

            else

                err

            end

        end
        if alphax<=0
            f_2=f_new;
            ux=u;
        else
            f_2=f_x;
            m_new=m_x;
        end
    end
    cnt2
%     delta_f=f_2-f_new;
    delta_u=ux-u;
    u=ux;
    
end
toc
cnt
norm(f_2)

m_new
u

figure(1)
hold on
for j=1:10
    
    ux1=0;
    uy1=0;
    ux2=u(2*j-1);
    uy2=u(2*j);
    x1=j-1;
    x2=j-1;
    y1=0;
    y2=1;
    
    plot([x1+ux1;x2+ux2],[y1+uy1;y2+uy2],'r--')
    
end
for j=1:9
    
    ux1=0;
    uy1=0;
    ux2=u(2*j+1);
    uy2=u(2*j+2);
    x1=j-1;
    x2=j;
    y1=0;
    y2=1;
    
    plot([x1+ux1;x2+ux2],[y1+uy1;y2+uy2],'r--')
    
end
for j=1:9
    
    ux1=0;
    uy1=0;
    ux2=u(2*j-1);
    uy2=u(2*j);
    x1=j;
    x2=j-1;
    y1=0;
    y2=1;
    
    plot([x1+ux1;x2+ux2],[y1+uy1;y2+uy2],'r--')
   
    
end
for j=1:30
    
    ux1=u(2*j-1);
    uy1=u(2*j);
    ux2=u(2*j+19);
    uy2=u(2*j+20);
    x1=j-1-floor((j-1)/10)*10;
    x2=x1;
    y1=floor((j-1)/10)+1;
    y2=y1+1;
    
    plot([x1+ux1;x2+ux2],[y1+uy1;y2+uy2],'r--')
    
end
for j=1:36
    fl=floor((j-1)/9);
    
    ux1=u(2*j-1+2*fl);
    uy1=u(2*j+2*fl);
    ux2=u(2*j+1+2*fl);
    uy2=u(2*j+2+2*fl);
    x1=j-1-floor((j-1)/9)*9;
    x2=x1+1;
    y1=floor((j-1)/9)+1;
    y2=y1;
    
    plot([x1+ux1;x2+ux2],[y1+uy1;y2+uy2],'r--')
    
    
end
for j=1:27
    fl=floor((j-1)/9);
    
    ux1=u(2*j-1+2*fl);
    uy1=u(2*j+2*fl);
    ux2=u(2*j+21+2*fl);
    uy2=u(2*j+22+2*fl);
    x1=j-1-floor((j-1)/9)*9;
    x2=x1+1;
    y1=floor((j-1)/9)+1;
    y2=y1+1;
    
    plot([x1+ux1;x2+ux2],[y1+uy1;y2+uy2],'r--')
    
    
end

for j=1:27
    fl=floor((j-1)/9);
    
    ux1=u(2*j+1+2*fl);
    uy1=u(2*j+2+2*fl);
    ux2=u(2*j+19+2*fl);
    uy2=u(2*j+20+2*fl);
    x1=j-floor((j-1)/9)*9;
    x2=x1-1;
    y1=floor((j-1)/9)+1;
    y2=y1+1;
    
    plot([x1+ux1;x2+ux2],[y1+uy1;y2+uy2],'r--')
    
    
end