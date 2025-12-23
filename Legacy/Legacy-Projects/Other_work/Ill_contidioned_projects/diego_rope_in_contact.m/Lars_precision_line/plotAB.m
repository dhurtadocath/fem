
A = load('withU0.mat').withU0;

B = load('woutU0.mat').woutU0;

figure(1)
titles = ["m" "|f|" "|du|" "iters"];
for i=1:4
    subplot(2,2,i)
    if i<4
        semilogy(Digs,A(:,i),'r'), hold on 
        semilogy(Digs,B(:,i),'b')
    else
        plot(Digs,A(:,i),'r'), hold on 
        plot(Digs,B(:,i),'b')
    end
    xlabel('Digits')
    legend('u0=u_{simple}','u0=0.0')

    title(titles(i))
end