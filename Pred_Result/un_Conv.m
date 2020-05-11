%提取所有score文件
clear all;
shuliang = 125;
for i = 1 : shuliang
    name = sprintf('results%d.npy',i);
    npy = readNPY(name);
    score(:,i) = npy;   %每一列是一条数据的score数据
end
    

%npy = readNPY('results2.npy');
% for i = 1 : length(npy)
%     a(i,1) = npy(:,:,i);
% end
for i = 1 : shuliang
    a = score(:,i);
    %a(1987:1988,1) = 0.00000000000000001;
    a = mapminmax(a,0,1);
    score_data(:,i) = a;
end
clear a;    
v1 = [0.9,1.2,1.5,2,1.5,1.2,0.9];
v1 = v1/sum(v1,2);
v2 = [1.2,1.5,2,2,1.5,1.2];
v2 = v2/sum(v2,2);
for w = 1 : shuliang
    for i = 1 : 1989
        for j = 1 : length(v1)
            a = score_data(:,w);
            B = a(i,1);
            Conv1(j,i) = B*v1(1,j);
        end
    end
    Convt = [];
    last_un_Conv = [];

    for i = 1 : length(a)+6
        if i == 1
            Convt(i,1) = Conv1(1,1);
        elseif i == 2
            Convt(i,1) = (Conv1(1,2) + Conv1(2,1)) / i;
        elseif i == 3
            Convt(i,1) = (Conv1(1,3) + Conv1(2,2) + Conv1(3,1)) / i;
        elseif i ==4
            Convt(i,1) = (Conv1(1,4) + Conv1(4,1) + Conv1(3,2) + Conv1(2,1)) / i;
        elseif i == 5
            Convt(i,1) = (Conv1(1,5) + Conv1(5,1) + Conv1(2,4) + Conv1(4,2) + Conv1(3,3)) / i;
        elseif i == 6
            Convt(i,1) = (Conv1(1,6) + Conv1(6,1) + Conv1(2,5) + Conv1(5,2) + Conv1(3,4) + Conv1(4,3)) / i;
        elseif i == 1990
            Convt(i,1) = (Conv1(7,i-6) + Conv1(2,i-1) + Conv1(3,i-2) + Conv1(4,i-3) + Conv1(5,i-4) + Conv1(6,i-5)) / 6;
        elseif i == 1991
            Convt(i,1) = (Conv1(7,i-6) + Conv1(2,i-2) + Conv1(3,i-3) + Conv1(4,i-4) + Conv1(5,i-5)) / 5;
        elseif i == 1992
            Convt(i,1) = (Conv1(7,i-6) + Conv1(3,i-3) + Conv1(4,i-4) + Conv1(5,i-5)) / 4;
        elseif i == 1993
            Convt(i,1) = (Conv1(7,i-6) + Conv1(4,i-4) + Conv1(5,i-5)) / 3;
        elseif i == 1994
            Convt(i,1) = (Conv1(7,i-6) + Conv1(5,i-5)) / 2;
        elseif i == 1995
            Convt(i,1) =  Conv1(7,i-6);
        else
            Convt(i,1) = (Conv1(1,i) + Conv1(6,i-5) + Conv1(2,i-1) + Conv1(3,i-2) + Conv1(4,i-3) + Conv1(5,i-4) + Conv1(7,i-6)) / 7;
        end
    end
    Conv2 = mapminmax(Convt,0,1);
    for i = 1 : length(Conv2)
        for j = 1 : length(v2)
            B = Conv2(i,1);
            Conv(j,i) = B*v2(1,j);
        end
    end

    for i = 1 : length(Conv2)+5
        if i == 1
            last_un_Conv(i,1) = Conv(1,1);
        elseif i == 2
            last_un_Conv(i,1) = (Conv(1,2) + Conv(2,1)) / i;
        elseif i == 3
            last_un_Conv(i,1) = (Conv(1,3) + Conv(2,2) + Conv(3,1)) / i;
        elseif i == 4
            last_un_Conv(i,1) = (Conv(1,4) + Conv(4,1) + Conv(3,2) + Conv(2,1)) / i;
        elseif i == 5
            last_un_Conv(i,1) = (Conv(1,5) + Conv(5,1) + Conv(2,4) + Conv(4,2) + Conv(3,3)) / i;
%         elseif i == 6
%             last_un_Conv(i,1) = (Conv(1,6) + Conv(6,1) + Conv(2,5) + Conv(5,2) + Conv(3,4) + Conv(4,3)) / i;
%         elseif i == 7
%             last_un_Conv(i,1) = (Conv(1,7) + Conv(7,1) + Conv(2,6) + Conv(6,2) + Conv(3,5) + Conv(5,3) + Conv(4,4)) / i;
%         elseif i == 1994
%             last_un_Conv(i,1) = (Conv(8,i-7) + Conv(7,i-6) + Conv(6,i-5) + Conv(5,i-4) + Conv(4,i-3) + Conv(3,i-2) + Conv(2,i-1)) / 7;
%         elseif i == 1995
%             last_un_Conv(i,1) = (Conv(8,i-7) + Conv(7,i-6) + Conv(6,i-5) + Conv(5,i-4) + Conv(4,i-3) + Conv(3,i-2)) / 6;
        elseif i == 1996
            last_un_Conv(i,1) = (Conv(6,i-5) + Conv(5,i-4) + Conv(4,i-3) + Conv(3,i-2) + Conv(2,i-1)) / 5;
        elseif i == 1997
            last_un_Conv(i,1) = (Conv(6,i-5) + Conv(5,i-4) + Conv(4,i-3) + Conv(3,i-2)) / 4;
        elseif i == 1998
            last_un_Conv(i,1) = (Conv(6,i-5) + Conv(5,i-4) + Conv(4,i-3)) / 3;
        elseif i == 1999
            last_un_Conv(i,1) = (Conv(6,i-5) + Conv(5,i-4)) / 2;
        elseif i == 2000
            last_un_Conv(i,1) = (Conv(6,i-5))/1;
        else
            last_un_Conv(i,1) = (Conv(6,i-4) + Conv(5,i-3) + Conv(4,i-2) + Conv(3,i-1) + Conv(2,i-5) + Conv(1,i)) / 6;
        end
    end
    uns1{w,1} = last_un_Conv;
end
 

clear Convt last_un_Conv
%% 
v1 = [0.8,1,1.5,1.5,1.5,1,0.8];
v1 = v1/sum(v1,2);
v2 = [1.1,1.3,1.5,1.5,1.3,1.1];
v2 = v2/sum(v2,2);
for w = 1 : shuliang
    for i = 1 : 1989
        for j = 1 : length(v1)
            a = score_data(:,w);
            B = a(i,1);
            Conv1(j,i) = B*v1(1,j);
        end
    end
    Convt = [];
    last_un_Conv = [];

    for i = 1 : length(a)+6
        if i == 1
            Convt(i,1) = Conv1(1,1);
        elseif i == 2
            Convt(i,1) = (Conv1(1,2) + Conv1(2,1)) / i;
        elseif i == 3
            Convt(i,1) = (Conv1(1,3) + Conv1(2,2) + Conv1(3,1)) / i;
        elseif i ==4
            Convt(i,1) = (Conv1(1,4) + Conv1(4,1) + Conv1(3,2) + Conv1(2,1)) / i;
        elseif i == 5
            Convt(i,1) = (Conv1(1,5) + Conv1(5,1) + Conv1(2,4) + Conv1(4,2) + Conv1(3,3)) / i;
        elseif i == 6
            Convt(i,1) = (Conv1(1,6) + Conv1(6,1) + Conv1(2,5) + Conv1(5,2) + Conv1(3,4) + Conv1(4,3)) / i;
        elseif i == 1990
            Convt(i,1) = (Conv1(7,i-6) + Conv1(2,i-1) + Conv1(3,i-2) + Conv1(4,i-3) + Conv1(5,i-4) + Conv1(6,i-5)) / 6;
        elseif i == 1991
            Convt(i,1) = (Conv1(7,i-6) + Conv1(2,i-2) + Conv1(3,i-3) + Conv1(4,i-4) + Conv1(5,i-5)) / 5;
        elseif i == 1992
            Convt(i,1) = (Conv1(7,i-6) + Conv1(3,i-3) + Conv1(4,i-4) + Conv1(5,i-5)) / 4;
        elseif i == 1993
            Convt(i,1) = (Conv1(7,i-6) + Conv1(4,i-4) + Conv1(5,i-5)) / 3;
        elseif i == 1994
            Convt(i,1) = (Conv1(7,i-6) + Conv1(5,i-5)) / 2;
        elseif i == 1995
            Convt(i,1) =  Conv1(7,i-6);
        else
            Convt(i,1) = (Conv1(1,i) + Conv1(6,i-5) + Conv1(2,i-1) + Conv1(3,i-2) + Conv1(4,i-3) + Conv1(5,i-4) + Conv1(7,i-6)) / 7;
        end
    end
    Conv2 = mapminmax(Convt,0,1);
    for i = 1 : length(Conv2)
        for j = 1 : length(v2)
            B = Conv2(i,1);
            Conv(j,i) = B*v2(1,j);
        end
    end

    for i = 1 : length(Conv2)+5
        if i == 1
            last_un_Conv(i,1) = Conv(1,1);
        elseif i == 2
            last_un_Conv(i,1) = (Conv(1,2) + Conv(2,1)) / i;
        elseif i == 3
            last_un_Conv(i,1) = (Conv(1,3) + Conv(2,2) + Conv(3,1)) / i;
        elseif i == 4
            last_un_Conv(i,1) = (Conv(1,4) + Conv(4,1) + Conv(3,2) + Conv(2,1)) / i;
        elseif i == 5
            last_un_Conv(i,1) = (Conv(1,5) + Conv(5,1) + Conv(2,4) + Conv(4,2) + Conv(3,3)) / i;
%         elseif i == 6
%             last_un_Conv(i,1) = (Conv(1,6) + Conv(6,1) + Conv(2,5) + Conv(5,2) + Conv(3,4) + Conv(4,3)) / i;
%         elseif i == 7
%             last_un_Conv(i,1) = (Conv(1,7) + Conv(7,1) + Conv(2,6) + Conv(6,2) + Conv(3,5) + Conv(5,3) + Conv(4,4)) / i;
%         elseif i == 1994
%             last_un_Conv(i,1) = (Conv(8,i-7) + Conv(7,i-6) + Conv(6,i-5) + Conv(5,i-4) + Conv(4,i-3) + Conv(3,i-2) + Conv(2,i-1)) / 7;
%         elseif i == 1995
%             last_un_Conv(i,1) = (Conv(8,i-7) + Conv(7,i-6) + Conv(6,i-5) + Conv(5,i-4) + Conv(4,i-3) + Conv(3,i-2)) / 6;
        elseif i == 1996
            last_un_Conv(i,1) = (Conv(6,i-5) + Conv(5,i-4) + Conv(4,i-3) + Conv(3,i-2) + Conv(2,i-1)) / 5;
        elseif i == 1997
            last_un_Conv(i,1) = (Conv(6,i-5) + Conv(5,i-4) + Conv(4,i-3) + Conv(3,i-2)) / 4;
        elseif i == 1998
            last_un_Conv(i,1) = (Conv(6,i-5) + Conv(5,i-4) + Conv(4,i-3)) / 3;
        elseif i == 1999
            last_un_Conv(i,1) = (Conv(6,i-5) + Conv(5,i-4)) / 2;
        elseif i == 2000
            last_un_Conv(i,1) = (Conv(6,i-5))/1;
        else
            last_un_Conv(i,1) = (Conv(6,i-4) + Conv(5,i-3) + Conv(4,i-2) + Conv(3,i-1) + Conv(2,i-5) + Conv(1,i)) / 6;
        end
    end
    uns2{w,1} = last_un_Conv;
%     attention_score = mapminmax(last_un_Conv,0,1);
%     save_file = sprintf('attention_score_%d.npy',w);
%     %writeNPY(attention_score,save_file);
%     figure;
%     plot(attention_score);
end    
    
for i = 1 :shuliang
    C = uns1{i,1}(:,1) + uns2{i,1}(:,1);
    C = mapminmax(C,0,1);
    save_file = sprintf('attention_score_%d.npy',i);
    writeNPY(C,save_file);
end
        