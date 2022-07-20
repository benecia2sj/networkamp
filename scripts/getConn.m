% getConn - compute absolute, negative, positive, mean functional
% connectivity (FC) from netmat
%
% conn = getConn(netmat,type);
%
% Input
% netmat: between-network FC (subject x nc)
%         each row of netmat is a vectorized upper triangular part of netmat for a subject
%         (nc in this study is 21*20/2 = 210)
%
% type: a string determining which computation type to use ('mean','abs','neg','pos')
%
% Output
% conn: absolute, negative, positive or mean FC  (subject x 21)
%
% % Author: Soojin Lee, 2022


function conn = getConn(netmat, type)

nsubj = size(netmat,1);
nnode = 21;  % number of network/node
conn = [];

if strcmp(type, 'mean')
    for s = 1:nsubj
        A = netmat(s,:)';
        B = triu(ones(nnode),1);
        B(B==1) = A;
        C = B+B'; % symmetric netmat
        % average
        conn(s,:) = mean(C)*nnode/(nnode-1);
    end
elseif strcmp(type, 'abs')
    for s = 1:nsubj
        A = netmat(s,:)';
        B = triu(ones(nnode),1);
        B(B==1) = A;
        C = B+B'; % symmetric netmat
        % absolute
        conn(s,:) = mean(abs(C))*nnode/(nnode-1); % remove one entry for 0
    end
elseif strcmp(type, 'neg')
    for s = 1:nsubj
        A = netmat(s,:)';
        B = triu(ones(nnode),1);
        B(B==1) = A;
        C = B+B'; % symmetric netmat
        % negative
        negind = C < 0;
        tmp = sum(C.*negind)./sum(negind);
        tmp(isnan(tmp)) = 0;
        conn(s,:) = tmp;
    end
elseif strcmp(type, 'pos')
    for s = 1:nsubj
        A = netmat(s,:)';
        B = triu(ones(nnode),1);
        B(B==1) = A;
        C = B+B'; % symmetric netmat
        % positive
        posind = C > 0;
        tmp = sum(C.*posind)./sum(posind);
        tmp(isnan(tmp)) = 0;
        conn(s,:) = tmp;
    end
end