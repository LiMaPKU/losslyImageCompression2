def binSearch(L, R):
    M = (L + R)//2
    if(M==L or M==R):
        return 0
    print('[',L,R,']','区间长度',R-L+1)
    binSearch(L, M)
    binSearch(M+1, R)

binSearch(0, 255)