#include<iostream>
#include<bits/stdc++.h>
using namespace std;


int max(int a,int b)
{
    return (a>b)?a:b;
}

int main()
{

    int wt[5] = {0,2,3,4,5};
    int p[5] = {0,1,2,5,6};
    int n = 4;
    int m = 8;
    int i,w;
    int k[5][9];

    for(i = 0; i<=n;i++){
        for(w = 0;w<=m;w++){
            if(i==0 || w==0){
                k[i][w]=0;
            }
            else if(wt[i]<=w){
                k[i][w]=max(p[i]+k[i-1][w-wt[i]], k[i-1][w]);
            }
            else{
                k[i][w]=k[i-1][w];
            }
        }
    }
    int profit = k[n][m];
    cout<<"Profit is :"<<profit<<endl<<"\n";
    for(i = 0; i<=n;i++){
        for(w = 0;w<=m;w++){
            cout<<k[i][w]<<"\t";
        }
        cout<<endl;
    }
    cout<<endl;

    int weight = m;
    for(i=n;i>0 && profit>0;i--)
    {
        if(k[i][weight]==k[i-1][weight])
        {
            cout<<"This item is not included "<<i<<" ->0"<<endl;
        }
        else
        {
            cout<<"This item is included"<<i<<" ->1"<<endl;
            weight = weight - wt[i];
        }
    }
    return 0;
}