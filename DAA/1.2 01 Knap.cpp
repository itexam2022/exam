#include<iostream>
#include<bits/stdc++.h>
using namespace std;


int max(int a,int b)
{
    return (a>b)?a:b;
}

void knapsack(int p[],int w[],int n,int k_capacity)
{
    int i,j,total_profit;
    int a[n+1][k_capacity+1]; 
    for(i=0;i<=n;i++) 
    {
        for(j=0;j<=k_capacity;j++)
            {
            if(i==0 || j==0)
            {
                a[i][j]=0; 
            }
            else if(w[i]<=j) 
            {
                a[i][j]=max(a[i-1][j],(a[i-1][j-w[i]]+p[i]));
            }
            else 
            {
                a[i][j]=a[i-1][j];
            }
        }
    }
    int profit=a[n][k_capacity];
    cout<<"Total profit: "<<profit<<endl;
    cout<<"Matrix generated for Dynamic Programming: "<<endl;
    for(i=0;i<=n;i++)
    {
        for(j=0;j<=k_capacity;j++)
        {
            cout<<a[i][j]<<"\t";
        }
        cout<<endl;
    }
    cout<<endl;
    
    int weight = k_capacity;
    for(i=n;i>0 && profit>0;i--)
    {
        if(a[i][weight]==a[i-1][weight])
        {
            cout<<"This item is not included "<<i<<" ->0"<<endl;
        }
        else
        {
            cout<<"This item is included"<<i<<" ->1"<<endl;
            weight = weight - w[i];
        }
    }
}

int main()
{
    int k_capacity = 8;

    int w[5] = {0,2,3,4,5};
    int p[5] = {0,1,2,5,6};

    int n = 4;
    knapsack(p,w,n,k_capacity); 
    return 0;
}