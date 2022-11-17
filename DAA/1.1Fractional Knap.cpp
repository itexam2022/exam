#include<iostream>
#include<algorithm>
#include<bits/stdc++.h>
using namespace std;

typedef struct obejct
{
    int profit;
    int weight;

    obejct(int profit,int weight) 
    {
        this->profit=profit;
        this->weight=weight;
    }
}obj;

//Case-I:Greedy about profit
int cmp1(obj a,obj b) //Compare function to return max profit
{
    int temp1,temp2;
    temp1=a.profit;
    temp2=b.profit;
    return temp1>temp2;
}

float ks1(obj a[],int k_capacity,int n)
{
    int curr_weight=0; //Current weight in knapsack
    float total_profit=0.0; //Total profit
    int r_capacity; //Remaining capacity
    sort(a,a+n,cmp1); //Sort function to sort object on basis of maximum profit
    //Going through all objects

    for(int i=0;i<n;i++)
    {
    //Add that object completely
        if(curr_weight+a[i].weight<=k_capacity)
        {
            curr_weight=curr_weight+a[i].weight;
            total_profit=total_profit+a[i].profit;
        }
        else
        {
    //If we can't add current object add fractional part of it
            r_capacity=k_capacity-curr_weight; //calculating remaining capacity of knapsack
            total_profit=total_profit+(float)(r_capacity*a[i].profit)/a[i].weight;
        break;
        }
    }
    return total_profit; 
}

//Case-II:Greedy about weight
int cmp2(obj a,obj b) //Compare function to return min weight
{
    float temp1,temp2;
    temp1=a.weight;
    temp2=b.weight;
    return temp1<temp2;
}
float ks2(obj a[],int k_capacity,int n)
{
    int curr_weight=0; //Current weight in knapsack
    float total_profit=0.0; //Total profit
    int r_capacity; //Remaining capacity
    sort(a,a+n,cmp2); //Sort function to sort object on basis of minimum weight
    //Going through all objects
    for(int i=0;i<n;i++)
    {
    //Add object in knapsack if weight of given object is less than knapsack capacity
    //Add that object completely
        if(curr_weight+a[i].weight<=k_capacity)
        {
            curr_weight=curr_weight+a[i].weight;
            total_profit=total_profit+a[i].profit;
        }
        else
        {
    //If we can't add current object add fractional part of it
            r_capacity=k_capacity-curr_weight; //calculating remaining capacity of knapsack
            total_profit=total_profit+(float)(r_capacity*a[i].profit)/a[i].weight;
            break;
        }
    }
    return total_profit; //Returning total final profit
}

//Case-III:Greedy about Profit/Weight ratio
int cmp3(obj a,obj b) //Compare function to return max P/W ratio
{
    float temp1,temp2;
    temp1=(float)a.profit/a.weight;
    temp2=(float)b.profit/b.weight;
    return temp1>temp2;
}
float ks3(obj a[],int k_capacity,int n)
{
    int curr_weight=0; //Current weight in knapsack
    float total_profit=0.0; //Total profit
    float r_capacity; //Remaining capacity
    sort(a,a+n,cmp3); //Sort function to sort object on basis of maximum P/W ratio
    //Going through all objects
    for(int i=0;i<n;i++)
    {
        if(curr_weight+a[i].weight<=k_capacity)
        {
    //Add object in knapsack if weight of given object is less than knapsack capacity
    //Add that object completely
            curr_weight=curr_weight+a[i].weight;
            total_profit=total_profit+a[i].profit;
        }
        else
        {
    //If we can't add current object add fractional part of it
            r_capacity=k_capacity-curr_weight; //calculating remaining capacity of knapsack
            total_profit=total_profit+(float)(r_capacity*a[i].profit)/a[i].weight;
            break;
        }
    }
    return total_profit; //Returning total final profit
}

int main()
{
    int k_capacity=20; //capacity of knapsack
    obj a[]={{25,18},{24,15},{15,10}}; //Profit and weight values as pairs
    //To find the size of array
    int size=sizeof(a)/sizeof(a[0]);
    cout<<"Case-I:Greedy about profit: "<<endl;
    cout<<"Total Profit:"<<ks1(a,k_capacity,size)<<endl;
    cout<<"Case-II:Greedy about weight: "<<endl;
    cout<<"Total profit:"<<ks2(a,k_capacity,size)<<endl;
    cout<<"Case-III:Greedy about Profit/Weight: "<<endl;
    cout<<"Total profit:"<<ks3(a,k_capacity,size)<<endl;
    return 0;
}