#include <iostream>
#include <algorith>
#include <bits/stdc++.h>
using namespace std;

typedef struct Object{

    int profit;
    int weight;

    Object(int profit, int weight){
        this->profit=profit;
        this->weight=weight;
    }
}obj;

int compareProfit(obj a, obj b){
    int temp1;
    int temp2;
    temp1 = a.profit;
    temp2 = b.profit;

    return temp1>temp2;
}

float knapsack1 (obj a[], int k_capacity, int n){



}

int main (){
    int k_capacity = 20;
    obj a[] ={{25, 18}, {25,8},{27,6}}

    size = sizeof(a)/sizeof(a[0]);

    knapsack1(a, k_capacity, size)
}