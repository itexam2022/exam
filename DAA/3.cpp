#include<iostream>
#include<bits/stdc++.h>
using namespace std;
 
bool isSafe(int** board,int x,int y,int n) //x is for row, y is for column and n is number of queens
{                                          //int** is double pointer in stack 
    //Checking if there is any safe position in columns else return false
    for(int row=0;row<x;row++)
    {
        if(board[row][y]==1)
        {
            return false;
        }
    }

    int row=x; //iterators
    int col=y;

    //Checking if there is any safe position in left upper diagonal else return false
    while(row>=0 && col>=0)
    {
        if(board[row][col]==1)
        {
            return false;
        }
        row--;
        col--; 
    }

    row=x;
    col=y;

    //Checking if there is any safe position in right upper diagonal else return false
    while(row>=0 && col<n)
    {
        if(board[row][col]==1)
        {
            return false;
        }
        row--;
        col++; 
    }
    return true;   //Finally return true if positions are safe
}

bool nQueen(int** board,int x,int n)
{
    if(x>=n)
    {
        return true;      //n queens are placed
    }
    //we are checking our safe positions for columns
    for(int col=0;col<n;col++)   //new col variable
    {
        if(isSafe(board,x,col,n))   //If isSafe() returning true place queen
        {
            board[x][col]=1;        //Queen is placed on board

            if(nQueen(board,x+1,n)) //Recursive call
            {
                return true;
            }
            board[x][col]=0;    //Backtracking
        }
    }
    return false;
}


int main()
{
    int n;
    cout<<"Enter Number of queens: "<<endl; 
    cin>>n;

    int** board=new int*[n];      //Board of n*n

    for(int i=0;i<n;i++)
    {
        board[i]=new int[n];
        for(int j=0;j<n;j++)
        {
            board[i][j]=0;    //Initialise board with 0
        }
    }

    //Calling nQueen() function
    if(nQueen(board,0,n))
    {
        cout<<"Solution for N-Queen: "<<endl;
        cout<<"1=Queen is place and 0=Box is empty "<<endl<<endl;
        for(int i=0;i<n;i++)     //Printing board(n*n)
        {
            for(int j=0;j<n;j++)
            {
                cout<<board[i][j]<<" ";   //If 1 is present that means
            }                             //there is queen placed
            cout<<endl;
        }
    }
    return 0;
}       
