#include <stdio.h>


void main()
{
    int at [] = {0, 1, 2, 3};
    int bt [] = {8, 5 , 10, 11};
    int temp[10];
    int i, sum=0,count=0, y, wt=0, tat=0;
    int NOP = 4;
    int quant = 6;
    y = NOP; 
  
    for(i=0; i<NOP; i++)
    {
        temp[i] = bt[i]; 
    }
    
    for(sum=0, i = 0; y!=0; )
    {
        if(temp[i] <= quant && temp[i] > 0) 
        {
            sum = sum + temp[i];
            temp[i] = 0;
            count=1;
        }
        else if(temp[i] > 0)
        {
            temp[i] = temp[i] - quant;
            sum = sum + quant;
        }
        if(temp[i]==0 && count==1)
        {
            y--; 
            wt = wt+sum-at[i]-bt[i];
            tat = tat+sum-at[i];
            count =0;
        }
        
        if(i==NOP-1)
        {
            i=0;
        }
        else if(at[i+1]<=sum)
        {
            i++;
        }
        else
        {
            i=0;
        }
    }
    float avg_wt=(float)wt/NOP;
    float avg_tat=(float)tat/NOP;
    printf("\n Average Waiting Time is %lf",avg_wt);
    printf("\n Average turnaround time is %lf",avg_tat);
}