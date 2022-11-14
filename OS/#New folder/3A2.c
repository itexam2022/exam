#include <stdio.h>

int main ()
{
  int at[10], bt[10], rt[10], completionTime, i, smallest;
  int remain = 0, n, time, sum_wait = 0, sum_turnaround = 0;
  printf ("Enter no of Processes : ");
  scanf ("%d", &n);
  for (i = 0; i < n; i++)
    {
      printf ("Enter arrival time for Process P%d : ", i + 1);
      scanf ("%d", &at[i]);
      printf ("Enter burst time for Process P%d ", i + 1);
      scanf ("%d", &bt[i]);
      rt[i] = bt[i];
    }
  printf ("\n\nProcess\t|Turnaround Timel Waiting Time\n\n");
  for (time = 0; remain != n; time++)
    {
      for (i = 0; i < n; i++)
	{
	  if (at[i] <= time && rt[i])
	    {
	      smallest = i;
	    }
	}
      rt[smallest]--;
      if (rt[smallest] == 0)
	{
	  remain++;
	  completionTime = time + 1;
	  printf ("\nP[%d]\t]\t%d\t]\t%d", smallest + 1,completionTime - at[smallest],completionTime - bt[smallest] - at[smallest]);
	  sum_wait = sum_wait + completionTime - bt[smallest] - at[smallest];
	  sum_turnaround = sum_turnaround + completionTime - at[smallest];
	}
    }
  printf ("\n\n Average waiting time:= %f\n", sum_wait * 1.0 / n);
  printf ("\n\n Average waiting time:= %f\n", sum_turnaround * 1.0 / n);
}