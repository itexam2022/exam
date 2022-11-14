#include<stdio.h>


void menu()
{
	printf("\n\n\t\t*MAIN MENU*");
	printf("\n\n\tPAGE REPLACEMENT ALGORITHMS");
	printf("\n\n\t1.FIRST IN FIRST OUT");
	printf("\n\n\t2.LEAST RECENTLY USED");
	printf("\n\n\t3.OPTIMAL");
	printf("\n\n\t4.QUIT");
}

void accept(int ref[],int n)
{
	int i;
	printf("\nENTER ELEMENT OF REFERENCE STRING ONE BY ONE::");
	for(i=0;i<n;i++)
	{
		scanf("%d",&ref[i]);
	}
}

void display(int ref[],int n)
{
	int i;
	//printf("\n\n");
	for(i=0;i<n;i++)
	{
		printf("%3d  ",ref[i]);
	}
}

int search(int frame[],int x,int fs)
{
	int i;
	for(i=0;i<fs;i++)
	{
		if(frame[i]==x)
			return 1;
	}
	return 0;
}

void fifo(int ref[],int n)
{
	int frame[10];
	int i,k,fs,temp,cnt;
	printf("\n\nENTER FRAME SIZE::");
	scanf("%d",&temp);

	cnt=k=fs=0;

	for(i=0;i<n;i++)
	{
		if(search(frame,ref[i],fs)==0)
		{
			frame[k]=ref[i];
			k=(k+1)%temp;
			cnt++;
			if(cnt<temp)
				fs=cnt;
			else
				fs=temp;
			printf("\n\nFOR PAGE %d ::\t",ref[i]);
			display(frame,fs);
		}
		else
		{
			printf("\n\nFOR PAGE %d ::\t",ref[i]);
			printf("  NO PAGE FAULT");
		}
	}
	printf("\n\nTOTAL NO. OF PAGE FAULTS = %d ",cnt);
}


int main()
{
	int ref[50],n,ch;
	
	printf("\n\nENTER SIZE OF REFERENCE STRING::");
	scanf("%d",&n);
	accept(ref,n);

while(ch!=4)
	{
		printf("\n\nENTER YOUR CHOICE::");
		scanf("%d",&ch);
		switch(ch)
		{
			case 1:
				printf("\nYOUR ENTERED REFERENCE STRING IS::\n\n");
				display(ref,n);
				fifo(ref,n);
				break;

		
		}

	}
return(0);
}