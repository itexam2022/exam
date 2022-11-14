echo "entere file name"
read fname

ch=1
while [ $ch -ne 6 ];do
echo "menue"
echo 1 Create record
echo 2 view record
echo 3 Delete record
echo 4 search
echo 5 modify
echo 6 exit
echo "entere ur choice"
read ch

case $ch in 
1)
echo entere name hno city
read name hno city
echo $name $hno $city >> $fname
echo record created
;;
2)
echo records in file $file are:
cat $fname
;;
3)
echo entere name to Delete
read name
if grep $name $fname
then 
grep -v $name $fname >> temp
rm $fname
mv temp $fname
echo deleted
else
echo record not found
fi
;;
4)
echo entere name to Delete
read name
if grep $name $fname
then 
echo found
else
echo record not found
fi
;;
5)
echo name want to modify
read name
if grep $name $fname
then 
echo entere name hno city
read name hno city
echo $name1 $hno2 $city2 >> temp
grep -v $name $fname >> temp
rm $fname
mv temp $fname
echo record modifies
else
echo not found
fi
;;
esac
done
