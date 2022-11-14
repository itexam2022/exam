echo "Enter the file name: "
read fname
echo

ch=1
while [ $ch -ne 6 ]; do
echo " M E N U"
echo "1. Create a new record."
echo "2. View all records."
echo "3. Delete a record."
echo "4. Search a record."
echo "5. Modify a record."
echo "6. Exit."
echo "Enter your choice: "
read ch

case $ch in
1)
echo "Enter Name Huse no. City Pincode "
read name hno city pin
echo $name $hno $city $pin >> $fname
echo "Record created successfully."
;;

2)
echo "Records in the file '$fname' are: "
cat $fname
;;

3)
echo "Enter name of the person you want to remove: "
read name
if grep $name $fname
then
grep -v $name $fname>> temp
rm $fname
mv temp $fname
echo "Record has been deleted."
else
echo "Record not found."
fi
;;

4)
echo "Enter name of the person you want to search: "
read name
if grep $name $fname
then
echo "Record found."
else
echo "Record not found."
fi
;;

5)
echo "Enter name of the person you want to modify: "
read name
if grep $name $fname
then
echo "Enter Name Huse no. City Pincode "
read name1 hno1 city1 pin1
echo $name1 $hno1 $city1 $pin1 >> temp
grep -v $name $fname>> temp
rm $fname
mv temp $fname
echo "Record modified successfully."
else
echo "Record not found."
fi
;;
esac

done