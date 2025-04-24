
echo "cleaning"
make clean || exit 1
binary=`find . -maxdepth 1 -name "stencil" | wc -l`
if [ 0 != "$binary" ]; then
	echo "make clean does not work as expected!"
	exit 1
fi

echo "building"
make || exit 1
binary=`find . -maxdepth 1 -name "stencil" | wc -l`
if [ 1 != "$binary" ]; then
	echo "stencil not built by 'make all'!"
	exit 1
fi
echo "build complete"

echo "Checking result of serial runs"
applications=( 1 )
for appl in ${applications[@]}; do
	./stencil input_files/input96.txt output_files/test_output96.txt $appl > /dev/null
	diff --ignore-all-space output_files/test_output96.txt refference_files/output96_${appl}_ref.txt
	if [ 0 != $? ] ; then
		echo "Wrong result of $appl application(s) on 96 elements!"
		exit 1
	fi
	#rm output_files/test_output96.txt
done
echo "OK"


echo "Checking result of parallel runs"
pe=( 4 )
for p in ${pe[@]}; do
	output_lines=`mpirun --bind-to none -np $p ./stencil input_files/input96.txt output_files/test_output96.txt 4 | wc -l`
	if [ 1 -lt $output_lines ]; then
		echo "Your program doesn't seem to be parallelized!"
		exit 1
	fi
	diff --ignore-all-space output_files/test_output96.txt refference_files/output96_4_ref.txt
	if [ 0 != $? ]; then
		echo "Wrong results of parallel run ($p processes)!"
		exit 1
	fi
	#rm output_files/test_output96.txt
done
echo "OK"

make clean || exit 1
#echo "Your file is ready for submission. Well done!