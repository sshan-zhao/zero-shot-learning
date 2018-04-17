for file in ./*
do 
if test -f $file
then
	cp $file ../zero-shot-learning/
fi
done
