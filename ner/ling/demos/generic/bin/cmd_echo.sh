CLASS=com.aliasi.demo.demos.EchoDemo
ARGS=

CMD=com.aliasi.demo.framework.DemoCommand

CP=../../:../lingpipe-demos.jar:../../../lingpipe-4.1.2.jar:../../lib/nekohtml-1.9.14.jar:../../lib/xercesImpl-2.9.1.jar:../../lib/xml-apis-2.9.1.jar
java -cp $CP $CMD -demoConstructor=$CLASS -demoConstructorArgs=$ARGS $1 $2 $3 $4 $5 $6 $7 $8 $9

