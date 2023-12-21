Run pipeline locally 

kedro run 

kedro viz

Run pipeline in Docker 

kedro docker init

kedro docker build

kedro docker run

kedro docker cmd --docker-args="-p=4141:4141" kedro viz --host=0.0.0.0

* docker does not contain data just pipeline so you will not see data 
