#!/bin/bash
echo "Hello World !"

impala-shell -i production-hadoop-slave-3 -q \
"select lng, lat, 
	index_datetime time, 
	(sum(start_num) + sum(end_num))  as flow,
	sum(demand_num) as dm 
 from   ops.dm_bike_index_one_hour 
 where 	date_time between '20170612' and '20170709' and 
       	index_datetime < '2017-07-09 23:59:59' and 
	lng between '116.458' and '116.462' and
	lat between '39.948' and '39.952' 
 group by lng,lat,time 
 order by lng,lat,time" \
-B --print_header --output_delimiter="," -o ./Python/DemandPrediction/DMData/bj_s_train.csv

#bj
#lng between '116.170' and '116.610' and
#lat between '39.794' and '40.030' 

#cd
#lng between '103.818' and '104.280' and 
#lat between '30.554' and '30.859'

#sh
#lng between '121.300' and '121.703' and
#lat between '31.127' and '31.346' 

#gz
#lng between '113.179' and '113.473' and
#lat between '22.982' and '23.222' 

#sz
#lng between '113.800' and '114.176' and
#lat between '22.456' and '22.643' 



