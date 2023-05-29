#!/bin/bash
# to measure average power consumed in 30sec with 1sec sampling interval
duration=3600
interval=0.1

RAILS=("VDD_IN /sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power0_input"
 "VDD_SYS_GPU /sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input"
 "VDD_SYS_CPU /sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power1_input"
 "VDD_SYS_SOC /sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power1_input")

for ((i = 0; i < ${#RAILS[@]}; i++)); do
 read name[$i] node[$i] pwr_sum[$i] pwr_count[$i] <<<$(echo "${RAILS[$i]} 0 0")
done
end_time=$(($(date '+%s') + duration))
while [ $(date '+%s') -le $end_time ]; do
 for ((i = 0; i < ${#RAILS[@]}; i++)); do
 pwr_sum[$i]=$((${pwr_sum[$i]} + $(cat ${node[$i]}))) &&
 pwr_count[$i]=$((${pwr_count[$i]} + 1))
 pwr_avg=$((${pwr_sum[$i]} / ${pwr_count[$i]}))
 echo "${name[$i]},$pwr_avg" >> power.txt
 done
 sleep $interval
done
