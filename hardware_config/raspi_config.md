# Raspi configuration

## Network credentials:
MAC: b8:27:eb:72:0f:55
Network: IoT-B19
password: *********

To know ip address:
`$ ifconfig`

To search raspi ip, use:
`$ sudo nmap -sP bradcast[3].1-254`

Connect using ssh:
`$ ssh pi@10.102.19.201 -l pi`
hint: connection isn't strong enough, so sometimes it is not working well


## Streaming:

option 1:
`$ cd flask-video-streaming/ && python app.py`
now we can take images from:
http://192.168.0.102:5000/video_feed

option 2:
`$ cd cd Sunfounder_Smart_Video_Car_Kit_for_RaspberryPi/mjpg-streamer/mjpg-streamer/ && sh start.sh`
now we can take images from:
http://192.168.0.102:8080/?action=stream

option 3:
source: https://www.instructables.com/id/How-to-Make-Raspberry-Pi-Webcam-Server-and-Stream-/
`$ sudo service motion restart`
`$ sudo motion`
now we can take images from:
http://192.168.0.102:8081
