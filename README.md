# Waterlinked Sonar3d driver

Please generate the protobuff python msg file before running the driver:

```bash
./generate_protobuf_file.sh
```

Then start the driver and enable accoustics:
```bash
ros2 run sonar3d_driver sonar3d_driver
ros2 service call /sonar3d/start std_srvs/srv/Trigger {}
```
