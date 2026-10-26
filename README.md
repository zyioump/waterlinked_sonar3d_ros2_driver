# Waterlinked Sonar3d driver

The driver uses Water Linked's [wlsonar](https://github.com/waterlinked/wlsonar) library,
which is not available through rosdep. Install it with pip before running the driver:

```bash
pip install wlsonar
```

wlsonar requires protobuf >= 6.33.2, which is newer than the version shipped with
Ubuntu/ROS, so pip will install a newer protobuf alongside it.

Then start the driver and enable accoustics:
```bash
ros2 run sonar3d_driver sonar3d_driver
ros2 service call /sonar3d/start std_srvs/srv/Trigger {}
```
