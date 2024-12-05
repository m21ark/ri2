# Extras that might help with bugs

### Sections 4, 5 and 10 are probably the most important for us

It seems that the **rcssserver3d** sometimes gets bugged, and even if you stop it, the next time you run it, it will just display the black screen. To fix this, you need to kill the process that is still running in the background.

```bash
ps aux | grep rcssserver3d # or sudo netstat -plten 

kill -9 <pid>
```

The **PlayOn** is important, since otherwise the ball won't move from its initial position, even by script.

The default arguments that can be passed through cmdline of the Script are recorded in the **config.json file**. You can change them.

For some reason I can't **change the team names**. I think this might be decided at startup when the server is started, so if you try to change this after the server is running, it won't work, I think.

The `player.behavior.execute()` command updates the internal array w.robot.joints_target_speed with target values for each joint. And these:
```
player.scom.commit_and_send( w.robot.get_command() )
player.scom.receive()
```

just commit the changes, send them to the server, and wait for a response.



