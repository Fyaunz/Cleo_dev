# Cleo — A Human-Robot-Interaction Demonstrator

Cleo is a 3-DOF robotic head — yaw plus a differential pitch/roll linkage — driven by Dynamixel
XL330 servos, plus two ears with one servo each. A daemon on a Raspberry Pi owns the hardware; a
client SDK on another machine commands it over ZMQ and receives telemetry, video and audio in return.

- **`cleosdk`** — the client: a `RobotHead` object, plus demos for movement, tracking, video,
  animation playback and speech.
- **`pi_controller`** — the daemon: servo HAL, Ruckig trajectory generation, animation playback, and
  the ZMQ sockets that expose them.
- **Full-duplex audio** through a USB array such as the ReSpeaker: the head's microphone streams to
  the client, the client's speech (TTS or any 16-bit PCM) plays through the head's speaker, and the
  array's direction of arrival is available as a query.

## Repository layout

| Path | Contents |
| --- | --- |
| `cleosdk/` | The client distribution — `cleo.py` and `demos/` |
| `pi_controller/` | The daemon: `main.py`, `hal/` (servos, solver, Ruckig, playback), `network/` (ZMQ sockets and routing) |
| `pi_controller/tests/` | Unit tests, hardware and network mocked |
| `e2e/` | End-to-end tests: real SDK against the daemon's real network stack over loopback |
| `deploy/` | systemd unit, udev rule and `install.sh` for running on the Pi at boot |
| `tools/` | Blender animation export, Ruckig plotting |

The repository holds two distributions. `cleosdk/` has its own `pyproject.toml` and a thin
dependency set (`pyzmq`, `numpy`, `scipy`, `opencv-python`) so it can be installed on a client
machine without the Pi-side servo, trajectory and audio stack. The root `pyproject.toml` is the
development environment for the whole repo. Both require **Python >= 3.13**.

## Installing the SDK

To drive the robot from another machine you only need `cleosdk`:

```bash
# from a local checkout
pip install ./cleosdk

# from git, over SSH (the repository is private for now)
pip install "git+ssh://git@github.com/Fyaunz/Cleo.git#subdirectory=cleosdk"
```

Or build a wheel and copy it to the machine that will drive the robot:

```bash
cd cleosdk
uv build            # or: python -m build
pip install dist/*.whl
```

Once installed the SDK imports from anywhere, with no `PYTHONPATH` juggling:

```python
from cleosdk import RobotHead, RobotError, CommandTimeout

head = RobotHead(ip="192.168.1.50", verbose=True)  # verbose logs each 'ok' reply
try:
    head.move_head(yaw=30, pitch=0, roll=0)
except RobotError as e:      # the robot refused the command, or the hardware faulted
    print(e)
except CommandTimeout as e:  # the robot never answered, even after retries
    print(e)
finally:
    head.disconnect()
```

Both exceptions derive from `CleoError`, so catch that to handle either at once. A reply arriving is
not proof the command worked — failures come back as an error status and are raised, never silently
swallowed.

The demos connect to `localhost`; to target a remote Pi, edit the `RobotHead(ip=...)` line in the
demo, or pass `--ip <PI_IP>` to `demo_audio`, the only one that takes it on the command line.

```bash
python -m cleosdk.demos.demo_move       # movement profiles
python -m cleosdk.demos.demo_ears       # ear movement
python -m cleosdk.demos.demo_track      # camera tracking
python -m cleosdk.demos.demo_video      # video stream
python -m cleosdk.demos.demo_animation  # baked Blender animation
python -m cleosdk.demos.demo_audio      # speech in and out
```

## Running the daemon

Point the daemon at your serial adapter — `CLEO_SERIAL` overrides the default in `main.py`, which is
how the same checkout runs on a development machine and on the Pi:

```bash
CLEO_SERIAL=/dev/ttyUSB0 python pi_controller/main.py
```

`servo_ids` is hardcoded in [pi_controller/main.py](pi_controller/main.py) and must match your
hardware. Its order matters: `[m1, m2, m3, ear_left, ear_right]`, the differential pair, then yaw,
then the ears. A list of three is still valid; the ear commands then refuse with an error.

### ZMQ endpoints

| Port | Pattern | Purpose |
| --- | --- | --- |
| 5555 | REQ/REP | Commands |
| 5556 | PUB/SUB | Telemetry, 20 Hz |
| 5557 | PUB/SUB | Video |
| 5558 | PUB/SUB | Microphone |
| 5559 | PUSH/PULL | Speaker (the only socket pointing *at* the Pi) |

Commands and telemetry run unconditionally. **Video and audio are opt-in**: their sockets are bound
at boot but stay silent until a client calls `start_video()` or `start_audio()`, so the camera and
the microphone are not held open by a daemon nobody is talking to. `stop_video()`, `stop_audio()`
and `disconnect()` release them again.

### Video

The daemon reaches the camera two ways and picks one when `start_video()` arrives:

| backend | how | where |
| --- | --- | --- |
| `rpicam` | `rpicam-vid --codec mjpeg` in a subprocess, JPEGs read off its stdout | the Pi's CSI Camera Module |
| `cv2` | `cv2.VideoCapture` plus `cv2.imencode` | USB webcams, and macOS during development |

`auto` (the default) takes `rpicam` if `rpicam-vid` or `libcamera-vid` is on `PATH` and falls back to
`cv2`. `CLEO_CAMERA=rpicam|cv2` pins it — useful on a Pi with a USB camera, where the CSI tools are
installed but not what you want. `CLEO_RPICAM_BIN` overrides where the binary is found.

The Camera Module sits in the head with its ribbon connector facing up, so the `rpicam` source turns
every frame **180°** and clients receive a picture that is already the right way up. That correction
is a property of the mount, so it does not apply to `cv2`, whose usual camera is a development
laptop's webcam. `CLEO_CAMERA_ROTATION=0|180` overrides either — `180` for a USB camera in the head
mount, `0` if the module is ever remounted the other way up. On the Pi the rotation costs nothing:
the ISP applies it before the frame is encoded.

The subprocess exists because the Camera Module is owned by libcamera, and picamera2 needs the
libcamera Python bindings — distro packages built against the distro's CPython and glibc, not
importable from the Nix-built interpreter devenv provides, and not carried by the PyPI package
either. A pipe has no ABI, so one environment covers the Pi and the dev machine.

### Audio

`start_audio()` opens both directions; each is optional, so `start_audio(mic=False)` gives a head
that only talks.

```python
head.start_audio()

# Out: any 16-bit PCM. The Pi resamples to whatever rate it opened the device at,
# so a TTS engine's native rate goes straight through.
head.play_wav("greeting.wav", wait=True)
head.flush_audio()               # barge-in: drop what is still queued

# In: blocks of captured audio, oldest first, each carrying its own format.
chunk = head.get_audio_chunk(timeout_ms=200)
if chunk:
    recognizer.AcceptWaveform(chunk.pcm)   # or chunk.samples for a numpy array
```

Unlike video, `get_audio_chunk()` does not skip to the newest message: only the latest camera frame
is worth having, but speech *is* the sequence, so every block comes back in order and the caller is
expected to keep up.

`CLEO_AUDIO_DEVICE` picks a device by name fragment or index when a machine has several; by default
the daemon takes the ReSpeaker if it is plugged in and the system default otherwise. To see what the
Pi can actually see:

```bash
python -c "import sys; sys.path.insert(0, 'pi_controller'); \
           from network.audio import list_devices; list_devices()"
```

### Direction of arrival

A ReSpeaker also reports which way it heard a voice from. The angle does not travel in the audio —
the array's DSP computes it and the Pi reads it over a vendor USB interface — so it is available
both as a query and alongside each block of audio:

```python
d = head.get_direction()          # answers even with the microphone closed
if d and d.speech:
    print(f"someone is talking at {d.angle} deg")

chunk = head.get_audio_chunk()    # or riding along with the audio it describes
chunk.direction                   # Direction(angle=..., speech=...) or None
```

**Always check `.speech` before acting on `.angle`.** The board latches the last direction it heard a
voice from and holds it through every silence, so an angle on its own is a memory rather than an
observation — a head that follows it blindly will keep staring at someone who left. `demo_audio.py
--follow` shows the intended shape: move only while `speech` is true, and rate-limit the moves.

`angle` is in the array's frame, not the head's: 0–180° across the arc it can distinguish, centred on
whichever way the board is physically mounted. `--offset` in the demo is where you correct for that,
measured once by talking from straight ahead.

Direction needs `pyusb` and permission to reach the USB device. Without either, everything above
returns `None` and audio is unaffected — on a Pi that usually means adding a udev rule or putting the
daemon's user in the right group.

## Running on the Pi as a service

`deploy/` contains everything needed to start the daemon at boot under systemd, so the Pi can be
powered on headless. From the checkout on the Pi, after `uv sync`:

```bash
sudo deploy/install.sh
```

The script renders [deploy/cleo.service](deploy/cleo.service) with this checkout's path, user and
interpreter, installs [deploy/99-cleo-servos.rules](deploy/99-cleo-servos.rules) with your adapter's
USB serial number, and enables the unit for boot. It deliberately does **not** start the service —
that energises the servos, so it stays a separate step:

```bash
sudo systemctl start cleo     # start now
systemctl status cleo         # check it came up
journalctl -u cleo -f         # follow the log
sudo systemctl stop cleo      # torque off, port closed
```

Three details in the unit and the udev rule matter more than they look:

- **`Restart=always` with `RestartSec=5`.** `ServoController` raises if the port will not open, so a
  missing or slow-to-enumerate adapter would otherwise leave the Pi with no daemon after a reboot.
- **`-u` on the interpreter.** journald hands the process a pipe rather than a tty, so Python
  block-buffers stdout and every `print()` — including hardware faults — would sit in the buffer
  instead of reaching the journal.
- **The udev rule.** `/dev/ttyUSB*` numbering follows enumeration order and can move between
  reboots. The rule pins the adapter to `/dev/cleo-servos`, which the unit passes as `CLEO_SERIAL`
  and orders its startup against.

`main.py` handles SIGTERM, which is how systemd stops a service, so `systemctl stop` and reboots
unwind through the normal shutdown path: servo torque disabled, serial port closed, camera and audio
device released.

To run the service as a different user, or from a different checkout, re-run `install.sh` from there
— or copy the unit to `/etc/systemd/system/cleo.service` and edit the `@...@` placeholders by hand.

## Tests

All three suites use the standard library `unittest`, so no extra packages are needed. Run them from the
project root inside the `devenv` shell.

**Unit tests** — fast, with the hardware and the network mocked out:

```bash
python -m unittest discover -s pi_controller/tests -t pi_controller
```

**End-to-end tests** — the real SDK against the real command listener and router over ZMQ on
loopback, with only the servos and Ruckig mocked. They cover what unit tests cannot: the REQ socket
reset, command resends, and the Pi's reply cache. Expect roughly 8 seconds, because they spend real
time waiting for network timeouts to fire.

```bash
python e2e/test_reliability.py

python e2e/test_reliability.py -v              # show each test as it runs
python e2e/test_reliability.py TestLostReply   # one class
python e2e/test_reliability.py -k verbose      # by substring
```

The audio path has its own end-to-end suite, with only PortAudio faked. It covers the two-part
`[header, pcm]` framing in both directions and the Pi's resampling of whatever rate a client sends:

```bash
python e2e/test_audio.py
```

Both e2e suites build their own copy of the daemon's network side on loopback ports counting up from
5700 (audio from 5800), so they neither need nor disturb a daemon already running on the default
ports.
