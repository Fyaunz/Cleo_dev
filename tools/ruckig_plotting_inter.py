import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ruckig import InputParameter, OutputParameter, Ruckig, Result

def generate_trajectory(yaw, pitch, roll, v_max, a_max, j_max):
    CENTER = 180.0
    target_m1 = CENTER + pitch + roll
    target_m2 = CENTER + pitch - roll
    target_m3 = CENTER + yaw

    control_rate = 0.02
    otg = Ruckig(3, control_rate)
    inp = InputParameter(3)
    out = OutputParameter(3)

    inp.max_velocity = [v_max, v_max, v_max]
    inp.max_acceleration = [a_max, a_max, a_max]
    inp.max_jerk = [j_max, j_max, j_max]

    inp.current_position = [CENTER, CENTER, CENTER]
    inp.current_velocity = [0.0, 0.0, 0.0]
    inp.current_acceleration = [0.0, 0.0, 0.0]
    
    inp.target_position = [target_m1, target_m2, target_m3]
    inp.target_velocity = [0.0, 0.0, 0.0]
    inp.target_acceleration = [0.0, 0.0, 0.0]

    time_log = []
    pos_log = [[], [], []]
    vel_log = [[], [], []]
    acc_log = [[], [], []]
    jerk_log = [[], [], []]

    current_time = 0.0

    while otg.update(inp, out) == Result.Working:
        time_log.append(current_time)
        for i in range(3):
            pos_log[i].append(out.new_position[i])
            vel_log[i].append(out.new_velocity[i])
            acc_log[i].append(out.new_acceleration[i])
            jerk_log[i].append(out.new_jerk[i])

        out.pass_to_input(inp)
        current_time += control_rate

    return time_log, pos_log, vel_log, acc_log, jerk_log

# Main Plotting Window ---
fig_plots, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
plots_manager = getattr(fig_plots.canvas, 'manager', None)
if plots_manager is not None:
    plots_manager.set_window_title('Trajectory Viewer')
fig_plots.suptitle('Ruckig OTG Kinematic Profiles', fontsize=16)
plt.subplots_adjust(hspace=0.3)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
labels = ['M1 (Left)', 'M2 (Right)', 'M3 (Base)']

init_vals = {'yaw': 60, 'pitch': 30, 'roll': -15, 'v': 180, 'a': 360, 'j': 1000}
t, p, v, a, j = generate_trajectory(*init_vals.values())

lines_p = [axs[0].plot(t, p[i], color=colors[i], label=labels[i], lw=2)[0] for i in range(3)]
lines_v = [axs[1].plot(t, v[i], color=colors[i], lw=2)[0] for i in range(3)]
lines_a = [axs[2].plot(t, a[i], color=colors[i], lw=2)[0] for i in range(3)]
lines_j = [axs[3].step(t, j[i], color=colors[i], lw=2, where='post')[0] for i in range(3)]

axs[0].set_ylabel('Pos (deg)')
axs[0].legend(loc="upper right", fontsize='small')
axs[1].set_ylabel('Vel (deg/s)')
axs[2].set_ylabel('Acc (deg/s²)')
axs[3].set_ylabel('Jerk (deg/s³)')
axs[3].set_xlabel('Time (seconds)')

for ax in axs:
    ax.grid(True)

# Control Window
fig_ctrl = plt.figure(figsize=(5, 6))
ctrl_manager = getattr(fig_ctrl.canvas, 'manager', None)
if ctrl_manager is not None:
    ctrl_manager.set_window_title('Kinematics Remote Control')
fig_ctrl.suptitle('Adjust Parameters', fontsize=14)

# [left, bottom, width, height] for the slider axes in the new window
ax_yaw   = fig_ctrl.add_axes([0.3, 0.80, 0.55, 0.05])
ax_pitch = fig_ctrl.add_axes([0.3, 0.68, 0.55, 0.05])
ax_roll  = fig_ctrl.add_axes([0.3, 0.56, 0.55, 0.05])
ax_v     = fig_ctrl.add_axes([0.3, 0.35, 0.55, 0.05])
ax_a     = fig_ctrl.add_axes([0.3, 0.23, 0.55, 0.05])
ax_j     = fig_ctrl.add_axes([0.3, 0.11, 0.55, 0.05])

s_yaw   = Slider(ax_yaw, 'Target Yaw', -90.0, 90.0, valinit=init_vals['yaw'])
s_pitch = Slider(ax_pitch, 'Target Pitch', -45.0, 45.0, valinit=init_vals['pitch'])
s_roll  = Slider(ax_roll, 'Target Roll', -45.0, 45.0, valinit=init_vals['roll'])
s_v     = Slider(ax_v, 'Max Velocity', 1.0, 1000.0, valinit=init_vals['v'])
s_a     = Slider(ax_a, 'Max Accel', 1.0, 5000.0, valinit=init_vals['a'])
s_j     = Slider(ax_j, 'Max Jerk', 1.0, 20000.0, valinit=init_vals['j'])

# Loop Update
def update(val):
    t_new, p_new, v_new, a_new, j_new = generate_trajectory(
        s_yaw.val, s_pitch.val, s_roll.val, s_v.val, s_a.val, s_j.val
    )
    
    for i in range(3):
        lines_p[i].set_data(t_new, p_new[i])
        lines_v[i].set_data(t_new, v_new[i])
        lines_a[i].set_data(t_new, a_new[i])
        lines_j[i].set_data(t_new, j_new[i])
        
    max_t = max(t_new) if t_new else 1.0
    for ax in axs:
        ax.set_xlim(0, max_t * 1.05) 
        ax.relim()
        ax.autoscale_view()
        
    fig_plots.canvas.draw_idle()

# Bind the update function
s_yaw.on_changed(update)
s_pitch.on_changed(update)
s_roll.on_changed(update)
s_v.on_changed(update)
s_a.on_changed(update)
s_j.on_changed(update)

plt.show()