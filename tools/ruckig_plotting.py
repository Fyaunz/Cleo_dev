import numpy as np
import matplotlib.pyplot as plt
from ruckig import InputParameter, OutputParameter, Ruckig, Result

def plot_3dof_trajectory(yaw, pitch, roll, v_max, a_max, j_max):
    CENTER = 180.0
    
    target_m1 = CENTER + pitch + roll  # Left Motor
    target_m2 = CENTER + pitch - roll  # Right Motor
    target_m3 = CENTER + yaw           # Pan

    control_rate = 0.02 # 50Hz
    otg = Ruckig(3, control_rate)
    inp = InputParameter(3)
    out = OutputParameter(3)

    inp.max_velocity = [v_max, v_max, v_max]
    inp.max_acceleration = [a_max, a_max, a_max]
    inp.max_jerk = [j_max, j_max, j_max]

    # Start and Target Positions
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

    print(f"🚀 Generating 3-DOF Trajectory with Jerk Profiling...")

    # Trajectory Loop
    while otg.update(inp, out) == Result.Working:
        time_log.append(current_time)
        
        for i in range(3):
            pos_log[i].append(out.new_position[i])
            vel_log[i].append(out.new_velocity[i])
            acc_log[i].append(out.new_acceleration[i])
            jerk_log[i].append(out.new_jerk[i]) # <-- Log the Jerk

        out.pass_to_input(inp)
        current_time += control_rate

    # Plots
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fig.suptitle('3-DOF Differential Head Trajectories (Pos/Vel/Acc/Jerk)', fontsize=16)

    labels = ['M1 (Left)', 'M2 (Right)', 'M3 (Base)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 1. Position Plot
    for i in range(3):
        axs[0].plot(time_log, pos_log[i], label=labels[i], color=colors[i], linewidth=2)
    axs[0].set_ylabel('Pos (deg)')
    axs[0].grid(True)
    axs[0].legend(loc="upper right", fontsize='small')
    
    # 2. Velocity Plot
    for i in range(3):
        axs[1].plot(time_log, vel_log[i], color=colors[i], linewidth=2)
    axs[1].set_ylabel('Vel (deg/s)')
    axs[1].grid(True)

    # 3. Acceleration Plot
    for i in range(3):
        axs[2].plot(time_log, acc_log[i], color=colors[i], linewidth=2)
    axs[2].set_ylabel('Acc (deg/s²)')
    axs[2].grid(True)

    # 4. Jerk Plot
    for i in range(3):
        axs[3].step(time_log, jerk_log[i], color=colors[i], linewidth=2, where='post')
    axs[3].set_ylabel('Jerk (deg/s³)')
    axs[3].set_xlabel('Time (seconds)')
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_3dof_trajectory(
        yaw=90, pitch=60, roll=45, 
            v_max=300,
            a_max=50,  
            j_max=30
    )