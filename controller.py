import math


def controller(affordances, prev_affordances, steering_record, state):
    # a controller processes the cnn output and get the optimal steering, acceleration/brake

    angle = affordances[0]
    toMarking_L = affordances[1]
    toMarking_M = affordances[2]
    toMarking_R = affordances[3]
    dist_L = affordances[4]
    dist_R = affordances[5]
    toMarking_LL = affordances[6]
    toMarking_ML = affordances[7]
    toMarking_MR = affordances[8]
    toMarking_RR = affordances[9]
    dist_LL = affordances[10]
    dist_MM = affordances[11]
    dist_RR = affordances[12]
    fast = affordances[13]

    slow_down = 100.0
    pre_dist_L = prev_affordances[10]
    pre_dist_R = prev_affordances[12]

    # TODO:
    road_width, steering_head, timer_set, lane_change, speed = state

    if (pre_dist_L < 20.0 and dist_LL < 20.0):  # left lane is occupied or not
        left_clear = 0
        left_timer = 0
    else:
        left_timer = left_timer + 1

    if (pre_dist_R < 20.0 and dist_RR < 20.0):  # right lane is occupied or not
        right_clear = 0
        right_timer = 0
    else:
        right_timer = right_timer + 1

    if (left_timer > timer_set):  # left lane is clear
        left_timer = timer_set
        left_clear = 1

    if (right_timer > timer_set):  # right lane is clear
        right_timer = timer_set
        right_clear = 1

    if (lane_change == 0 and dist_MM < 15.0):  # if current lane is occupied
        steer_trend = steering_record[0] + steering_record[1] + steering_record[2] + steering_record[3] + steering_record[4]  # am I turning or not

        if (toMarking_LL > -8.0 and left_clear == 1 and steer_trend >= 0 and steer_trend < 0.2):  # move to left lane
            lane_change = -2
            coe_steer = 6
            right_clear = 0
            right_timer = 0
            left_clear = 0
            left_timer = 30
            timer_set = 60

        elif (toMarking_RR < 8.0 and right_clear == 1 and steer_trend <= 0 and steer_trend > -0.2):  # move to right lane
            lane_change = 2
            coe_steer = 6
            left_clear = 0
            left_timer = 0
            right_clear = 0
            right_timer = 30
            timer_set = 60

        else:
            v_max = 20.0
            c = 2.772
            d = -0.693
            slow_down = v_max * (1 - math.exp(-c / v_max * dist_MM - d))  # optimal velocity car-following model
            if (slow_down < 0):
                slow_down = 0

    ######### prefer to stay in the central lane
    elif (lane_change == 0 and dist_MM >= 15):
        steer_trend = steering_record[0] + steering_record[1] + steering_record[2] + steering_record[3] + steering_record[4]  # am I turning or not

        if (toMarking_RR > 8 and left_clear == 1 and steer_trend >= 0 and steer_trend < 0.2):  # in right lane, move to central lane
            lane_change = -2
            coe_steer = 6
            left_clear = 0
            left_timer = 30

        elif (toMarking_LL < -8 and right_clear == 1 and steer_trend <= 0 and steer_trend > -0.2):  # in left lane, move to central lane
            lane_change = 2
            coe_steer = 6
            right_clear = 0
            right_timer = 30

    ########## END prefer to stay in the central lane

    ########### implement lane changing or car-following
    if (lane_change == 0):
        if ((-toMarking_ML + toMarking_MR) < 5.5):
            coe_steer = 1.5
            center_line = (toMarking_ML + toMarking_MR) / 2.0
            pre_ML = toMarking_ML
            pre_MR = toMarking_MR
            if (toMarking_M < 1):
                coe_steer = 0.4
        else:
            if (-pre_ML > pre_MR):
                center_line = (toMarking_L + toMarking_M) / 2.0
            else:
                center_line = (toMarking_R + toMarking_M) / 2.0
            coe_steer = 0.3

    elif (lane_change == -2):
        if ((-toMarking_ML + toMarking_MR) < 5.5):
            center_line = (toMarking_LL + toMarking_ML) / 2.0
            if (toMarking_L > -5 and toMarking_M < 1.5):
                center_line = (center_line + (toMarking_L + toMarking_M) / 2.0) / 2.0
        else:
            center_line = (toMarking_L + toMarking_M) / 2.0
            coe_steer = 20.0
            lane_change = -1

    elif (lane_change == -1):
        if (toMarking_L > -5 and toMarking_M < 1.5):
            center_line = (toMarking_L + toMarking_M) / 2.0
            if ((-toMarking_ML + toMarking_MR) < 5.5):
                center_line = (center_line + (toMarking_ML + toMarking_MR) / 2.0) / 2.0
        else:
            center_line = (toMarking_ML + toMarking_MR) / 2.0
            lane_change = 0

    elif (lane_change == 2):
        if ((-toMarking_ML + toMarking_MR) < 5.5):
            center_line = (toMarking_RR + toMarking_MR) / 2.0
            if (toMarking_R < 5.0 and toMarking_M < 1.5):
                center_line = (center_line + (toMarking_R + toMarking_M) / 2.0) / 2.0
        else:
            center_line = (toMarking_R + toMarking_M) / 2.0
            coe_steer = 20.0
            lane_change = 1

    elif (lane_change == 1):
        if (toMarking_R < 5 and toMarking_M < 1.5):
            center_line = (toMarking_R + toMarking_M) / 2.0
            if ((-toMarking_ML + toMarking_MR) < 5.5):
                center_line = (center_line + (toMarking_ML + toMarking_MR) / 2.0) / 2.0
        else:
            center_line = (toMarking_ML + toMarking_MR) / 2.0
            lane_change = 0

    ############## END implement lane changing or car-following
    # ??? shared->steerCmd
    steerCmd = (angle - (center_line / road_width)) / 0.541052 / coe_steer  # steering control, "steerCmd" [-1,1] is the value sent back to TORCS

    if (lane_change == 0 and coe_steer > 1 and steerCmd > 0.1):   # reshape the steering control curve
        steerCmd = steerCmd * (2.5 * steerCmd + 0.75)

    steering_record[steering_head] = steerCmd  # update previous steering record
    steering_head = steering_head + 1
    if (steering_head == 5):
        steering_head = 0

    if (fast == 1):
        desired_speed = 20.0
    else:
        ssum = steering_record[0] + steering_record[1] + steering_record[2] + steering_record[3] + steering_record[4]
        desired_speed = 20.0 - abs(ssum) * 4.5
    if (desired_speed < 10):
        desired_speed = 10.0

    if (slow_down < desired_speed):
        desired_speed = slow_down

    ############ speed control
    # ??? shared->speed, shared->accelCmd, shared->brakeCmd      
    if (desired_speed >= speed):
        accelCmd = 0.2 * (desired_speed - speed + 1)
        if (accelCmd > 1):
            accelCmd = 1.0
        brakeCmd = 0.0
    else:
        brakeCmd = 0.1 * (speed - desired_speed)
        if (brakeCmd > 1):
            brakeCmd = 1.0
        accelCmd = 0.0

    ########### END speed control

    print("M_LL:%.2lf, M_ML:%.2lf, M_MR:%.2lf, M_RR:%.2lf, d_LL:%.2lf, d_MM:%.2lf, d_RR:%.2lf",
        (toMarking_LL, toMarking_ML, toMarking_MR, toMarking_RR, dist_LL, dist_MM, dist_RR))
    print("M_L:%.2lf, M_M:%.2lf, M_R:%.2lf, d_L:%.2lf, d_R:%.2lf, angle:%.3lf, fast:%d",
        (toMarking_L, toMarking_M, toMarking_R, dist_L, dist_R, angle, fast))
    print("coe_steer:%.1lf, lane_change:%d, steer:%.2lf, d_speed:%d, speed:%d, l_clear:%d, r_clear:%d, timer_set:%d\n",
        (coe_steer, lane_change, steerCmd, math.floor(desired_speed * 3.6), math.floor(speed * 3.6), left_clear, right_clear, timer_set))
    ########## END a controller processes the cnn output and get the optimal steering, acceleration/brake

    action = [steerCmd, brakeCmd, accelCmd]
    return action
